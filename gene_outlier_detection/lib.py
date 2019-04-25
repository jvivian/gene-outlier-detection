import os
import pickle
import time
from typing import List, Dict, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import pairwise_distances
from tqdm.autonotebook import tqdm


def select_k_best_genes(
    df: pd.DataFrame, genes: List[str], group="tissue", n=50
) -> List[str]:
    """
    Select K genes based on ANOVA F-value

    Args:
        df: Background dataset
        genes: Genes to include in selection process
        group: Column in dataset that distinguishes groups
        n: Number of genes to select (k)

    Returns:
        List of selected genes
    """
    k = SelectKBest(k=n)
    k.fit_transform(df[genes], df[group])
    return [genes[i] for i in k.get_support(indices=True)]


def get_sample(df_path: str, sample_name: str) -> pd.Series:
    """
    Loads dataframe containing sample and returns sample

    Args:
        df_path: Path to DataFrame containing sample
        sample_name: Name of sample in the index of the DataFrame

    Returns:
        Sample vector
    """
    if df_path.endswith(".csv"):
        df = pd.read_csv(df_path, index_col=0)
    elif df_path.endswith(".tsv"):
        df = pd.read_csv(df_path, sep="\t", index_col=0)
    else:
        try:
            df = pd.read_hdf(df_path)
        except Exception as e:
            print(e)
            raise RuntimeError(f"Failed to open DataFrame: {df_path}")

    if sample_name in df.index:
        return df.loc[sample_name]
    else:
        raise RuntimeError(
            f"Sample {sample_name} not located in index of DataFrame {df_path}"
        )


def load_df(df_path: str) -> pd.DataFrame:
    """
    Load background DataFrame

    Args:
        df_path: Path to DataFrame

    Returns:
        Background DataFrame
    """
    if df_path.endswith(".csv"):
        df = pd.read_csv(df_path, index_col=0)
    elif df_path.endswith(".tsv"):
        df = pd.read_csv(df_path, sep="\t", index_col=0)
    else:
        try:
            df = pd.read_hdf(df_path)
        except Exception as e:
            print(e)
            raise RuntimeError(f"Failed to open DataFrame: {df_path}")
    return df


def pca_distances(
    sample: pd.Series,
    df: pd.DataFrame,
    genes: List[str],
    group: str = "tissue",
    n_components=25,
):
    """
    Runs PCA to create an embedding of the sample and background dataset, then calculates distance to each
    group via pairwise distance

    Args:
        sample: n-of-1 sample. Gets own label
        df: background dataset
        genes: genes to use for pairwise distance
        group: Column to use as class discriminator
        n_components: Number of components to use in PCA. 25 is the inflection point for GTEx

    Returns:
        DataFrame of pairwise distances
    """
    # Check n_components value which must be between 0 and min(n_samples, n_features)
    n_samples, n_features = df.shape
    if n_components >= min(n_samples, n_features):
        n_components = min(n_samples, n_features)
        click.secho(f"Number of components changed to {n_components}", fg="yellow")

    # Concatenate sample to background
    concat = df.append(sample)

    # PCA embedding
    embedding = pd.DataFrame(
        PCA(n_components=n_components).fit_transform(concat[genes])
    )
    embedding[group] = concat[group].values
    components = list(embedding.columns[:-1])

    # Separate sample and background
    sample = embedding.iloc[-1]
    df = embedding.iloc[:-1]

    # Pairwise distances
    dist = pairwise_distances(
        np.array(sample[components]).reshape(1, -1), df[components]
    )
    dist = pd.DataFrame([dist.ravel(), df[group]]).T
    dist.columns = ["Distance", "Group"]

    # Median by group and sort
    med_dist = (
        dist.groupby("Group").apply(lambda x: x["Distance"].median()).reset_index()
    )
    med_dist.columns = ["Group", "MedianDistance"]
    return med_dist.sort_values("MedianDistance").reset_index(drop=True)


def run_model(
    sample: pd.Series,
    df: pd.DataFrame,
    training_genes: List[str],
    group: str = "tissue",
    **kwargs,
):
    """
    Run Bayesian model using prefit Y's for each Gene and Dataset distribution

    Args:
        sample: N-of-1 sample to run
        df: Background dataframe to use in comparison
        training_genes: Genes to use during training
        group:
        **kwargs:

    Returns:
        Model and Trace from PyMC3
    """
    # Importing here since Theano base_compiledir needs to be set prior to import
    import pymc3 as pm

    classes = sorted(df[group].unique())
    df = df[[group] + training_genes]

    # Collect fits
    ys = {}
    for gene in training_genes:
        for i, dataset in enumerate(classes):
            # "intuitive" prior parameters
            prior_mean = 0.0
            prior_std_dev = 1.0
            pseudocounts = 1.0

            # convert to prior params of normal-inverse gamma
            kappa_0 = pseudocounts
            mu_0 = prior_mean
            alpha_0 = 0.5 * pseudocounts
            beta_0 = 0.5 / prior_std_dev ** 2

            # collect summary statistics for data
            observations = np.array(df[df[group] == dataset][gene])
            n = len(observations)
            obs_sum = np.sum(observations)
            obs_mean = obs_sum / n
            obs_ssd = np.sum(np.square(observations - obs_mean))

            # compute the posterior params
            kappa_n = kappa_0 + n
            mu_n = (kappa_0 * mu_0 + obs_sum) / (kappa_0 + n)
            alpha_n = alpha_0 + 0.5 * n
            beta_n = beta_0 + 0.5 * (
                obs_ssd + kappa_0 * n * (obs_mean - mu_0) ** 2 / (kappa_0 + n)
            )

            # from https://www.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf, equation (110)
            # convert to the params of a PyMC student-t (i.e. integrate out the prior)
            mu = mu_n
            nu = 2.0 * alpha_n
            lambd = alpha_n * kappa_n / (beta_n * (kappa_n + 1.0))

            ys[f"{gene}={dataset}"] = (mu, nu, lambd)

    click.echo("Building model")
    with pm.Model() as model:
        # Linear model priors
        a = pm.Normal("a", mu=0, sd=1)
        b = [1] if len(classes) == 1 else pm.Dirichlet("b", a=np.ones(len(classes)))
        # Model error
        eps = pm.InverseGamma("eps", 2.1, 1)

        # Linear model declaration
        for gene in tqdm(training_genes):
            mu = a
            for i, dataset in enumerate(classes):
                name = f"{gene}={dataset}"
                m, nu, lambd = ys[name]
                y = pm.StudentT(name, nu=nu, mu=m, lam=lambd)
                mu += b[i] * y

            # Embed mu in laplacian distribution
            pm.Laplace(gene, mu=mu, b=eps, observed=sample[gene])
        # Sample
        trace = pm.sample(**kwargs)
    return model, trace


def calculate_weights(groups: List[str], trace) -> pd.DataFrame:
    """
    Calculates weights of the background groups by examining the beta coefficient in the trace

    Args:
        groups: List of groups trained in the model
        trace: PyMC3 Trace

    Returns:
        DataFrame of classes and weights
    """
    class_col = []
    for c in groups:
        class_col.extend([c for _ in range(len(trace["a"]))])

    weight_by_class = pd.DataFrame(
        {
            "Class": class_col,
            "Weights": np.array([trace["b"][:, x] for x in range(len(groups))]).ravel(),
        }
    )
    return weight_by_class


def plot_weights(groups: List[str], trace, output: str = None) -> pd.DataFrame:
    """
    Plot model coefficients associated with each group

    Args:
        groups: List of groups trained in the model
        trace: PyMC3 Trace
        output: Optional output location for plot
    """
    # Construct weight by class DataFrame
    weights = calculate_weights(groups, trace)

    plt.figure(figsize=(12, 4))
    sns.boxplot(data=weights, x="Class", y="Weights")
    plt.xticks(rotation=90)
    plt.title("Median Beta Coefficient Weight by Tissue for N-of-1 Sample")
    if output:
        plt.savefig(output, bbox_inches="tight")
    return weights


def posterior_predictive_check(trace, genes: List[str]) -> Dict[str, np.array]:
    """
    Posterior predictive check for a list of genes trained in the model

    Args:
        trace: PyMC3 trace
        genes: List of genes of interest

    Returns:
        Dictionary of [genes, array of posterior sampling]
    """
    d = {}
    for gene in genes:
        d[gene] = _gene_ppc(trace, gene)
    return d


def _gene_ppc(trace, gene: str) -> np.array:
    """
    Calculate posterior predictive for a gene

    Args:
        trace: PyMC3 Trace
        gene: Gene of interest

    Returns:
        Random variates representing PPC of the gene
    """
    y_gene = [x for x in trace.varnames if x.startswith(f"{gene}=")]
    b = trace["a"]
    if "b" in trace.varnames:
        for i, y_name in enumerate(y_gene):
            b += trace["b"][:, i] * trace[y_name]

    # If no 'b' in trace.varnames then there was only one comparison group
    else:
        for i, y_name in enumerate(y_gene):
            b += 1 * trace[y_name]
    return np.random.laplace(loc=b, scale=trace["eps"])


def posterior_predictive_pvals(
    sample: pd.Series, ppc: Dict[str, np.array]
) -> pd.DataFrame:
    """
    Produces Series of posterior p-values for all genes in the Posterior Predictive Check (PPC) dictionary

    Args:
        sample: N-of-1 sample
        ppc: Posterior predictive check dictionary

    Returns:
        Genes and their posterior p-value
    """
    pvals = {}
    for gene in ppc:
        z_true = sample[gene]
        z = st.laplace.rvs(*st.laplace.fit(ppc[gene]), size=100_000)
        pvals[gene] = _ppp_one_gene(z_true, z)
    ppp = pd.DataFrame(pvals.items(), columns=["Gene", "Pval"]).sort_values("Pval")
    return ppp.set_index("Gene", drop=True)


def _ppp_one_gene(z_true, z):
    """Calculates ppp for one gene"""
    return round(np.sum(z_true < z) / len(z), 5)


def pickle_model(output_path: str, model, trace):
    """Pickles PyMC3 model and trace"""
    with open(output_path, "wb") as buff:
        pickle.dump({"model": model, "trace": trace}, buff)


def display_runtime(t0: float, total=False) -> Tuple[float, str]:
    """
    Displays runtime given an initial timepoint

    :param t0: The initial time point generated via time.time()
    :param total: If this constitutes the total runtime over all models
    :return: runtime and unit of the runtime (min / hr)
    """
    runtime = round((time.time() - t0) / 60, 2)
    unit = "min" if runtime < 60 else "hr"
    runtime = runtime if runtime < 60 else round(runtime / 60, 2)
    msg = "Total runtime over all models" if total else "Model runtime"
    click.secho(f"{msg}: {runtime} ({unit})", fg="green")
    return runtime, unit


def save_traceplot(trace, out_dir: str, b: bool = True) -> None:
    """
    Saves traceplot of PyMC3 run

    :param trace: PyMC3 trace
    :param out_dir: Output directory of plot
    :param b: Boolean indicating whether there is a beta parameter
    """
    import pymc3 as pm

    if b:
        fig, axarr = plt.subplots(3, 2, figsize=(10, 5))
        varnames = ["a", "b", "eps"]
    else:
        fig, axarr = plt.subplots(2, 2, figsize=(10, 5))
        varnames = ["a", "eps"]
    pm.traceplot(trace, varnames=varnames, ax=axarr)
    traceplot_out = os.path.join(out_dir, "traceplot.png")
    fig.savefig(traceplot_out)
    # Save values
    trace_vals = {}
    trace_vals["median"] = [np.median(trace["a"]), np.median(trace["eps"])]
    trace_vals["std"] = [np.std(trace["a"]), np.std(trace["eps"])]
    trace_df = pd.DataFrame(trace_vals, index=["a", "eps"])
    trace_df.to_csv(os.path.join(out_dir, "_parameters.tsv"), sep="\t")


def save_weights(trace, groups: List[str], out_dir: str) -> None:
    """
    Save weights as both a table and plot

    :param trace: PyMC3 trace
    :param groups: List of groups trained in the model
    :param out_dir: Output directory for weights
    """
    weight_out = os.path.join(out_dir, "weights.png")
    weights = plot_weights(groups, trace, output=weight_out)
    # Convert weights to summarized information of median and std
    weights = weights.groupby("Class").agg({"Weights": ["median", "std"]})
    weights = weights.sort_values(("Weights", "median"), ascending=False)
    weights.columns = ["Median", "std"]
    weights.index.name = None
    weights.to_csv(os.path.join(out_dir, "weights.tsv"), sep="\t")
