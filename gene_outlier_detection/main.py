import os
import shutil
import time
import warnings
from argparse import Namespace

import click
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import scipy.stats as st

from gene_outlier_detection.lib import display_runtime
from gene_outlier_detection.lib import get_sample
from gene_outlier_detection.lib import load_df
from gene_outlier_detection.lib import pca_distances
from gene_outlier_detection.lib import pickle_model
from gene_outlier_detection.lib import plot_weights
from gene_outlier_detection.lib import posterior_predictive_check
from gene_outlier_detection.lib import posterior_predictive_pvals
from gene_outlier_detection.lib import select_k_best_genes

warnings.filterwarnings("ignore")


def iter_run(opts: Namespace):
    """
    Run model until P-values converge or num-backgrounds is reached

    :param opts: Namespace object containing CLI variables
    :return: None
    """
    # Load input data
    click.echo("Loading input data")
    opts.sample = get_sample(opts.sample, opts.name)
    opts.df = load_df(opts.background)
    opts.df = opts.df.sort_values(opts.group)
    opts.genes = opts.df.columns[opts.col_skip :]

    # Calculate ranks of background datasets
    opts.ranks = pca_distances(opts.sample, opts.df, opts.genes, opts.group)
    ranks_out = os.path.join(opts.out_dir, "ranks.tsv")
    opts.ranks.to_csv(ranks_out, sep="\t")
    opts.n_bg = opts.n_bg if opts.n_bg < len(opts.ranks) else len(opts.ranks)

    # Parse training genes
    if opts.gene_list is None:
        click.secho(
            f"No gene list provided. Selecting {opts.n_train} genes via SelectKBest (ANOVA F-value)",
            fg="yellow",
        )
        # Select genes based on maximum number of background datasets
        train_set = opts.df[
            opts.df[opts.group].isin(opts.ranks.head(opts.n_bg)["Group"])
        ]
        opts.base_genes = select_k_best_genes(
            train_set, opts.genes, group=opts.group, n=opts.n_train
        )
    else:
        with open(opts.gene_list, "r") as f:
            opts.base_genes = [x.strip() for x in f.readlines() if not x.isspace()]

    # Iteratively run model until convergence or maximum number of training background sets is reached
    pval_runs = pd.DataFrame()
    pearson_correlations = []
    t0 = time.time()
    for i in range(1, opts.n_bg + 1):
        i = opts.n_bg if opts.disable_iter else i
        train_set, model, trace, ppp = run(opts, i)

        # Add PPP to DataFrame of all pvalues collected
        pval_runs = pd.concat([pval_runs, ppp], axis=1, sort=True).dropna()
        pval_runs.columns = list(range(len(pval_runs.columns)))

        # Early stop conditions
        if i == 1:
            continue
        if opts.n_bg == 1 or opts.disable_iter:
            break

        # Check Pearson correlation of last two runs between
        x, y = pval_runs.columns[-2:]
        pr, _ = st.pearsonr(pval_runs[x], pval_runs[y])
        pearson_correlations.append(str(pr))
        if pr > opts.pval_cutoff:
            click.secho(
                f"P-values converged at {pr} across {len(pval_runs)} genes.", fg="green"
            )
            break
        else:
            click.secho(
                f"P-value Pearson correlation currently: {round(pr, 3)} between run {i - 1} and {i}",
                fg="yellow",
            )

    # Total runtime of all iterations of model
    display_runtime(t0, total=True)

    # Output P-value runs
    pval_runs_out = os.path.join(opts.out_dir, "_pval_runs.tsv")
    pval_runs.to_csv(pval_runs_out, sep="\t")

    # Output Pearson correlations from run
    if pearson_correlations:
        pearson_out = os.path.join(opts.out_dir, "_pearson_correlations.txt")
        with open(pearson_out, "w") as f:
            f.write("\n".join(pearson_correlations))

    # Traceplot
    fig, axarr = plt.subplots(3, 2, figsize=(10, 5))
    pm.traceplot(trace, varnames=["a", "b", "eps"], ax=axarr)
    traceplot_out = os.path.join(opts.out_dir, "traceplot.png")
    fig.savefig(traceplot_out)

    # Weight plot and weight table
    classes = train_set[opts.group].unique()
    weight_out = os.path.join(opts.out_dir, "weights.png")
    weights = plot_weights(classes, trace, output=weight_out)
    # Convert weights to summarized information of median and std
    weights = weights.groupby("Class").agg({"Weights": ["median", "std"]})
    weights = weights.sort_values(("Weights", "median"), ascending=False)
    weights.to_csv(os.path.join(opts.out_dir, "weights.tsv"), sep="\t")

    # Output posterior predictive p-values
    ppp_out = os.path.join(opts.out_dir, "pvals.tsv")
    ppp.to_csv(ppp_out, sep="\t")

    # Save Model
    model_out = os.path.join(opts.out_dir, "model.pkl")
    pickle_model(model_out, model, trace)


def run(opts: Namespace, num_backgrounds: int):
    """
    Constitutes one model run

    :param opts: Namespace object containing CLI variables
    :param num_backgrounds: Number of background sets to run
    :return: All unique components of a run: training samples, model, trace, and posterior pvalues
    """
    # Select training set
    click.echo(f"\nSelecting {num_backgrounds} background sets")
    train_set = opts.df[
        opts.df[opts.group].isin(opts.ranks.head(num_backgrounds)["Group"])
    ]
    train_set = train_set.sort_values(opts.group)

    # Pad training genes with additional genes from SelectKBest based on `max-genes` argument
    if len(opts.base_genes) < opts.max_genes:
        diff = opts.max_genes - len(opts.base_genes)
        click.secho(
            f"Adding {diff} genes via SelectKBest (ANOVA F-value) to reach {opts.max_genes} total genes",
            fg="yellow",
        )
        training_genes = opts.base_genes + select_k_best_genes(
            train_set, opts.genes, group=opts.group, n=diff
        )
        training_genes = sorted(set(training_genes))
    else:
        training_genes = opts.base_genes

    # Set env variable for base_compiledir before importing model
    os.environ["THEANO_FLAGS"] = f"base_compiledir={opts.theano_dir}"
    os.makedirs(opts.theano_dir, exist_ok=True)
    from gene_outlier_detection.lib import run_model

    # Run model and output runtime
    t0 = time.time()
    model, trace = run_model(opts.sample, train_set, training_genes, group=opts.group)
    display_runtime(t0)

    # PPC / PPP
    ppc = posterior_predictive_check(trace, training_genes)
    ppp = posterior_predictive_pvals(opts.sample, ppc)

    # Cleanup
    shutil.rmtree(opts.theano_dir)

    return train_set, model, trace, ppp


@click.command()
@click.option(
    "-s",
    "--sample",
    required=True,
    type=str,
    help="Sample(s) by Genes matrix (csv/tsv/hd5)",
)
@click.option(
    "-b",
    "--background",
    required=True,
    type=str,
    help="Samples by Genes matrix with metadata columns first "
    "(including a categorical column that discriminates samples by some category) (csv/tsv/hd5)",
)
@click.option(
    "-n",
    "--name",
    required=True,
    type=str,
    help="Name of row in the sample matrix that corresponds to the desired sample to run",
)
@click.option(
    "-l",
    "--gene-list",
    type=str,
    help="Single column file of genes to train model and derive p-values for",
)
@click.option(
    "-o",
    "--out-dir",
    default="./",
    type=str,
    show_default=True,
    help="Output directory",
)
@click.option(
    "-g",
    "--group",
    default="tissue",
    show_default=True,
    type=str,
    help="Name of the categorical column vector in the background matrix",
)
@click.option(
    "-c",
    "--col-skip",
    default=1,
    show_default=True,
    type=int,
    help="Number of metadata columns to skip in background matrix. All columns after this value should be genes",
)
@click.option(
    "-n",
    "--num-backgrounds",
    "n_bg",
    default=5,
    type=int,
    show_default=True,
    help="Maximum number of background categorical groups to include in the model training. "
    "Model will run starting with one background dataset and iteratively add more until the p-values converge.",
)
@click.option(
    "-m",
    "--max-genes",
    default=125,
    type=int,
    show_default=True,
    help="Maximum number of genes to run. I.e. if a gene list is provided, how many additional genes to add via "
    "SelectKBest. Useful for improving beta coefficients if gene list does not contain enough tissue-specific "
    "genes. A good rule of thumb is to set --max-genes to 1.5 times the number of genes in --gene-list",
)
@click.option(
    "-t",
    "--num-training-genes",
    "n_train",
    default=50,
    type=int,
    show_default=True,
    help="If gene-list is empty, will use SelectKBest to choose gene set. Not typically useful outside of testing.",
)
@click.option(
    "-p",
    "--pval-convergence-cutoff",
    "pval_cutoff",
    default=0.99,
    type=float,
    show_default=True,
    help="P-value Pearson correlation cutoff to stop adding additional background datasets.",
)
@click.option(
    "-d",
    "--disable-iter",
    is_flag=True,
    help="This flag disables iterative runs and runs one model with `--num-backgrounds`",
)
def cli(
    sample,
    background,
    name,
    out_dir,
    group,
    col_skip,
    n_bg,
    gene_list,
    max_genes,
    n_train,
    pval_cutoff,
    disable_iter,
):
    click.clear()
    click.secho("Gene Expression Outlier Detection", fg="green", bg="black", bold=True)

    # Create output directories and begin run
    opts = Namespace(**locals())
    opts.out_dir = os.path.abspath(os.path.join(out_dir, name))
    opts.theano_dir = os.path.join(opts.out_dir, ".theano")
    os.makedirs(opts.theano_dir, exist_ok=True)
    iter_run(opts)
