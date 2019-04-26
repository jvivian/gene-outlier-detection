import os
import shutil
import time
import warnings
from argparse import Namespace

import click
import pandas as pd
import scipy.stats as st

from gene_outlier_detection.cli import common_cli
from gene_outlier_detection.lib import display_runtime
from gene_outlier_detection.lib import get_sample
from gene_outlier_detection.lib import load_df
from gene_outlier_detection.lib import pca_distances
from gene_outlier_detection.lib import pickle_model
from gene_outlier_detection.lib import posterior_predictive_check
from gene_outlier_detection.lib import posterior_predictive_pvals
from gene_outlier_detection.lib import save_weights
from gene_outlier_detection.lib import select_k_best_genes

warnings.filterwarnings("ignore")


def iter_run(opts: Namespace):
    """
    Run model until P-values converge or num-backgrounds is reached

    :param opts: Namespace object containing CLI variables
    :return: None
    """
    from gene_outlier_detection.lib import save_traceplot

    # Load input data
    click.echo("Loading input data")
    opts.sample = get_sample(opts.sample, opts.name)
    opts.df = load_df(opts.background)
    opts.df = opts.df.sort_values(opts.group)
    opts.genes = opts.df.columns[opts.col_skip :]
    pval_runs_out = os.path.join(opts.out_dir, "_pval_runs.tsv")
    pearson_out = os.path.join(opts.out_dir, "_pearson_correlations.txt")

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
    train_set, model, trace, ppp = None, None, None, None
    for i in range(1, opts.n_bg + 1):
        if opts.disable_iter:
            click.secho(
                f"Performing one run with {i} backgrounds due to disable-iter flag",
                fg="red",
            )
            i = opts.n_bg

        # Execute single model run with i background datasets
        train_set, model, trace, ppp = run(opts, i)

        # Add PPP to DataFrame of all pvalues collected
        pval_runs = pd.concat([pval_runs, ppp], axis=1, sort=True).dropna()
        pval_runs.columns = list(range(len(pval_runs.columns)))
        pval_runs.to_csv(pval_runs_out, sep="\t")

        # Early stop conditions
        if i == 1:
            continue
        if opts.n_bg == 1 or opts.disable_iter:
            break

        # Check Pearson correlation of last two runs
        x, y = pval_runs.columns[-2:]
        pr, _ = st.pearsonr(pval_runs[x], pval_runs[y])
        pearson_correlations.append(str(pr))

        # Output Pearson correlations from run
        with open(pearson_out, "w") as f:
            f.write("\n".join(pearson_correlations))

        # Check if p-values have converged and break out of loop if so
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

    # Traceplot - if there is only one background then b = 1 instead of a Dirichlet RV
    b = True if opts.n_bg > 1 else False
    save_traceplot(trace, opts.out_dir, b=b)

    # Weight plot and weight table if num_backgrounds > 1
    if opts.n_bg > 1:
        classes = train_set[opts.group].unique()
        save_weights(trace, classes, opts.out_dir)

    # Output posterior predictive p-values
    ppp_out = os.path.join(opts.out_dir, "pvals.tsv")
    ppp.to_csv(ppp_out, sep="\t")

    # Save Model
    model_out = os.path.join(opts.out_dir, "model.pkl")
    pickle_model(model_out, model, trace)

    # Move _info files to subdir _info
    output = os.listdir(opts.out_dir)
    info_files = [os.path.join(opts.out_dir, x) for x in output if x.startswith("_")]
    info_dir = os.path.join(opts.out_dir, "_info")
    os.makedirs(info_dir, exist_ok=True)
    [shutil.move(x, info_dir) for x in info_files]


def run(opts: Namespace, num_backgrounds: int):
    """
    Constitutes one model run

    :param opts: Namespace object containing CLI variables
    :param num_backgrounds: Number of background sets to run
    :return: All unique components of a run: training samples, model, trace, and posterior pvalues
    """
    from gene_outlier_detection.lib import run_model

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

    # Run model
    t0 = time.time()
    model, trace = run_model(opts.sample, train_set, training_genes, group=opts.group)
    display_runtime(t0)

    # PPC / PPP
    ppc = posterior_predictive_check(trace, training_genes)
    ppp = posterior_predictive_pvals(opts.sample, ppc)

    return train_set, model, trace, ppp


@click.command()
@common_cli
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
    os.environ["THEANO_FLAGS"] = f"base_compiledir={opts.theano_dir}"
    os.makedirs(opts.theano_dir, exist_ok=True)
    iter_run(opts)


if __name__ == "__main__":
    cli()
