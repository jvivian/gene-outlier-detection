import os
import shutil
import time
import warnings
from argparse import Namespace

import click

from gene_outlier_detection.cli import common_cli
from gene_outlier_detection.lib import Model

warnings.filterwarnings("ignore")


def run(opts: Namespace):
    """
    Run model until p-values converge or num-backgrounds is reached

    :param opts: Namespace object containing CLI variables
    :return: None
    """
    # Declare Model object which contains all information for run
    m = Model(opts)

    # Save background rank information
    m.save_ranks()

    # Iteratively run model until convergence or maximum number of training background sets is reached
    t0 = time.time()
    for i in range(1, m.n_bg + 1):
        if m.disable_iter:
            msg = f"Performing one run with {m.n_bg} backgrounds due to `disable-iter` flag"
            click.secho(msg, fg="red")
            i = opts.n_bg

        # Run model a single time with i background datasets
        m.select_training_set(num_backgrounds=i)
        m.select_training_genes()
        t0_run = time.time()
        m.run_model()
        m.display_runtime(t0_run)

        # Calculate posterior predictive p-values
        m.posterior_predictive_check()
        m.posterior_predictive_pvals()

        # Update per-run P-values and save
        m.update_pvals()
        m.save_pval_runs()

        # Early stop conditions
        if i == 1:
            continue
        if m.n_bg == 1 or m.disable_iter:
            break

        # Check Pearson correlation of last two runs and save
        pr = m.update_pearson_correlations()
        m.save_pearson_correlations()

        # Check if p-values have converged and break out of loop if so
        if pr > m.pval_cutoff:
            msg = f"P-values converged at {pr} across {len(m.pval_runs)} genes."
            click.secho(msg, fg="green")
            break
        else:
            msg = f"P-value Pearson correlation currently: {round(pr, 3)} between run {i - 1} and {i}"
            click.secho(msg, fg="yellow")

    # Save Model output
    m.output_run_info(*m.display_runtime(t0, total=True))
    m.save_traceplot()
    m.save_pvalues()
    m.save_gelman_rubin()
    if m.n_bg > 1:
        m.save_weights()
    if m.save_model:
        m.pickle_model()

    # Move _info files to subdir _info
    output = os.listdir(opts.out_dir)
    info_files = [os.path.join(m.out_dir, x) for x in output if x.startswith("_")]
    info_dir = os.path.join(opts.out_dir, "_info")
    os.makedirs(info_dir, exist_ok=True)
    [shutil.move(x, info_dir) for x in info_files]


@click.command()
@common_cli
def cli(
    sample_path,
    background_path,
    name,
    out_dir,
    group,
    col_skip,
    n_bg,
    gene_list,
    max_genes,
    n_train,
    pval_cutoff,
    tune,
    disable_iter,
    save_model,
):
    click.clear()
    click.secho("Gene Expression Outlier Detection", fg="green", bold=True)

    # Create output directories and begin run
    opts = Namespace(**locals())
    opts.out_dir = os.path.abspath(os.path.join(out_dir, name))
    opts.theano_dir = os.path.join(opts.out_dir, ".theano")
    os.environ["THEANO_FLAGS"] = f"base_compiledir={opts.theano_dir}"
    os.makedirs(opts.theano_dir, exist_ok=True)
    run(opts)


if __name__ == "__main__":
    cli()
