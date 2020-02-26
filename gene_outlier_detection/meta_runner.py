"""
Calls main.py in its own subprocess to avoid absurd Theano crash during compiledir cleanup
"""

import os
import shutil
import subprocess

import click

from gene_outlier_detection.cli import common_cli


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
    """
    \b
    N-of-1 Gene Outlier Detection
    Details: https://github.com/jvivian/gene-outlier-detection

    \b
    This workflow takes gene expression data as input and outputs the following:

    \b
    SAMPLE_UUID
    ├── _info
    │   ├── _gelman-rubin.tsv
    │   ├── _pearson_correlations.txt
    │   ├── _pval_runs.tsv
    │   └── _run_info.tsv
    ├── model.pkl
    ├── pvals.tsv
    ├── ranks.tsv
    ├── traceplot.png
    ├── weights.png
    └── weights.tsv
    """
    cdir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(cdir, "main.py")
    parameters = [
        "python",
        main_path,
        "-s",
        sample_path,
        "-b",
        background_path,
        "-n",
        name,
        "-o",
        out_dir,
        "-g",
        group,
        "-c",
        str(col_skip),
        "-nbg",
        str(n_bg),
        "-m",
        str(max_genes),
        "-ntg",
        str(n_train),
        "-p",
        str(pval_cutoff),
        "--tune",
        str(tune),
    ]
    if disable_iter:
        parameters.append("-d")
    if save_model:
        parameters.append("--save-model")
    if gene_list:
        parameters.extend(["-l", gene_list])
    subprocess.check_call(parameters)

    # Delete theano dir
    out_dir = os.path.abspath(os.path.join(out_dir, name))
    theano_dir = os.path.join(out_dir, ".theano")
    shutil.rmtree(theano_dir)
