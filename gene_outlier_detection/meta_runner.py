import os
import shutil
import subprocess

import click

from gene_outlier_detection.cli import common_cli


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
    """
    Calls main.py in its own subprocess to avoid absurd Theano crash during compiledir cleanup
    """
    cdir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(cdir, "main.py")
    parameters = [
        "python",
        main_path,
        "-s",
        sample,
        "-b",
        background,
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
    ]
    if disable_iter:
        parameters.append("-d")
    if gene_list:
        parameters.extend(["-l", gene_list])
    subprocess.check_call(parameters)

    # Delete theano dir
    out_dir = os.path.abspath(os.path.join(out_dir, name))
    theano_dir = os.path.join(out_dir, ".theano")
    shutil.rmtree(theano_dir)
