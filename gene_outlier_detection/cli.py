from typing import Callable

import click


def common_cli(function: Callable) -> Callable:
    """
    Common CLI decorator to avoid duplication of options in meta_runner.py and main.py

    :param function: CLI function
    :return:
        Input function
    """
    function = click.option(
        "--save-model",
        is_flag=True,
        help="This flag will save a serialized PyMC3 model and trace object",
    )(function)
    function = click.option(
        "-d",
        "--disable-iter",
        is_flag=True,
        help="This flag disables iterative runs and runs one model with `--num-backgrounds`",
    )(function)
    function = click.option(
        "-t",
        "--tune",
        default=500,
        type=int,
        help="Number of tuning steps in the MCMC sampling process. "
        "If you get an error asking to increase the number of tune steps, try increasing to 750 or 1,000",
    )(function)
    function = click.option(
        "-ntg",
        "--num-training-genes",
        "n_train",
        default=50,
        type=int,
        show_default=True,
        help="If gene-list is empty, will use SelectKBest to choose gene set. Not typically useful outside of testing.",
    )(function)
    function = click.option(
        "-p",
        "--pval-convergence-cutoff",
        "pval_cutoff",
        default=0.99,
        type=float,
        show_default=True,
        help="P-value Pearson correlation cutoff to stop adding additional background datasets.",
    )(function)
    function = click.option(
        "-m",
        "--max-genes",
        default=125,
        type=int,
        show_default=True,
        help="Maximum number of genes to run. I.e. if a gene list is provided, how many additional genes to add via "
        "SelectKBest. Useful for improving beta coefficients if gene list does not contain enough tissue-specific "
        "genes. A good rule of thumb is to set --max-genes to 1.5 times the number of genes in --gene-list",
    )(function)
    click.option(
        "-nbg",
        "--num-backgrounds",
        "n_bg",
        default=5,
        type=int,
        show_default=True,
        help="Maximum number of background categorical groups to include in the model training. "
        "Model will run, starting with one background dataset, and iteratively adding more until the p-values converge.",
    )(function)
    function = click.option(
        "-g",
        "--group",
        default="tissue",
        show_default=True,
        type=str,
        help="Name of the categorical column vector in the background matrix",
    )(function)
    click.option(
        "-c",
        "--col-skip",
        default=1,
        show_default=True,
        type=int,
        help="Number of metadata columns to skip in background matrix. All columns after this value should be genes",
    )(function)
    function = click.option(
        "-l",
        "--gene-list",
        type=str,
        help="Single column file of genes to train model and derive p-values for",
    )(function)
    function = click.option(
        "-o",
        "--out-dir",
        default="./",
        type=str,
        show_default=True,
        help="Output directory",
    )(function)
    function = click.option(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of row in the sample matrix that corresponds to the desired sample to run",
    )(function)
    function = click.option(
        "-b",
        "--background",
        "background_path",
        required=True,
        type=str,
        help="Path to samples by Genes matrix with metadata columns first "
        "(including a categorical column that discriminates samples by some category) (csv/tsv/hd5)",
    )(function)
    function = click.option(
        "-s",
        "--sample",
        "sample_path",
        required=True,
        type=str,
        help="Path to sample(s) by Genes matrix (csv/tsv/hd5)",
    )(function)
    return function
