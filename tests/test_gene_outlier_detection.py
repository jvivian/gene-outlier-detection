import os
from distutils import dir_util

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def datadir(tmpdir):
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    dir_util.copy_tree(datadir, str(tmpdir))
    return str(tmpdir)


@pytest.fixture
def load_data(datadir):
    from gene_outlier_detection.lib import load_df, get_sample

    df_path = os.path.join(datadir, "normal.tsv")
    sample_path = os.path.join(datadir, "input.tsv")
    df = load_df(df_path)
    genes = df.columns[5:]
    sample = get_sample(sample_path, "TCGA-DJ-A2PX-01")
    return sample, df, genes


@pytest.fixture
def model_output(load_data):
    import warnings

    warnings.filterwarnings("ignore")
    from gene_outlier_detection.lib import run_model, select_k_best_genes

    sample, df, genes = load_data
    training_genes = select_k_best_genes(df, genes, n=10)
    return run_model(sample, df, training_genes)


@pytest.fixture
def ppc(model_output, load_data):
    from gene_outlier_detection.lib import (
        posterior_predictive_check,
        select_k_best_genes,
    )

    sample, df, genes = load_data
    training_genes = select_k_best_genes(df, genes, n=10)
    m, t = model_output
    return posterior_predictive_check(t, training_genes)


@pytest.fixture
def load_opts(tmpdir, load_data, datadir):
    from argparse import Namespace
    from gene_outlier_detection.lib import pca_distances
    from gene_outlier_detection.lib import select_k_best_genes

    # Get paths
    df_path = os.path.join(datadir, "normal.tsv")
    sample_path = os.path.join(datadir, "input.tsv")
    # Build opts object
    opts = Namespace()
    sample, df, genes = load_data
    opts.sample = sample_path
    opts.background = df_path
    opts.df = df
    opts.group = "tissue"
    opts.name = "TCGA-DJ-A2PX-01"
    opts.n_train = 10
    opts.max_genes = 10
    opts.pval_cutoff = 0.99
    opts.col_skip = 5
    opts.n_bg = 2
    opts.gene_list = None
    opts.disable_iter = None
    opts.genes = opts.df.columns[opts.col_skip :]
    opts.out_dir = tmpdir
    opts.theano_dir = os.path.join(opts.out_dir, ".theano")
    return opts


def test_select_k_best_genes(datadir):
    from gene_outlier_detection.lib import select_k_best_genes
    import warnings

    warnings.filterwarnings("ignore")
    df = pd.read_csv(os.path.join(datadir, "normal.tsv"), sep="\t", index_col=0)
    genes = df.columns[5:]
    assert select_k_best_genes(df, genes, n=5) == [
        "AP1M2",
        "RP4-568C11.4",
        "MXRA5",
        "TSHR",
        "GRM3",
    ]


def test_get_sample(datadir):
    from gene_outlier_detection.lib import get_sample

    sample_path = os.path.join(datadir, "input.tsv")
    sample = get_sample(sample_path, "TCGA-DJ-A2PX-01")
    assert sample.shape[0] == 26549
    assert sample.tissue == "Thyroid"


def test_load_df(datadir):
    from gene_outlier_detection.lib import load_df

    df_path = os.path.join(datadir, "normal.tsv")
    df = load_df(df_path)
    assert df.shape == (10, 26549)


def test_pca_distances(load_data):
    from gene_outlier_detection.lib import pca_distances

    sample, df, genes = load_data
    dist = pca_distances(sample, df, genes)
    assert list(dist.Group) == ["Thyroid", "Brain"]
    assert [int(x) for x in dist.MedianDistance] == [169, 267]


def test_run_model(model_output):
    m, t = model_output
    assert "b" in t.varnames


def test_calculate_weights(model_output):
    from gene_outlier_detection.lib import calculate_weights

    m, t = model_output
    weights = calculate_weights(["Thyroid", "Brain"], t)
    assert list(weights.Class.unique()) == ["Thyroid", "Brain"]


def test_plot_weights(tmpdir, model_output):
    from gene_outlier_detection.lib import plot_weights

    output = os.path.join(tmpdir, "plot.png")
    m, t = model_output
    plot_weights(["Thryoid", "Brain"], t, output)
    assert os.path.exists(output)


def test_posterior_predictive_check(ppc):
    assert len(ppc.keys()) == 10
    assert len(ppc[list(ppc.keys())[0]]) == 1000


def test__gene_ppc(model_output):
    from gene_outlier_detection.lib import _gene_ppc

    m, t = model_output
    assert len(_gene_ppc(t, "PAX8")) == 1000


def test_posterior_predictive_pvals(load_data, ppc):
    from gene_outlier_detection.lib import posterior_predictive_pvals

    sample, df, genes = load_data
    ppp = posterior_predictive_pvals(sample, ppc)
    assert ppp.shape == (10, 1)

    genes_in_model = {
        "DSG2",
        "AP1M2",
        "CDH1",
        "TSHR",
        "CTD-2182N23.1",
        "RP4-568C11.4",
        "MXRA5",
        "GRM3",
        "PAX8",
        "CCL21",
    }
    inter = set(ppp.index).intersection(genes_in_model)
    assert len(inter) == 10


def test__ppp_one_gene():
    from gene_outlier_detection.lib import _ppp_one_gene
    import numpy as np

    z = np.array(range(10))
    z_true = 5
    assert _ppp_one_gene(z_true, z) == 0.4


def test_pickle_model(tmpdir, model_output):
    from gene_outlier_detection.lib import pickle_model

    m, t = model_output
    out = os.path.join(tmpdir, "model.pkl")
    pickle_model(out, m, t)
    assert os.path.exists(out)


def test_main(datadir):
    from gene_outlier_detection.main import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--sample",
            os.path.join(datadir, "input.tsv"),
            "--background",
            os.path.join(datadir, "normal.tsv"),
            "--name",
            "TCGA-DJ-A2PX-01",
            "--out-dir",
            datadir,
            "--group",
            "tissue",
            "--col-skip",
            "5",
            "--num-backgrounds",
            "2",
            "--max-genes",
            "10",
            "--num-training-genes",
            "10",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(datadir, "TCGA-DJ-A2PX-01"))


def test_display_runtime():
    from gene_outlier_detection.lib import display_runtime
    import time

    t0 = time.time() - 300
    runtime, unit = display_runtime(t0)
    assert unit == "min"
    assert int(runtime) == 5
    t0 = time.time() - 3600
    runtime, unit = display_runtime(t0)
    assert unit == "hr"
    assert int(runtime) == 1


def test_run(load_opts):
    from gene_outlier_detection.main import run
    from gene_outlier_detection.lib import get_sample
    from gene_outlier_detection.lib import pca_distances
    from gene_outlier_detection.lib import select_k_best_genes

    opts = load_opts
    opts.sample = get_sample(opts.sample, opts.name)
    opts.ranks = pca_distances(opts.sample, opts.df, opts.genes, opts.group)
    train_set = opts.df[opts.df[opts.group].isin(opts.ranks.head(opts.n_bg)["Group"])]
    opts.base_genes = select_k_best_genes(
        train_set, opts.genes, opts.group, opts.n_train
    )
    os.makedirs(opts.theano_dir, exist_ok=True)
    run(opts, 1)


def test_iter_run(load_opts):
    from gene_outlier_detection.main import iter_run

    opts = load_opts
    os.makedirs(opts.theano_dir, exist_ok=True)
    iter_run(opts)
