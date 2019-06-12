import os
from argparse import Namespace

import pytest
from click.testing import CliRunner


@pytest.fixture(scope="session")
def data_dir():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    return data_dir


@pytest.fixture(scope="session")
def model(data_dir):
    from gene_outlier_detection.lib import Model

    opts = Namespace()
    opts.sample_path = os.path.join(data_dir, "input.tsv")
    opts.background_path = os.path.join(data_dir, "normal.tsv")
    opts.name = "TCGA-DJ-A2PX-01"
    opts.out_dir = data_dir
    opts.group = "tissue"
    opts.col_skip = 5
    opts.n_bg = 2
    opts.gene_list = os.path.join(data_dir, "test-drug-genes.txt")
    opts.max_genes = 11
    opts.n_train = 10
    opts.pval_cutoff = 0.99
    opts.disable_iter = False
    return Model(opts)


@pytest.fixture(scope="session")
def tr_model(model):
    # First run
    model.select_training_set(1)
    model.select_training_genes()
    model.run_model()
    model.posterior_predictive_check()
    model.posterior_predictive_pvals()
    model.update_pvals()
    model.update_pearson_correlations()
    # Second run
    model.select_training_set(2)
    model.select_training_genes()
    model.run_model()
    model.posterior_predictive_check()
    model.posterior_predictive_pvals()
    model.update_pvals()
    model.update_pearson_correlations()
    # Now that run is over, calculate weights
    model.calculate_weights()
    return model


@pytest.fixture(scope="session")
def params(data_dir):
    return [
        "--sample",
        os.path.join(data_dir, "input.tsv"),
        "--background",
        os.path.join(data_dir, "normal.tsv"),
        "--name",
        "TCGA-DJ-A2PX-01",
        "--col-skip",
        "5",
        "--num-backgrounds",
        "2",
        "--max-genes",
        "10",
        "--gene-list",
        os.path.join(data_dir, "test-drug-genes.txt"),
        "--num-training-genes",
        "10",
    ]


def test_load_df(model, data_dir):
    df_path = os.path.join(data_dir, "normal.csv")
    df = model.load_df(df_path)
    assert df.shape == (10, 26549)


def test_get_sample(model):
    sample = model.get_sample()
    assert sample.shape[0] == 26549
    assert sample.tissue == "Thyroid"


def test_select_k_best_genes(model):
    import warnings

    warnings.filterwarnings("ignore")
    assert model.select_k_best_genes(model.df, n=5) == [
        "AP1M2",
        "RP4-568C11.4",
        "MXRA5",
        "TSHR",
        "GRM3",
    ]


def test_anova_distances(model):
    dist = model.anova_distances(percent_genes=0.10)
    assert list(dist.Group) == ["Thyroid", "Brain"]
    assert [int(x) for x in dist.MedianDistance] == [63, 142]


def test_save_ranks(tmpdir, model):
    model.out_dir = tmpdir
    model.save_ranks()
    assert os.path.exists(os.path.join(tmpdir, "ranks.tsv"))


def test_parse_gene_list(model):
    assert len(model.initial_genes) == 10
    model.gene_list = None
    assert len(model.parse_gene_list()) == 10


def test_select_training_set(model):
    model.select_training_set(2)
    assert len(model.backgrounds) == 2


def test_select_training_genes(model):
    model.select_training_set(2)
    model.select_training_genes()
    assert len(model.training_genes) == model.max_genes


def test_run_model(tr_model):
    assert "b" in tr_model.trace.varnames


def test_t_fits(tr_model):
    assert "JAK1=Thyroid" in tr_model.fits
    assert len(tr_model.fits["JAK1=Thyroid"]) == 4


def test_posterior_predictive_check(tr_model):
    ppc = tr_model.ppc
    assert len(ppc.keys()) == 10
    assert len(ppc[list(ppc.keys())[0]]) == 1000


def test__gene_ppc(tr_model):
    assert len(tr_model._gene_ppc("JAK1")) == 1000


def test_posterior_predictive_pvals(data_dir, tr_model):
    ppp = tr_model.ppp
    assert ppp.shape == (10, 1)
    inter = set(ppp.index).intersection(set(tr_model.training_genes))
    assert len(inter) == 10


def test_update_pvals(tmpdir, tr_model):
    # Assertions
    tr_model.update_pvals()
    assert len(tr_model.pval_runs.columns) == 3


def test_save_pval_runs(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.save_pval_runs()
    assert os.path.exists(os.path.join(tmpdir, "_pval_runs.tsv"))


def test_update_pearson_correlations(tmpdir, tr_model):
    tr_model.update_pearson_correlations()
    assert len(tr_model.pearson_correlations) == 2


def test_save_pearson_correlations(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.save_pearson_correlations()
    path = os.path.join(tmpdir, "_pearson_correlations.txt")
    assert os.path.exists(path)
    tr_model.pearson_correlations = []
    assert tr_model.save_pearson_correlations() is None


def test_calculate_weights(tr_model):
    assert sorted(tr_model.weights.Class.unique()) == ["Brain", "Thyroid"]


def test_plot_weights(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.plot_weights()
    output = os.path.join(tmpdir, "weights.png")
    assert os.path.exists(output)


def test_save_weights(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.weights = None
    tr_model.save_weights()
    os.path.exists(os.path.join(tmpdir, "weights.tsv"))


def test_save_traceplot(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.save_traceplot()
    assert os.path.exists(os.path.join(tmpdir, "traceplot.png"))


def test_pickle_model(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.pickle_model()
    out = os.path.join(tmpdir, "model.pkl")
    assert os.path.exists(out)


def test_output_run_info(tmpdir, tr_model):
    tr_model.out_dir = tmpdir
    tr_model.output_run_info(50, "minutes")
    path = os.path.join(tmpdir, "_run_info.tsv")
    assert os.path.exists(path)


def test_meta_runner(params, tmpdir, data_dir):
    from gene_outlier_detection.meta_runner import cli

    params.extend(["--out-dir", tmpdir])

    runner = CliRunner()
    result = runner.invoke(cli, params, catch_exceptions=False)
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(tmpdir, "TCGA-DJ-A2PX-01"))


def test_meta_runner_disable(params, tmpdir, data_dir):
    from gene_outlier_detection.meta_runner import cli

    params.extend(["--out-dir", tmpdir, "-d", "--save-model"])
    runner = CliRunner()
    result = runner.invoke(cli, params, catch_exceptions=False)
    assert result.exit_code == 0
    assert os.path.exists(os.path.join(tmpdir, "TCGA-DJ-A2PX-01"))


def test_display_runtime(model):
    import time

    t0 = time.time() - 300
    runtime, unit = model.display_runtime(t0)
    assert unit == "min"
    assert int(runtime) == 5
    t0 = time.time() - 3600
    runtime, unit = model.display_runtime(t0)
    assert unit == "hr"
    assert int(runtime) == 1


def test_missing_sample(data_dir, model):
    model.name = "FOO"
    with pytest.raises(RuntimeError):
        model.get_sample()


def test_bad_extension(data_dir, model):
    sample_path = os.path.join(data_dir, "input.foo")
    with pytest.raises(RuntimeError):
        model.load_df(sample_path)
