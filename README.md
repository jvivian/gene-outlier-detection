# N-of-1 Gene Outlier Detection
### For RNA-seq Expression Data

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![Build Status](https://travis-ci.com/jvivian/gene-outlier-detection.svg?branch=master)](https://travis-ci.com/jvivian/gene-outlier-detection) [![Coverage Status](https://coveralls.io/repos/github/jvivian/gene-outlier-detection/badge.svg)](https://coveralls.io/github/jvivian/gene-outlier-detection)

This package identifies outliers for gene expression data by building a consensus distribution from background datasets that are informed by an N-of-1 sample. 
See [Model Explanation](#model-explanation) for more information or the [preprint](https://www.biorxiv.org/content/early/2019/06/06/662338.full.pdf).

<p align="center"> 
<img src="/imgs/Experimental-Protocol.png" height="50%" width="50%">
</p>

This workflow takes gene expression data as input and outputs the following:

    SAMPLE_UUID
    ├── _info
    │   ├── _gelman-rubin.tsv
    │   ├── _pearson_correlations.txt
    │   ├── _pval_runs.tsv
    │   └── _run_info.tsv
    ├── model.pkl
    ├── pvals.tsv
    ├── ranks.tsv
    ├── traceplot.png
    ├── weights.png
    └── weights.tsv
- The **_info** subdirectory contains secondary information about the model run
    - **_gelman-rubin.tsv** - TSV of [Gelman-Rubin diagnostic](https://docs.pymc.io/api/diagnostics.html#pymc3.diagnostics.gelman_rubin) for every model parameter, including a median across parameters which should be about ~1.0.
    - **_pearson_correlations.txt** - Single column list of the Pearson correlation of gene p-values between runs (unless `-d` is provided)
    - **_pval_runs.tsv** - Table of gene p-values as background datasets are added
    - **_run_info.tsv** - TSV of software parameters for reproducibility and model runtime
- **model.pkl** — A python pickle of the [PyMC3](https://docs.pymc.io) `model` and `trace`. Model must be run with `--save-model` flag and can be retrieved via
```python
import pickle
with open(pkl_path, 'rb') as buff:
    data = pickle.load(buff)
model, trace = data['model'], data['trace']
```
- **pvals.tsv** — P-values for all genes the model was trained on
- **ranks.tsv** — The median rank of all groups as measured by pairwise euclidean distance
- **traceplot.png** — Traceplot from PyMC3 linear model coefficients and model error
- **weights.png** — Boxplot of model weights for all background datasets
- **weights.tsv** — Mean and SD of model weights as related to background datasets

# Quickstart
1. Install
```bash
pip install --pre gene-outlier-detection
```
2. Download the prerequisite [inputs](https://github.com/jvivian/gene-outlier-detection/wiki/Model-Inputs)
3. Run the model
```bash
outlier-detection --sample /data/tumor.hd5 \
        --background /data/gtex.hd5 \
        --name TCGA-OR-A5KV-01 \
        --gene-list /data/drug-genes.txt \
        --col-skip 5
```

# Dependencies and Installation

This workflow has been tested on ubuntu 18.04 and Mac OSX, but should also run on other unix based systems or under an Anaconda installation.

1. Python 3.6
2. [Docker](https://docs.docker.com/install/) if using the Docker version or Toil workflow version
3. HDF5 library (if inputs are in HDF5 format)
    1. `conda install -c anaconda hdf5`
4. C++ / GCC compiler for PyMC3's Theano
    1. You have a couple options
        1. `conda install theano`
        1. `apt-get update && apt-get install -y libhdf5-serial-dev build-essential gcc`
    
You _may_ need to modify your `~/.theanorc` to support larger bracket depth for this model.
```
[gcc]
cxxflags = -fbracket-depth=1024
```

# Model Explanation

In the following plate notation, G stands for Gene, and D stands for Dataset (background dataset).

<p align="center"> 
<img src="/imgs/Plate-Notation.png" height="50%" width="50%">
</p>

The core of our method is a Bayesian statistical model for the N-of-1 sample’s gene expression.
The model implicitly assumes that the sample’s gene expression can be approximated by a
convex mixture of the gene expression of the background datasets. The coefficients of this
mixture are shared across genes, much like a linear model in which each data point is the
vector of expressions for a gene across the background datasets. In addition, we model
expression for each gene from each background dataset as a random variable itself. This allows
us to incorporate statistical uncertainty from certain background sets’ small sample size directly
in the model without drowning out any background set’s contribution through pooling 

The model can be explored using Markov chain Monte Carlo (MCMC) to obtain samples
for _y_ (per gene) that approximate its posterior distribution. If we have an observed expression value for a
gene of interest (from the N-of-1 cancer sample), we can compare it to the sampled values. The
proportion of sampled values for _y_ that are greater (or lesser) than the observed value is an
estimate of the posterior predictive p-value for this expression value. The posterior predictive
p-value can be seen as a measure of how much of an outlier the expression is given the
expectations of the comparison set

# Defining Inputs

The model requires two **Sample** by **Gene** matrices, one containing the N-of-1 sample and one containing samples to use as the background comparison set. 
They must both contain the same set of genes and the background dataset must contain at least one metadata column with labels that differentiate groups (e.g. tissue, subtype, experiment, etc) at the start of the matrix.

# Docker Container

A Docker container containing the program can be executed as follows:

```bash
docker run --rm -v $(pwd):/data jvivian/gene-outlier-detection \
        outlier-model \
        --sample /data/inputs/tumor.hd5 \
        --background /data/inputs/gtex.hd5 \
        --name=TCGA-OR-A5KV-01 \
        --gene-list /data/inputs/drug-genes.txt \
        --out-dir /data/outputs/ \
        --col-skip=5
```

# Toil-version of Workflow

A [Toil](https://toil.readthedocs.io/) version of the workflow is available [here](https://github.com/jvivian/gene-outlier-detection/blob/master/toil/toil-outlier-detection.py). This allows the model to be run on multiple samples at scale on a cluster or cloud computing cluster, but requires Python 2.7 and `pip install pandas toil==3.19.0`

# Arguments
Explanation of arguments used when running the program.

- `--sample`
    - A path to the matrix (.tsv / .hd5) that contains the sample.
- `--background`
    - A path to the matrix that contains the background datasets and at least one column at the beginning used as the label vector for different groups in the background.
- `--name`
    - Name of row in the sample matrix that corresponds to the desired sample to run.
- `--out-dir`
    - Output directory
- `--gene-list`
    - Single column file of genes for the model to train on and calcluate p-values for. After ~100-200 genes, it is better to split the genes into batches and run in parallel.
- `--col-skip`
    - Number of metadata columns to skip in background matrix. All columns after this number of columns should be genes with expression values. 
- `--group`
    - Name of the categorical column vector in the background matrix which distinguishes the different background datasets.
- `--num-backgrounds`  
    - Maximum number of background categorical groups to include in the model training. Model will run, starting with one background dataset, and iteratively adding more until the p-values converge or `num-backgrounds` is met.
- `--max-genes`
    - Maximum number of genes to run. I.e. if a gene list is provided, how many additional genes using ANOVA. Useful for improving beta coefficients if gene list does not contain enough tissue-specific genes. It is recommended to run the model with `max-genes` set to a minimum of 10-20 more genes than exist in the `--gene-list`.
- `--pval-convergence-cutoff`
    - P-value Pearson correlation cutoff to stop adding additional background datasets.
- `--num-training-genes`
    - If gene-list is empty, will use ANOVA to choose gene set. Not typically useful outside of testing.
- `--tune`
    - Number of tuning steps to start MCMC sampling process. Default is 500, but 750 or 1,000 may be useful in circumstances where the model is having difficulty converging.
- `--disable-iter`
    - This flag disables iterative runs and runs one model with `--num-backgrounds`.
- `--save-model`
    - This flag will save a serialized version of the model/trace. Useful for debugging or inspection of all model parameters.
