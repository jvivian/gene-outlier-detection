# N-of-1 Gene Outlier Detection
### For RNA-seq Expression Data

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![Build Status](https://travis-ci.com/jvivian/gene-outlier-detection.svg?branch=master)](https://travis-ci.com/jvivian/gene-outlier-detection) [![Coverage Status](https://coveralls.io/repos/github/jvivian/gene-outlier-detection/badge.svg)](https://coveralls.io/github/jvivian/gene-outlier-detection)

This package identifies outliers for gene expression data by building a consensus distribution from background datasets that are informed by an N-of-1 sample. 
See [Model Explanation](#model-explanation) for more information.

<p align="center"> 
<img src="/imgs/Experimental-Protocol.png" height="50%" width="50%">
</p>

This workflow takes gene expression data as input and outputs the following:

    SAMPLE-UUID
    ├── model.pkl
    ├── pvals.tsv
    ├── ranks.tsv
    ├── traceplot.png
    ├── weights.png
    └── weights.tsv

- **model.pkl** — A python pickle of the [PyMC3](https://docs.pymc.io) `model` and `trace`. Can be retrieved via
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

This workflow has been tested on ubuntu 18.04 and Mac OSX, but should also run on other unix based systems.

1. Python 3.6
2. [Docker](https://docs.docker.com/install/) if using the Docker version or Toil workflow version
3. HDF5 library (if inputs are in HDF5 format)
4. C++ / GCC compiler for PyMC3's Theano 
    1. `apt-get update && apt-get install -y libhdf5-serial-dev build-essential gcc`

# Model Explanation

In the following plate notation, G stands for Gene, and D stands for Dataset (background dataset).

<p align="center"> 
<img src="/imgs/Plate-Notation.png" height="50%" width="50%">
</p>

We build a consensus distribution for a gene by dynamically generating one linear model per gene, where the independent 
variables represent different background dataset’s expression for that gene. 
The beta coefficients, or weights, are shared between linear models to learn the relative contribution of each of the background datasets.
A Dirichlet distribution was chosen as a penalization of the beta coefficients and for its ease of interpretability 
as coefficients are positive and sum to 1. 
To better model outliers, a Laplacian distribution was chosen as the likelihood function for its long-tailed properties. The model is trained for each individual N-of-1 patient using the No-U-turn Markov Chain Monte Carlo sampling process. 

# Defining Custom Inputs

The model requires two **Sample** by **Gene** matrices, one containing the N-of-1 sample and one containing samples to use as the background comparison set. 
They must both contain the same set of genes and the background dataset must contain at least one metadata column with labels that differentiate groups (e.g. tissue, subtype, experiment, etc) at the start of the matrix.

# Docker Container

A Docker container containing the program can be executed as follows:

```bash
docker run --rm -v $(pwd):/data jvivian/gene-outlier-detection \
        outlier-model --sample /data/inputs/tumor.hd5 \
        --background /data/inputs/gtex.hd5 \
        --name=TCGA-OR-A5KV-01 \
        --gene-list /data/inputs/drug-genes.txt \
        --out-dir /data/outputs/ \
        --col-skip=5
```

# Toil-version of Workflow

A [Toil](https://toil.readthedocs.io/) version of the workflow is available [here](https://github.com/jvivian/gene-outlier-detection/blob/master/toil/toil-outlier-model.py). This allows the model to be run on multiple samples at scale on a cluster or cloud computing cluster, but requires Python 2.7. 
