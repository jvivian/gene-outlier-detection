#!/usr/bin/env bash
poetry run outlier-detection \
    -s ../gene-outlier-detection/data/input.csv \
    -b ../gene-outlier-detection/data/normal.hdf \
    -n TCGA-DJ-A2PX-01 \
    -l ../gene-outlier-detection/data/test-drug-genes.txt \
    -c 5 \
    -m 11
