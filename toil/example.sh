#!/usr/bin/env bash
# First create/activate your py2.7 environment for toil and have Docker installed
# conda create --name toil python=2.7
# source activate toil
# pip install pandas toil==3.19.0
python toil-outlier-detection.py \
    --sample /data/input.csv \
    --background /data/normal.csv \
    --group tissue \
    --gene-list /data/test-drug-genes.txt \
    --col-skip 5 \
    --num-backgrounds 5 \
    --max-genes 125 \
    --manifest /data/manifest.txt \
    --workDir /scratch/ \
    ./jobStore
