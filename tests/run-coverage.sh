#!/usr/bin/env bash
 pytest \
    --cov-report=html \
    --cov-report=term-missing \
    --cov=gene_outlier_detection/
