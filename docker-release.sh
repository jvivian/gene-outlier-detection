#!/usr/bin/env bash

VERSION=0.16a

# Build and push Docker
docker build -t jvivian/gene-outlier-detection:${VERSION} ./docker
docker push jvivian/gene-outlier-detection:${VERSION}

docker tag jvivian/gene-outlier-detection:${VERSION} jvivian/gene-outlier-detection:latest
docker push jvivian/gene-outlier-detection:latest
