#!/usr/bin/env bash

VERSION=0.2.0a

# Build and push Docker
docker build -t jvivian/gene-outlier-detection:${VERSION} ./docker
docker push jvivian/gene-outlier-detection:${VERSION}
