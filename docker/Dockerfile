FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libhdf5-serial-dev build-essential gcc

RUN pip install gene-outlier-detection==0.5.0a

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["outlier-detection"]
CMD ["--help"]
