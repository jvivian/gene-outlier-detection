FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libhdf5-serial-dev build-essential gcc g++
RUN conda install -y mkl-service numpy scipy theano pygpu

RUN pip install gene-outlier-detection==1.0

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["outlier-detection"]
CMD ["--help"]
