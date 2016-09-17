FROM nvidia/cuda:7.5-cudnn5-devel

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
	git \
	libprotobuf-dev \
	protobuf-compiler

# Install torch
RUN git clone https://github.com/torch/distro.git /torch --recursive && \
	cd /torch && \ 
	bash install-deps && \
	./install.sh
# Add torch to path
ENV PATH /torch/install/bin:$PATH

# Install loadcaffe and other torch dependencies
RUN luarocks install loadcaffe

# wget necessary to download models
RUN apt-get install wget

RUN mkdir app
WORKDIR app
RUN git clone https://github.com/jcjohnson/neural-style.git

