FROM kaixhin/cuda-torch
MAINTAINER "Álvaro Barbero Jiménez, https://github.com/albarji"

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
	git \
	libprotobuf-dev \
	protobuf-compiler \
	wget

# Install loadcaffe and other torch dependencies
RUN luarocks install loadcaffe

# Clone neural-style app
WORKDIR /
RUN git clone https://github.com/jcjohnson/neural-style.git
WORKDIR neural-style

# Download precomputed network weights
RUN bash models/download_models.sh
RUN mkdir /models

# Declare volume for storing network weights
VOLUME ["/neural-style/models"]

# Install python miniconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.0.5-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install python dependencies
RUN conda install scikit-image
RUN pip install connexion

# Copy server app
COPY server /neural-style

# Copy variants app
COPY variants.py /neural-style

ENTRYPOINT ["th", "neural_style.lua"]
CMD ["-backend", "cudnn", "-cudnn_autotune"]

