FROM kaixhin/cuda-torch
MAINTAINER "Álvaro Barbero Jiménez, https://github.com/albarji"

# Install system dependencies
RUN set -ex && \
	apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
	libprotobuf-dev \
	protobuf-compiler \
	wget \
	&& rm -rf /var/lib/apt/lists/*

# Install loadcaffe and other torch dependencies
RUN luarocks install loadcaffe

# Clone neural-style app
WORKDIR /
RUN set -ex && \
	wget --no-check-certificate https://github.com/jcjohnson/neural-style/archive/master.tar.gz && \
	tar -xvzf master.tar.gz && \
    mv neural-style-master neural-style && \
	rm master.tar.gz
WORKDIR neural-style

# Download precomputed network weights
RUN bash models/download_models.sh
RUN mkdir /models

# Declare volume for storing network weights
VOLUME ["/neural-style/models"]

# Install python miniconda
RUN set -ex && \
	echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.0.5-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Install python dependencies
RUN conda install scikit-image
RUN pip install connexion

COPY ["server", "/scripts/variants.py", "/scripts/neural-style.sh", "/neural-style"]

# Add neural-style to path
ENV PATH /neural-style:$PATH

# Prepare folder for mounting images and workplaces
WORKDIR /images
VOLUME ["/images"]

# Expose API ports
EXPOSE 80

ENTRYPOINT ["neural-style.sh"]
CMD ["-backend", "cudnn", "-cudnn_autotune"]

