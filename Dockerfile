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

COPY ["/scripts/variants.sh", "/scripts/neural-style.sh", "/neural-style/"]

# Add neural-style to path
ENV PATH /neural-style:$PATH

# Prepare folder for mounting images and workplaces
WORKDIR /images
VOLUME ["/images"]

# Expose API ports
EXPOSE 80

ENTRYPOINT ["neural-style.sh"]
CMD ["-backend", "cudnn", "-cudnn_autotune"]

