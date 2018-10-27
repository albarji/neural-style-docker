FROM nvidia/cuda:8.0-cudnn5-devel
MAINTAINER "Álvaro Barbero Jiménez, https://github.com/albarji"

# Install system dependencies
RUN set -ex && \
	apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
	ca-certificates \
	sudo \
	libprotobuf-dev \
	protobuf-compiler \
	wget \
	git \
	&& apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install torch
RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
    bash install-deps
RUN cd /root/torch && ./install.sh
RUN ln -s /root/torch/install/bin/* /usr/local/bin

# Install additional necessary torch dependencies
RUN luarocks install loadcaffe && \
    luarocks install autograd

# Install Python miniconda3 + requirements
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && chmod +x Miniconda3-latest-Linux-x86_64.sh \
  && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
&& rm Miniconda3-latest-Linux-x86_64.sh
COPY pip.txt pip.txt
RUN pip install -r pip.txt && \
    rm -f pip.txt

# Clone neural-style app
WORKDIR /app
RUN set -ex && \
	wget --no-check-certificate https://github.com/jcjohnson/neural-style/archive/master.tar.gz && \
	tar -xvzf master.tar.gz && \
    mv neural-style-master neural-style && \
	rm master.tar.gz

# Download precomputed VGG network weights
WORKDIR /app/neural-style
RUN bash models/download_models.sh

# Add neural-style to path
ENV PATH /app/neural-style:$PATH

# Clone style-swap app
WORKDIR /app
RUN set -ex && \
	wget --no-check-certificate https://github.com/rtqichen/style-swap/archive/master.tar.gz && \
	tar -xvzf master.tar.gz && \
    mv style-swap-master style-swap && \
	rm master.tar.gz
# Link precomputed VGG network weights
RUN rm -rf /app/style-swap/models
RUN ln -s /app/neural-style/models /app/style-swap/models
# Add precomputed inverse network model
ADD models/dec-tconv-sigmoid.t7 /app/style-swap/models/dec-tconv-sigmoid.t7

# Copy wrapper scripts and config files
COPY ["entrypoint.py" ,"/app/entrypoint/"]
COPY ["/neuralstyle/*.py", "/app/entrypoint/neuralstyle/"]
COPY ["gpuconfig.json", "/app/entrypoint/"]

WORKDIR /app/entrypoint
ENTRYPOINT ["python", "/app/entrypoint/entrypoint.py"]

