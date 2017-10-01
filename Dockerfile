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

# Torch environment variables
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# Install loadcaffe and other torch dependencies
RUN luarocks install loadcaffe

# Clone neural-style app
WORKDIR /
RUN set -ex && \
	wget --no-check-certificate https://github.com/jcjohnson/neural-style/archive/master.tar.gz && \
	tar -xvzf master.tar.gz && \
    mv neural-style-master neural-style && \
	rm master.tar.gz

# Download precomputed network weights
WORKDIR neural-style
RUN bash models/download_models.sh
RUN mkdir /models

# Declare volume for storing network weights
VOLUME ["/neural-style/models"]

# Copy wrapper scripts
COPY ["/scripts/variants.sh", "/scripts/neural-style.sh", "/neural-style/"]

# Add neural-style to path
ENV PATH /neural-style:$PATH

# Prepare folder for mounting images and workplaces
WORKDIR /images
VOLUME ["/images"]

ENTRYPOINT ["neural-style.sh"]
CMD ["-backend", "cudnn", "-cudnn_autotune"]

