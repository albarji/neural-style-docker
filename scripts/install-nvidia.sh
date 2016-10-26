#!/bin/bash
# Based on Jonathan Calmels approachhttps://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2

set -xu

# Install NVIDIA drivers 361.42
sudo apt-get install --no-install-recommends -y gcc make libc-dev
wget -P /tmp http://us.download.nvidia.com/XFree86/Linux-x86_64/361.42/NVIDIA-Linux-x86_64-361.42.run
sudo sh /tmp/NVIDIA-Linux-x86_64-361.42.run --silent

# Install nvidia-docker and nvidia-docker-plugin
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

# Add login user to docker and nvidia-docker groups
sudo usermod -aG docker,nvidia-docker ubuntu

# Install our ML 
git clone https://github.com/albarji/neural-style-docker.git

# Tests install
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi

set +xu

