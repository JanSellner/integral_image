FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid Docker build freeze due to region selection
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt update && apt-get -y install tzdata

# Basic tools
RUN apt update && apt install -y \
     build-essential \
     cmake \
     curl \
     gdb \
     git \
     libopencv-dev \
     rsync \
     tree \
     unzip \
     valgrind \
     vim

# Torch C++
RUN curl -o /libtorch.zip -L -O https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip \
 && unzip /libtorch.zip \
 && rm -f /libtorch.zip

ENV CMAKE_PREFIX_PATH=/libtorch/share/cmake

# Python
ENV PATH="/opt/conda/bin:${PATH}"

RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash Miniforge3-Linux-x86_64.sh -b -p "/opt/conda" \
 && rm -f Miniforge3-Linux-x86_64.sh

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /workspaces/integral_image
