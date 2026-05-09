# ARM64 NVIDIA Linux image.
#
# aihwkit-gpu is not published for linux-aarch64, so this image uses an
# NVIDIA PyTorch CUDA base and builds AIHWKit from source with CUDA enabled.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM --platform=linux/arm64 ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
# NVIDIA PyTorch images set pip constraints that pin packaging==23.2; AIHWKit's
# setup path needs setuptools with packaging.licenses, available in packaging>=24.2.
ENV PIP_CONSTRAINT=

RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates bzip2 build-essential \
    cmake ninja-build libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN PIP_CONSTRAINT= python -m pip install --no-cache-dir --upgrade pip setuptools wheel "packaging>=24.2" && \
    PIP_CONSTRAINT= python -m pip install --no-cache-dir \
    cmake ninja scikit-build pybind11 mypy protobuf scipy requests numpy \
    jupyter matplotlib wandb transformers datasets

ARG AIHWKIT_REF=v1.1.0
ARG CUDA_ARCHITECTURES="80;86;87;89;90"
RUN git clone --depth 1 --branch "${AIHWKIT_REF}" https://github.com/IBM/aihwkit.git /tmp/aihwkit && \
    cd /tmp/aihwkit && \
    python setup.py install -j"$(nproc)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    -DRPU_BLAS=OpenBLAS \
    -DUSE_CUDA=ON \
    -DRPU_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    rm -rf /tmp/aihwkit

WORKDIR /workspace
CMD ["/bin/bash"]
