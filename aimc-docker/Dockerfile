FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_OVERRIDE_CUDA=12.4

RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates bzip2 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

RUN conda create -y -n aihwkit -c conda-forge -c pytorch -c nvidia \
    python=3.11 \
    aihwkit-gpu \
    pytorch \
    jupyter \
    matplotlib && \
    conda clean -afy

SHELL ["conda", "run", "-n", "aihwkit", "/bin/bash", "-c"]

WORKDIR /workspace
CMD ["/bin/bash"]
