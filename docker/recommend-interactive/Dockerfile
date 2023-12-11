# Example build command: `DOCKER_BUILDKIT=1 docker build --tag snapatac2:2.5.1-recommend-interactive-py3.11 .`
# Use a 2-step build so our final image doesn't include bulky compile tools
ARG BASE_PYTHON_IMAGE=python:3.11-slim
ARG SNAP_ATAC_VERSION=v2.5.1
FROM ${BASE_PYTHON_IMAGE} AS builder-image

# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG SNAP_ATAC_VERSION

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir /python-wheel-dir
WORKDIR /python-wheel-dir

# Install necessary dependencies to create wheels for all snapATAC2 dependencies
RUN apt update \
    && apt install -y \
        build-essential \
        zlib1g-dev

RUN python3 -m pip install wheel \
    && python3 -m pip wheel --wheel-dir=/python-wheel-dir \
        "snapatac2[recommend]==${SNAP_ATAC_VERSION}" \
        jupyterlab

# =================================================================================================
# Second build stage. Results in a roughly 7.03 GB image
# (majority of size come from scvi-tools dependency via PyTorch and Nvidia CUDA packages).
FROM ${BASE_PYTHON_IMAGE}

# https://docs.docker.com/engine/reference/builder/#scope
ARG SNAP_ATAC_VERSION

# Mount our first stage builder-image *temporarily* and install from the compiled .whl files
RUN --mount=type=bind,from=builder-image,source=/python-wheel-dir,target=/python-wheel-dir \
    python3 -m pip install \
        --no-index --no-cache-dir --find-links=/python-wheel-dir \
        "snapatac2[recommend]==${SNAP_ATAC_VERSION}" \
        jupyterlab

RUN useradd --create-home --shell /bin/bash jupyter
USER jupyter
EXPOSE 8888

WORKDIR /home/jupyter/notebooks
ENTRYPOINT ["jupyter", "lab", "--notebook-dir=/home/jupyter/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
