# Example build command: `DOCKER_BUILDKIT=1 docker build --tag snapatac2:2.5.1-default-py3.11 .`
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
    && python3 -m pip wheel \
        --wheel-dir=/python-wheel-dir \
        "snapatac2==${SNAP_ATAC_VERSION}"

# =================================================================================================
# Second build stage. Results in a roughly 1.38 GB image.
FROM ${BASE_PYTHON_IMAGE}

# https://docs.docker.com/engine/reference/builder/#scope
ARG SNAP_ATAC_VERSION

# Mount our first stage builder-image *temporarily* and install from the compiled .whl files
RUN --mount=type=bind,from=builder-image,source=/python-wheel-dir,target=/python-wheel-dir \
    python3 -m pip install \
        --no-index --no-cache-dir --find-links=/python-wheel-dir \
        "snapatac2==${SNAP_ATAC_VERSION}"
