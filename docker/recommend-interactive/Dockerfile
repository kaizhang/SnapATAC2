# Example build command: `DOCKER_BUILDKIT=1 docker build --tag snapatac2:2.6.0-recommend-interactive-py3.11 .`
# Use a 2-step build so our final image doesn't include bulky compile tools
ARG BASE_PYTHON_IMAGE=python:3.11-slim
ARG SNAP_ATAC_VERSION=v2.6.0
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

# Install gosu to gracefully step-down from root in a Docker context
# https://github.com/tianon/gosu/tree/master
RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
    # verify that the binary works
	gosu nobody true

# Mount our first stage builder-image *temporarily* and install from the compiled .whl files
RUN --mount=type=bind,from=builder-image,source=/python-wheel-dir,target=/python-wheel-dir \
    python3 -m pip install \
        --no-index --no-cache-dir --find-links=/python-wheel-dir \
        "snapatac2[recommend]==${SNAP_ATAC_VERSION}" \
        jupyterlab

# Use entrypoint that helps us step-down from root
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Setup SnapATAC2 data download directory
RUN mkdir -p /data
ENV SNAP_DATA_DIR=/data

WORKDIR /notebooks
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
EXPOSE 8888
CMD ["jupyter", "lab", "--notebook-dir=/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
