SnapATAC2 Dockerfiles
=====================

These Dockerfiles are provided to help make SnapATAC2 more portable and easy to use. 

The [snapatac2-default Dockerfile](./default/Dockerfile) is intended to be a useful
starting point for running SnapATAC2 at scale (via HPC or in the Cloud).

The [snapatac2-recommend-interactive Dockerfile](./recommend-interactive/Dockerfile)
is intended to provide a ready to use Jupyter Lab notebook environment with SnapATAC2 and
recommended packages (scanpy + scvi-tools) installed.

## Pulling pre-built SnapATAC2 Dockerfiles

- Prebuilt versions of these Dockerfiles can be found at: (TBA)

## Building SnapATAC2 Dockerfiles

### Build Requirements

- This guide assumes you have installed either [Docker Engine](https://docs.docker.com/engine/install/)
  or [Docker Desktop](https://docs.docker.com/get-docker/) which includes Docker Engine.
- The Docker Engine version should be at least 23.0 in order to support Docker [BuildKit](https://docs.docker.com/build/buildkit/) (you can check with `docker --version`)

### Build Instructions

This guide assumes you are building with `docker build` terminal commands

1. To build, change directory to either the [default](./default/) or [recommend-interactive](./snapatac2-recommend-interactive/)

2. Then run the build command:
    - For the default flavor of snapatac2 run:
        `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --tag snapatac2:v2.5.1-default-py3.11 .`
    - For the recommend-interactive flavor of snapatac2 run:
        `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --tag snapatac2:v2.5.1-recommend-interactive-py3.11 .`

> [!NOTE]
> You can also provide `BASE_PYTHON_IMAGE` and `SNAP_ATAC_VERSION` build args to customize the image that gets built.
> As an example, if you want an image with python 3.1.2 and SnapATAC2 v2.5.0 you could run a build command like:
> `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg BASE_PYTHON_IMAGE=python:3.12-slim --build-arg SNAP_ATAC_VERSION=v2.5.0 --tag snapatac2:v2.5.0-default-py3.12 .`

> [!WARNING]
> Depending on the version of BASE_PYTHON_IMAGE and SNAP_ATAC_VERSION, the
> resulting images are *NOT* guaranteed to be well-tested or even functional!

### Run Instructions for `snapatac2:v2.5.1-recommend-interactive-py3.11` Image

Once the image has been built, you can run it with:

`docker run --interactive --tty --rm --publish 8888:8888 --volume <path_to_notebooks_on_your_local_machine>:/home/jupyter/notebooks snapatac2:v2.5.1-recommend-interactive-py3.11`

You can then navigate in your browser to the `http://127.0.0.1:8888/lab?token=<jupyter-lab-token>` link to access Jupyter Lab

> [!NOTE]
> You can learn more about the `docker run` command options from the [official Docker documentation](https://docs.docker.com/engine/reference/commandline/run/#usage)
