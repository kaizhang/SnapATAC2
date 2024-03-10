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
        `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --tag snapatac2:v2.6.0-default-py3.11 .`
    - For the recommend-interactive flavor of snapatac2 run:
        `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --tag snapatac2:v2.6.0-recommend-interactive-py3.11 .`

> [!NOTE]
> You can also provide `BASE_PYTHON_IMAGE` and `SNAP_ATAC_VERSION` build args to customize the image that gets built.
> As an example, if you want an image with python 3.1.2 and a different version of SnapATAC2 (e.g. v2.5.1) you could run a build command like:
> `DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg BASE_PYTHON_IMAGE=python:3.12-slim --build-arg SNAP_ATAC_VERSION=v2.5.1 --tag snapatac2:v2.5.1-default-py3.12 .`

> [!WARNING]
> Depending on the version of BASE_PYTHON_IMAGE and SNAP_ATAC_VERSION, the
> resulting images are *NOT* guaranteed to be well-tested or even functional!

## Running SnapATAC2 Docker Images

### Run Instructions for `snapatac2:v2.6.0-recommend-interactive-py3.11` Image on Linux/MacOS (amd64)

> [!WARNING]
> If you want the recommended image to make use of CUDA (GPU) functionality, you will need to separately install the Nvidia container toolkit.
> [Official instructions can be found here (don't accidentally skip the Configuration section either!!)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
> When invoking the `docker run` command you will also need to provide the `--gpus` arg.
> You can read more about how to do this in the [official Docker documentation](https://docs.docker.com/config/containers/resource_constraints/#gpu).

1. Once the image has been built, you can run it with:

```docker run --interactive --tty --rm --env LOCAL_USER_ID=`id -u $USER` --publish 8888:8888 --volume <path_to_local_machine_notebooks>:/notebooks --volume <path_where_you_want_data_saved>:/data snapatac2:v2.6.0-recommend-interactive-py3.11```

2. You can then navigate in your browser to the `http://127.0.0.1:8888/lab?token=<jupyter-lab-token>` link to access Jupyter Lab

> [!NOTE]
> You can learn more about the `docker run` command options from the [official Docker documentation](https://docs.docker.com/engine/reference/commandline/run/#usage)

### Run Instructions for `snapatac2:v2.6.0-recommend-interactive-py3.11` Image on MacOS (arm64) [UNTESTED, EXPERIMENTAL]

> [!WARNING]
> This section is COMPLETELY UNTESTED so no guarantees that it will work at all

1. Similar to the above except you should add `--platform linux/amd64` to the `docker run` command like so:

```docker run --platform linux/amd64 --interactive --tty --rm --env LOCAL_USER_ID=`id -u $USER` --publish 8888:8888 --volume <path_to_local_machine_notebooks>:/notebooks --volume <path_where_you_want_data_saved>:/data snapatac2:v2.6.0-recommend-interactive-py3.11```

### Run Instructions for `snapatac2:v2.6.0-recommend-interactive-py3.11` Image on Windows [EXPERIMENTAL]

1. Install Docker Desktop for Windows: https://docs.docker.com/desktop/install/windows-install/
2. Start up Docker Desktop for Windows
3. Look for the snapatac2 image that you want to run and `pull` it
4. Go back to the main images menu and 'run' the image

<img src="docker-windows-tutorial-0.png" width=25% height=25%>
5. Before clicking `run` open up the `optional settings` and fill in the following:

<img src="docker-windows-tutorial-1.png" width=25% height=25%>
6. A new container should be spun up and you should see in the `logs` section the following:

<img src="docker-windows-tutorial-2.png" width=25% height=25%>
7. Pasting the url in the `logs` section into your browser should let you access Jupyter Lab

