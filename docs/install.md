Installation
============

Stable version
--------------

Stable versions of SnapATAC2 are published on PyPI.
Precompiled binaries are available for x86_64 Linux systems and macOS.
So installing it is as simple as running:

```
pip install snapatac2
```

If there are no precompiled binaries published for your system, you will have to
build the package from source.
Building the SnapATAC2 library requires `cmake >= 3.5.1` and
the [Rust](https://www.rust-lang.org/tools/install) compiler. You need to install
them first if they are not available on your system.
You can find the instructions for installing cmake [here](https://cmake.org/install/).
The Rust compiler can be installed using:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Once you have cmake and the Rust compiler properly installed,
running `pip install snapatac2` will build SnapATAC2 from the source package and
install it just as it would if there was a prebuilt binary available.

Nightly build
-------------

The nightly build is the build from the latest source codes, which includes the
latest features, enhancements, and bug fixes that haven't been released. 
The nightly build can be unstable and include some untested features.

You can download the wheel files of the latest Nightly build from this
[link](https://nightly.link/kaizhang/SnapATAC2/workflows/wheels/main/artifact.zip).

After downloading the file, unzip it and then select the appropriate wheel file
for your platform and use `pip install` to install it.

Build from the latest source code 
---------------------------------

Building the SnapATAC2 library requires `cmake >= 3.5.1` and
the [Rust](https://www.rust-lang.org/tools/install) compiler. You need to install
them first if they are not available on your system.
You can find the instructions for installing cmake [here](https://cmake.org/install/).
The Rust compiler can be installed using:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Once you have cmake and the Rust compiler properly installed,
clone the source code repository and go to the `snapatac2-python` directory,
use `pip install .` to install the package.

```
git clone https://github.com/kaizhang/SnapATAC2.git
cd SnapATAC2/snapatac2-python
pip install .
```

Optional dependencies
---------------------

`pip install snapatac2` installs the essential dependencies needed for SnapATAC2.
For certain features, however, additional optional dependencies are necessary:

- harmonypy: For the `snapatac2.pp.harmony` function.
- scanorama: For the `snapatac2.pp.scanorama_integrate` function.
- xgboot: For network structure inference.

To install these optional dependencies, use `pip install snapatac2[extra]`.

For downstream analysis, some helpful but not mandatory packages, such as scanpy and scvi-tools, are available.
They can be installed with `pip install snapatac2[recommend]`.

To obtain all optional dependencies at once, use `pip install snapatac2[all]`.

Note that the detailed dependencies of SnapATAC2 can be found in the [setup.py](https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/setup.py).