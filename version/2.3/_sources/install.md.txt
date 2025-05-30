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
rustup default nightly
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
rustup default nightly
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

Here is a list of optional dependencies used by SnapATAC2:

- `macs2`: Peak calling.
- `harmonypy`: Batch correction using the Harmony.
- `xgboost`: Regulatory network analysis.

Optional dependencies are not installed by SnapATAC2. Please install them
manually if necessary.
