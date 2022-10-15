Installation
============

Stable version
--------------

Stable versions of SnapATAC2 are published on pypi.
Precompiled binaries are available for x86_64 Linux systems and macOS.
So installing it is as simple as running:

::

    pip install snapatac2

If there are no precompiled binaries published for your system you'll have to
build the package from source.
However, to be able able to build the package from the published source package,
You need to install `rust <https://www.rust-lang.org/tools/install>`_ first:

::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Once you have rust properly installed, running:

::

    pip install snapatac2

will build SnapATAC2 for your local system from the source package and
install it just as it would if there was a prebuilt binary available.


Nightly build
-------------

The nightly build is the build from the latest source codes, which includes the
the latest features, enhancements, and bug fixes that haven't been released. 
The nightly build can be unstable and include some untested features.

You can download the wheel files of the latest Nightly build from this
`link <https://nightly.link/kaizhang/SnapATAC2/workflows/wheels/main/artifact.zip>`_.

After downloading the file, unzip it and then select the appropriate wheel file
for your platform and use `pip install` to install it.

Build from the source code 
--------------------------

You need to install `rust <https://www.rust-lang.org/tools/install>`_ first
in order to compile the library, which can be accomplished by running:

::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Clone the source code repository and go to the `snapatac2-python` directory,
use `pip install .` or `python setup.py install` to install the package.

::

    git clone https://github.com/kaizhang/SnapATAC2.git
    cd snapatac2-python
    pip install .