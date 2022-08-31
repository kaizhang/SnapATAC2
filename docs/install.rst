Installation
============

Option 1: PyPI
---------------

Precompiled version of SnapATAC2 is published on pypi for x86_64 Linux systems
and macOS. So installing it is as simple as running:

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


Option 2: Development Version
------------------------------

You need to install `rust <https://www.rust-lang.org/tools/install>`_ first
in order to compile the library, which can be accomplished by running:

::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Clone the source code repository and go to the `snapatac2-python` directory,
use `pip install . --use-feature=in-tree-build` or
`python setup.py install` to install the package.

::

    git clone https://github.com/kaizhang/SnapATAC2.git
    cd snapatac2-python
    pip install . --use-feature=in-tree-build