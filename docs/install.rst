Installation
============

PyPI
----

::

    pip install snapatac2

Development Version
-------------------

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