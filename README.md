SnapATAC2
=========

![Continuous integration](https://github.com/kaizhang/SnapATAC2/workflows/Continuous%20integration/badge.svg)
[![PyPI version](https://badge.fury.io/py/taiji-utils.svg)](https://badge.fury.io/py/taiji-utils)

SnapATAC2 is the successor of the SnapATAC R package, featuring:

- Faster and less memory usage, scale to >1M cells.
- Improved dimension reduction and sampling algorithm.

## Installation

`pip install snapatac2`

## Build from source

You need to install `rust` first in order to compile the library.

Go to the `snapatac2-python` directory.

`pip install . --use-feature=in-tree-build`