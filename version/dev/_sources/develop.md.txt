Contributing to SnapATAC2
=========================

## A brief guide to adding new features to SnapATAC2

The SnapATAC2 Github repository contains three libraries:

1. `snapatac2-core`: Core functions written in Rust.
2. `snapatac2-python`: High-level user-facing functions written in mostly Python.
3. `snapatac2-contrib`: Python package containing additional features from external contributors.

Unless the new feature is essential and highly relevant to functions in the core library, it is recommended to add it to the `snapatac2-contrib` library.
Otherwise, add the feature to the `snapatac2-python` library under the `snapatac2.experimental` module.
We will move function from `snapatac2.experimental` to `snapatac2` once they are sufficiently tested and stable.