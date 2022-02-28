SnapATAC2
=========

![PyPI](https://img.shields.io/pypi/v/snapatac2)
[![Downloads](https://pepy.tech/badge/snapatac2)](https://pepy.tech/project/snapatac2)
![Continuous integration](https://github.com/kaizhang/SnapATAC2/workflows/Continuous%20integration/badge.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/kaizhang/SnapATAC2?style=social)

SnapATAC2 is the successor of the SnapATAC R package, featuring:

- Faster and less memory usage, scale to >1M cells.
- Improved dimension reduction and sampling algorithm.

Read the [Documentation](https://kzhang.org/SnapATAC2/). 

# Benchmark

## Impact of dimension reduction algorithms on clustering accuracy

Same matrices were given to each algorithm to perform dimension reduction, the
result was then input to the same clustering procedure (KNN + leiden) to get clusters and compare
with ground truth.
During the clustering step, we tested a wide range of parameters for each method and the
best outcome was recorded.

![benchmark](docs/_static/benchmark.svg)

## Subsampling test

![subsample](docs/_static/benchmark_subsample.svg)

## Running time and space complexity

![performance](docs/_static/benchmark_performance.svg)

## Example visualization

![Zhang_Cell_2021](docs/_static/Zhang_Cell_2021_GI.png)