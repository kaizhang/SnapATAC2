SnapATAC2: A Python/Rust package for single-cell epigenomics analysis
=====================================================================

![PyPI](https://img.shields.io/pypi/v/snapatac2)
![PyPI - Downloads](https://img.shields.io/pypi/dm/snapatac2)
![Continuous integration](https://github.com/kaizhang/SnapATAC2/workflows/Continuous%20integration/badge.svg)
[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://snapatac2.zulipchat.com/join/rs5zviisizhtx7abznm77xmq/)
![GitHub Repo stars](https://img.shields.io/github/stars/kaizhang/SnapATAC2?style=social)

SnapATAC2 is a flexible, versatile, and scalable single-cell omics analysis framework, featuring:

- Scale to more than 10 million cells.
- Blazingly fast preprocessing tools for BAM to fragment files conversion and count matrix generation.
- Matrix-free spectral embedding algorithm that is applicable to a wide range of single-cell omics data, including single-cell ATAC-seq, single-cell RNA-seq, single-cell Hi-C, and single-cell methylation.
- Efficient and scalable co-embedding algorithm for single-cell multi-omics data integration.
- End-to-end analysis pipeline for single-cell ATAC-seq data, including preprocessing, dimension reduction, clustering, data integration, peak calling, differential analysis, motif analysis, regulatory network analysis.
- Seamless integration with other single-cell analysis packages such as Scanpy.
- Implementation of fully backed AnnData.

Resources
---------

- **Full Documentation**: https://kzhang.org/SnapATAC2/
- **Installation instructions**: https://kzhang.org/SnapATAC2/install.html
- **Tutorial/Demo**: https://kzhang.org/SnapATAC2/tutorial.html
- **Benchmarks**: https://github.com/kaizhang/single-cell-benchmark

How to cite
-----------

We are working on the SnapATAC2 manuscript... 