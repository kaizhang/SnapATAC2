SnapATAC2
=========

![PyPI](https://img.shields.io/pypi/v/snapatac2)
![PyPI - Downloads](https://img.shields.io/pypi/dm/snapatac2)
![Continuous integration](https://github.com/kaizhang/SnapATAC2/workflows/Continuous%20integration/badge.svg)
[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://snapatac2.zulipchat.com/join/rs5zviisizhtx7abznm77xmq/)
![GitHub Repo stars](https://img.shields.io/github/stars/kaizhang/SnapATAC2?style=social)

SnapATAC2 is the successor of the SnapATAC R package, featuring:

- Faster and less memory usage than the R version and other alternatives, scale to >1M cells.
- Blazingly fast preprocessing tools for BAM to fragment files conversion and count matrix generation.
- One of the best dimension reduction algorithms for single-cell ATAC data.
- End-to-end analysis pipeline, including preprocessing, dimension reduction, clustering, data integration, peak calling, differential analysis, motif analysis, regulatory network analysis.
- Seamless integration with other single-cell analysis packages such scanpy.

**To find tutorials and the full APIs, please go to this [website](https://kzhang.org/SnapATAC2/).**

How to cite
-----------

The SnapATAC2 manuscript has not been published yet. The key algorithms used in
SnapATAC2 have been described in the following papers:

- Zhang, K. et al. A single-cell atlas of chromatin accessibility in the human genome. Cell 184, 5985-6001.e19 (2021).
- Fang, R. et al. Comprehensive analysis of single cell ATAC-seq data with SnapATAC. Nat Commun 12, 1337 (2021).