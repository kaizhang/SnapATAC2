SnapATAC2: A Python/Rust package for single-cell epigenomics analysis
=====================================================================

SnapATAC2 is a flexible, versatile, and scalable single-cell omics analysis framework, featuring:

- Scale to more than 10 million cells.
- Blazingly fast preprocessing tools for BAM to fragment files conversion and count matrix generation.
- Matrix-free spectral embedding algorithm that is applicable to a wide range of single-cell omics data, including single-cell ATAC-seq, single-cell RNA-seq, single-cell Hi-C, and single-cell methylation.
- Efficient and scalable co-embedding algorithm for single-cell multi-omics data integration.
- End-to-end analysis pipeline for single-cell ATAC-seq data, including preprocessing, dimension reduction, clustering, data integration, peak calling, differential analysis, motif analysis, regulatory network analysis.
- Seamless integration with other single-cell analysis packages such as Scanpy.
- Implementation of fully backed AnnData.

How to cite
-----------

The SnapATAC2 manuscript has not been published yet. The key algorithms used in
SnapATAC2 have been described in the following papers:

- Zhang, K. et al. A single-cell atlas of chromatin accessibility in the human genome. Cell 184, 5985-6001.e19 (2021).
- Fang, R. et al. Comprehensive analysis of single cell ATAC-seq data with SnapATAC. Nat Commun 12, 1337 (2021).

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   tutorials/index
   reference/index
   Development<develop>