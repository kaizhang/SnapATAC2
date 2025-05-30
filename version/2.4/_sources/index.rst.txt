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

Kai Zhang, Nathan Zemke, Ethan Armand, Bing Ren.
SnapATAC2: a fast, scalable and versatile tool for single-cell omics analysis.
bioRxiv 2023.09.11.557221; doi: https://doi.org/10.1101/2023.09.11.557221

.. toctree::
   :maxdepth: 3
   :hidden:

   install
   tutorials/index
   reference/index
   Development<develop>