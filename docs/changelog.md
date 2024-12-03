# Release Notes

## Nightly (unreleased)

### Features:

  - Implement `BPM` normalization in `ex.export_coverage`.
  - Add `include_for_norm` and `exclude_for_norm` to `ex.export_coverage`.
  - BedGraph generation in `ex.export_coverage` is 10x faster.
  - Implement broad peak calling in `tl.macs3`.
  - Add `pp.import_values` for importing single base pair values.

## Release 2.7.1 (released October 29, 2024)

### Features:

  - Implement barcode correction algorithms.
  - Add `smooth_length` to `ex.export_coverage`.

### Bugs fixed:

  - Fix #335: GTF parsing error when textual attributes contain semicolons.
  - Fix #347: Add file name sanity check in various places.

## Release 2.7.0 (released August 27, 2024)

### Features:

  - Return more QC metrics from `pp.make_fragment_file`.
  - Also compute library-level TSSe in `metrics.tsse`.
  - `pp.make_fragment_file` can now work with 10X BAM files by specifying `source=10x`.
  - Add `pp.call_cells` to identify nonempty barcodes from raw data.
  - Add `pp.recipe_10x_metrics` to compute 10X metrics.
  - Add `pp.import_contacts` to process scHi-C data.
  - Implement pseudo-bulk peak calling in `tl.macs3`.

### Other Changes:

  - Change the default counting strategy from "insertion" to "paired-insertion" in
    `pp.add_tile_matrix`, `pp.make_peak_matrix`, `pp.make_gene_matrix`.
  - Minor changes in TSSe calculation.

## Release 2.6.4 (released May 28, 2024)

### Features:

  - Implement #304: Add some flexibility to `pp.make_gene_matrix` by allowing the user to change
    upstream and downstream distances for TSS calculation.
  - Other minor improvements.

## Release 2.6.3 (released May 15, 2024)

### Bugs fixed:

  - Fixed external sorter issues in version 2.6.2.

## Release 2.6.2 (released May 9, 2024)

### Features:

  - Add the argument `exclude_chroms` to `metrics.tsse` to exclude certains chromosomes
    from TSSe calculation. The default is "chrM".

## Release 2.6.1 (released May 2, 2024)

### Features:

  - Add the argument `inplace` to AnnData `subset` function. As a result, if you
    want to perform out-of-place subset, you need to set `inplace=False` explicitly.
    Before you just need to set the `out` parameter. The benefit of this change
    is that you can save the subset in memory by setting `out=None, inplace=False`,
    which cannot be achieved before.
  - Use only the unique TSSs instead of all TSSs read from the GTF file in `metrics.tsse` calculation.

### Bugs fixed:

  - Fix #252: `tl.spectral` does not raise an error when the input matrix is in
    compressed column format, whereas it should expect a compressed sparse row format.
  - Fix a bug in `pp.import_data` which produces incorrect duplication rates when
    the input data contains mitochondrial reads.

## Release 2.6.0 (released March 9, 2024)

### Features:

  - Add a argument `counting_strategy` to `pp.add_tile_matrix`, `pp.make_peak_matrix`, and `pp.make_gene_matrix`, which allows one to use different strategies
  (insertion-based, fragment-based, or paired insertion counting) to compute feature counts.
  - Fix #233: Add Apple silicon wheel files.

### Bugs fixed:

- Fix #221: 'pp.knn' with 'method=pynndescent' invalid csr matrix.
- Fix #226: Backed AnnData does not support numpy string array. 
- Fix #232: `tempdir` is not respected in `tl.macs3`.
- Fix #242: Change default value of `min_len` to `extsize` in `tl.macs3`, in order to be
  consistent with the `macs3` command line tool.
- Other minor bug fixes.

## Release 2.5.3 (released Jan 16, 2024)

### Features:

- Add `count_frag_as_reads` argument to `pp.make_tile_matrix`, `pp.make_peak_matrix`,
  and `pp.make_gene_matrix`.

### Bugs fixed:

- Fix: `ex.export_coverage` generates empty files in v2.5.2.

## Release 2.5.2 (released Jan 4, 2024)

### Features:

- Support anndata v0.10.
- Make it easier to build custom genome objects.
- Allow customized "gene_id" and "gene_name" keys in `pp.make_gene_matrix`.
- Add `ex.export_coverage` to export coverage tracks as bedGraph or bigWig files.
  This deprecates `ex.export_bigwig`.
- Add `read_10x_mtx` to read 10X mtx files.
- Implement #192: filtering fragments by size before quantification. This adds
  `min_frag_size` and `max_frag_size` parameters to `pp.add_tile_matrix`,
  `pp.make_peak_matrix`, and `pp.make_gene_matrix`.

### Bugs fixed:

- Fix a bug in `pp.add_tile_matrix` that causes the function to fail when the genome
  contains a large number (>1000) of chromosomes.
- Fix memory leak in `tl.macs3`.
- Fix #177: counter overflow in `pp.make_fragment_file` when duplicate reads are more than 2^16.
- Fix #178: issue a warning instead of an error when no cells pass the filter in `pp.import_data`.
- Fix #179: AnnDataSet to AnnData conversion error when `.X` is empty.
- Fix #182: compression type detection problem.

## Release 2.5.1 (released Oct 23, 2023)

### Function renaming:

- `ex.export_bed` is renamed to `ex.export_fragments`.

### Features:

- Add zstandard file support at various places.
- Change the default output format of `ex.export_bed` to `zst`. To restore the old behavior, set `suffix='.bed.gz'`.

### Bugs fixed:

- Fix a bug in `pp.import_data` that causes the function to occasionally fail on small chunk sizes.

## Release 2.5.0 (released Oct 10, 2023)

### Breaking changes:

- Fragments are now stored in `.obms['fragment_single']` or `.obsm['fragment_paired']`, depending on whether the data is single-end or paired-end. As a result, the h5ad files generated prior to this version are no longer compatible.
- `tl.call_peaks` has been renamed to `tl.macs3`, and the underlying algorithm has been significantly improved, see #142. `tl.macs3` doesn't automatically merge returned peaks anymore. Use `tl.merge_peaks` to merge peaks. Please read the updated tutorial for more details.
- `pp.import_data` now doesn't compute the TSS enrichment scores. Use `metrics.tsse` to compute the TSS enrichment scores. `genome` parameter is renamed to `chrom_sizes`.

### Features:

- Add `tl.merge_peaks` to perform peak merging.
- Allow using custom mitochondrial chromosome names in `pp.import_data`.
- Change the default algorithm in `pp.knn` to kd-tree.

### Bugs fixed:

- #165: Fix reproducibility issues in various functions.

## Release 2.4.0 (released Sep 12, 2023)

### Features:

- Add multiprocessing support to various preprocessing functions like `pp.import_data`.
- Add experimental `pp.import_contacts` to import scHi-C data.
- Add `pp.scanorama_integrate` to perform batch correction using Scanorama.

### Bugs fixed:

- #145, #148, and others.

## Release 2.3.1 (released Jun 21, 2023)

### Features:

- Add `pp.add_frip` to calculate the fraction of reads in peaks or other types of feature sets.
- Make `pp.make_fragment_file`'s behavior more deterministic.

### Bugs fixed:

- #119, #127, #131.

## Release 2.3.0 (released Apr 14, 2023)

### Breaking changes:

- Due to a major upgrade of the `anndata-rs` package. The `AnnData` object generated by older versions is no longer compatible with the new version.
- All dataframes in backed mode are now indexless.
- The `pp.call_doublets` and `pl.scrublet` has been removed. `pp.scrublet` now automatically calls doublets.
- Rename the argument `gff_file` to `gene_anno` in `pp.import_data` and `pp.make_gene_matrix`. Because the gene annotation file is no longer required to be a GFF file.

### Features:

- Updates to `tl.spectral`:
  1. Change the default similarity metric to the cosine similarity.
  2. Significantly improve the scalability of the algorithm when using the cosine similarity. Both the time and space complexity are now linear!
  3. The returned eigenvectors are now weighted by eigenvalues so that using the elbow method to select the number of eigenvectors is no longer necessary. To restore the old behavior, set `weighted_by_sd=False`.
- Implement `tl.multi_spectral` to perform dimension reduction on multiple modalities simultaneously.
- Most preprocessing functions can now take a list of `AnnData` objects as input, and process them in parallel.
- `pp.make_tile_matrix`: allow excluding certain chromosomes by setting `exclude_chroms`.

### Bugs fixed:

- Fix a bug in `pp.call_doublets` that leads to incorrect thresholds for certain datasets.
- Fix #80, #97, #102, #103, #109, #110.

## Release 2.2.0 (released Dec 2, 2022)

### Breaking changes:

- `create_dataset` has been removed. Please use the `AnnDataSet` constructor to create the AnnDataSet object.
- Argument `storage` is renamed to `file` in a variety of AnnData IO functions, including `create_dataset`, `read_mtx`, and `read_csv`.
- Argument `no_check` is removed in `read_dataset`.
- `mode` is renamed to `backed` in `.read()`.

### Features:

- Support in-memory AnnData and flexible conversions between backed and in-memory AnnData.

## Release 2.1.3 (released Nov 2, 2022)

### Features:

- Add a low-memory mode in `pp.import_data` for parsing large unsorted fragment files.

### Bugs fixed:

- Minor bug fixes and performance improvements.
- Fix 10X bam file parsing in `pp.make_fragment_file`.

## Release 2.1.2 (released Oct 1, 2022)

### Bugs fixed:

- Minor bug fixes and performance improvements.

## Release 2.1.1 (released Sep 28, 2022)

### Features:

- Add "tl.motif_enrichment" to perform motif enrichment analysis.

### Bugs fixed:

- Fix bugs in "ex.export_bigwig". NOTE: prior to this version, this function does not work as expected. Please update and rerun if you have used this function in 2.1.0.

## Release 2.1.0 (released Sep 17, 2022)

### Breaking changes:

- "pp.make_tile_matrix" has been renamed to "pp.add_tile_matrix".
- "snapatac2.genome" has been redesigned. A new "Genome" class is added for automatic download of gene annotation files. Objects in "snap.genome.*" are now instances of the Genome object.
- Rename each occurrence of "group_by" to "groupby" to be consistent with other packages.

### Features:

- Add "tl.diff_test" to identify differentially accessible regions.
- Add "tl.make_fragment_file" to convert BAM files to fragment files.
- Add various functions to perform transcriptional regulatory network analysis.
- Add "read_10x_mtx" for reading files produced by 10X CellRanger pipeline.
- Add "ex.export_bigwig" to generate bigwig files.

### Bugs fixed:

- Various bug fixes and performance improvements.

## Release 2.0.0 (released Jul 7, 2022)

This is the first official release of SnapATAC2!
