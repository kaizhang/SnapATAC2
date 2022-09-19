v2.2.0 (2022-XX-XX)
===================

v2.1.0 (2022-09-17)
===================

Major changes:

- Various bug fixes and performance improvements.
- Add "tl.diff_test" to identify differentially accessible regions.
- Add "tl.make_fragment_file" to convert BAM files to fragment files.
- Add various functions to perform transcriptional regulatory network analysis.
- Add "read_10x_mtx" for reading files produced by 10X CellRanger pipeline.
- Add "ex.export_bigwig" to generate bigwig files.

Breaking changes:

- "pp.make_tile_matrix" has been renamed to "pp.add_tile_matrix".
- "snapatac2.genome" has been redesigned. A new "Genome" class is added for
  automatic download of gene annotation files. Objects in "snap.genome.*" are
  now instances of the Genome object.
- Rename each occurrence of "group_by" to "groupby" to be consistent with other
  packages.

v2.0.0 (2022-07-07)
===================

Initial release.