from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np
from anndata import AnnData
import logging

import snapatac2
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome

__all__ = ['make_fragment_file', 'import_data', 'import_contacts', 'add_tile_matrix',
           'make_peak_matrix', 'filter_cells', 'select_features', 'make_gene_matrix'
]

def make_fragment_file(
    bam_file: Path,
    output_file: Path,
    is_paired: bool = True,
    barcode_tag: str | None = None,
    barcode_regex: str | None = None,
    umi_tag: str | None = None,
    umi_regex: str | None = None,
    shift_left: int = 4,
    shift_right: int = -5,
    min_mapq: int | None = 30,
    chunk_size: int = 50000000,
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
    tempdir: Path | None = None,
) -> internal.PyFlagStat:
    """
    Convert a BAM file to a fragment file.

    Convert a BAM file to a fragment file by performing the following steps:

        1. Filtering: remove reads that are unmapped, not primary alignment, mapq < 30,
           fails platform/vendor quality checks, or optical duplicate.
           For paired-end sequencing, it also removes reads that are not properly aligned.
        2. Deduplicate: Sort the reads by cell barcodes and remove duplicated reads
           for each unique cell barcode.
        3. Output: Convert BAM records to fragments (if paired-end) or single-end reads.

    The bam file needn't be sorted or filtered.

    Note
    ----
    - When using `barcode_regex` or `umi_regex`, the regex must contain exactly one capturing group
    (Parentheses group the regex between them) that matches the barcodes or UMIs.
    Writting the correct regex is tricky. You can test your regex online at https://regex101.com/.
    BAM files produced by the 10X Genomics Cell Ranger pipeline are not supported,
    as they contain invalid BAM headers. Specifically, Cell Ranger ATAC <= 2.0 produces BAM
    files with no @VN tag in the header, and Cell Ranger ATAC >= 2.1 produces BAM files
    with invalid @VN tag in the header.
    It is recommended to use the fragment files produced by Cell Ranger ATAC instead.
    - This function generates large temporary files in `tempdir` during sorting.
    For large files, it is recommended to set `tempdir` to a location with
    sufficient space in order to avoid running out of disk space.

    Parameters
    ----------
    bam_file
        File name of the BAM file.
    output_file
        File name of the output fragment file.
    is_paired
        Indicate whether the BAM file contain paired-end reads
    barcode_tag
        Extract barcodes from TAG fields of BAM records, e.g., `barcode_tag="CB"`.
    barcode_regex
        Extract barcodes from read names of BAM records using regular expressions.
        Reguler expressions should contain exactly one capturing group 
        (Parentheses group the regex between them) that matches
        the barcodes. For example, `barcode_regex="(..:..:..:..):\\w+$"`
        extracts `bd:69:Y6:10` from
        `A01535:24:HW2MMDSX2:2:1359:8513:3458:bd:69:Y6:10:TGATAGGTTG`.
    umi_tag
        Extract UMI from TAG fields of BAM records.
    umi_regex
        Extract UMI from read names of BAM records using regular expressions.
        See `barcode_regex` for more details.
    shift_left
        Insertion site correction for the left end. Note this has no effect on single-end reads.
    shift_right
        Insertion site correction for the right end. Note this has no effect on single-end reads.
    min_mapq
        Filter the reads based on MAPQ.
    chunk_size
        The size of data retained in memory when performing sorting. Larger chunk sizes
        result in faster sorting and greater memory usage.
    compression
        Compression type. If `None`, it is inferred from the suffix.
    compression_level
        Compression level. 1-9 for gzip, 1-22 for zstandard.
        If `None`, it is set to 6 for gzip and 3 for zstandard.
    tempdir
        Location to store temporary files. If `None`, system temporary directory
        will be used.

    Returns
    -------
    PyFlagStat
        Various statistics.

    See Also
    --------
    import_data
    """
    if barcode_tag is None and barcode_regex is None:
        raise ValueError("Either barcode_tag or barcode_regex must be set.")
    if barcode_tag is not None and barcode_regex is not None:
        raise ValueError("Only one of barcode_tag or barcode_regex can be set.")

    if compression is None:
        _, compression = snapatac2._utils.get_file_format(output_file)

    return internal.make_fragment_file(
        bam_file, output_file, is_paired, shift_left, shift_right, chunk_size,
        barcode_tag, barcode_regex, umi_tag, umi_regex, min_mapq,
        compression, compression_level, tempdir,
    )

def import_data(
    fragment_file: Path | list[Path],
    chrom_sizes: Genome | dict[str, int],
    *,
    file: Path | list[Path] | None = None,
    min_num_fragments: int = 200,
    sorted_by_barcode: bool = True,
    whitelist: Path | list[str] | None = None,
    chrM: list[str] = ["chrM", "M"],
    shift_left: int = 0,
    shift_right: int = 0,
    chunk_size: int = 2000,
    tempdir: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    n_jobs: int = 8,
) -> internal.AnnData:
    """Import data fragment files and compute basic QC metrics.

    A fragment refers to the sequence data originating from a distinct location
    in the genome. In single-ended sequencing, one read equates to a fragment.
    However, in paired-ended sequencing, a fragment is defined by a pair of reads.
    This function is designed to handle, store, and process input files with
    fragment data, further yielding a range of basic Quality Control (QC) metrics.
    These metrics include the total number of unique fragments, duplication rates,
    and the percentage of mitochondrial DNA detected.

    How fragments are stored is dependent on the sequencing approach utilized.
    For single-ended sequencing, fragments are found in `.obsm['fragment_single']`.
    In contrast, for paired-ended sequencing, they are located in
    `.obsm['fragment_paired']`.

    Diving deeper technically, the fragments are internally structured within a
    Compressed Sparse Row (CSR) matrix. In this configuration, each row signifies
    a specific cell, while each column represents a unique genomic position.
    Fragment starting positions dictate the column indices. Matrix values
    capture the lengths of the fragments for paired-end reads or the lengths of
    the reads for single-ended scenarios. It's important to note that for
    single-ended reads, the values are signed, with the sign providing information
    on the fragment's strand orientation. Additionally, it is worth noting that
    cells may harbor duplicate fragments, leading to the presence of duplicate
    column indices within the matrix. As a result, the matrix deviates from
    the standard CSR format, and it is not advisable to use the matrix for linear
    algebra operations.
    
    .. image:: /_static/images/func+import_data.svg
        :align: center

    Note
    ----
    - This function accepts both single-end and paired-end reads. 
      If the records in the fragment file contain 6 columns with the last column
      representing the strand of the fragment, the fragments are considered single-ended.
      Otherwise, the fragments are considered paired-ended.
    - When `file` is not `None`, this function uses constant memory regardless of
      the size of the input file.
    - When `sorted_by_barcode` is `False`, this function will sort the fragment file
      first, during which temporary files will be created in `tempdir`. The size of
      temporary files is proportional to the number of records in the fragment file.
      For large fragment files, it is recommended to set `tempdir` to a location with
      sufficient space in order to avoid running out of disk space.

    Parameters
    ----------
    fragment_file
        File name of the fragment file, optionally compressed with gzip or zstd.
        This can be a single file or a list of files.
        If it is a list of files, a separate AnnData object will be created for each file.
        A fragment file must contain at least 5 columns:
        chromosome, start, end, barcode, count.
        Optionally it can contain one more column indicating the strand of the fragment.
        When strand is provided, the fragments are considered single-ended.
    chrom_sizes
        A Genome object or a dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
        If `fragment_file` is a list of files, `file` must also be a list of files if provided.
    min_num_fragments
        Number of unique fragments threshold used to filter cells
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes.
        This function will be faster if `sorted_by_barcode==True`.
        Note the :func:`~snapatac2.pp.make_fragment_file` will always sort the
        fragment file by barcode.
    whitelist
        File name or a list of barcodes. If it is a file name, each line
        must contain a valid barcode. When provided, only barcodes in the whitelist
        will be retained.
    shift_left
        Insertion site correction for the left end. This is set to 0 by default,
        as shift correction is usually done in the fragment file generation step.
    chrM
        A list of chromosome names that are considered mitochondrial DNA. This is
        used to compute the fraction of mitochondrial DNA.
    shift_right
        Insertion site correction for the right end. Note this has no effect on single-end reads.
        For single-end reads, `shift_right` will be set using the value of `shift_left`.
        This is set to 0 by default, as shift correction is usually done in the fragment
        file generation step.
    chunk_size
        Increasing the chunk_size may speed up I/O but will use more memory.
        The speed gain is usually not significant.
    tempdir
        Location to store temporary files. If `None`, system temporary directory
        will be used.
    backend
        The backend.
    n_jobs
        Number of jobs to run in parallel when `fragment_file` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to regions. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    make_fragment_file
    :func:`~snapatac2.ex.export_fragments` 

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> print(data)
    AnnData object with n_obs × n_vars = 585 × 0
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
        uns: 'reference_sequences'
        obsm: 'fragment_paired'
    """
    chrom_sizes = chrom_sizes.chrom_sizes if isinstance(chrom_sizes, Genome) else chrom_sizes
    if len(chrom_sizes) == 0:
        raise ValueError("chrom_size cannot be empty")

    if whitelist is not None:
        if isinstance(whitelist, str) or isinstance(whitelist, Path):
            with open(whitelist, "r") as fl:
                whitelist = set([line.strip() for line in fl])
        else:
            whitelist = set(whitelist)

    if isinstance(fragment_file, list):
        n = len(fragment_file)
        if file is None:
            adatas = [AnnData() for _ in range(n)]
        else:
            if len(file) != n:
                raise ValueError("The length of 'file' must be the same as the length of 'fragment_file'")
            adatas = [internal.AnnData(filename=f, backend=backend) for f in file]

        snapatac2._utils.anndata_ipar(
            list(enumerate(adatas)),
            lambda x: internal.import_fragments(
                x[1], fragment_file[x[0]], chrom_sizes, chrM, min_num_fragments,
                sorted_by_barcode, shift_left, shift_right, chunk_size, whitelist, tempdir,
            ),
            n_jobs=n_jobs,
        )
        return adatas
    else:
        adata = AnnData() if file is None else internal.AnnData(filename=file, backend=backend)
        internal.import_fragments(
            adata, fragment_file, chrom_sizes, chrM, min_num_fragments,
            sorted_by_barcode, shift_left, shift_right, chunk_size, whitelist, tempdir,
        )
        return adata

def import_contacts(
    contact_file: Path,
    *,
    file: Path | None = None,
    genome: Genome | None = None,
    chrom_size: dict[str, int] | None = None,
    sorted_by_barcode: bool = True,
    chunk_size: int = 2000,
    tempdir: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
) -> internal.AnnData:
    """Import chromatin contacts.

    Parameters
    ----------
    contact_file
        File name of the fragment file.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
    genome
        A Genome object, providing gene annotation and chromosome sizes.
        If not set, `gff_file` and `chrom_size` must be provided.
        `genome` has lower priority than `gff_file` and `chrom_size`.
    chrom_size
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
        This is required if `genome` is not set.
        Setting `chrom_size` will override the chrom_size from the `genome` parameter.
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes.
        If `sorted_by_barcode == True`, this function makes use of small fixed amout of 
        memory. If `sorted_by_barcode == False` and `low_memory == False`,
        all data will be kept in memory. See `low_memory` for more details.
    chunk_size
        Increasing the chunk_size speeds up I/O but uses more memory.
    tempdir
        Location to store temporary files. If `None`, system temporary directory
        will be used.
    backend
        The backend.

    Returns
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to regions. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    if genome is not None:
        if chrom_size is None:
            chrom_size = genome.chrom_sizes

    adata = AnnData() if file is None else internal.AnnData(filename=file, backend=backend)
    internal.import_contacts(
        adata, contact_file, chrom_size, sorted_by_barcode, chunk_size, tempdir
    )
    return adata

def add_tile_matrix(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    bin_size: int = 500,
    inplace: bool = True,
    chunk_size: int = 500,
    exclude_chroms: list[str] | str | None = ["chrM", "chrY", "M", "Y"],
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'insertion',
    file: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    n_jobs: int = 8,
) -> internal.AnnData | None:
    """Generate cell by bin count matrix.

    This function is used to generate and add a cell by bin count matrix to the AnnData
    object.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects when `inplace=True`.
        In this case, the function will be applied to each AnnData object in parallel.
    bin_size
        The size of consecutive genomic regions used to record the counts.
    inplace
        Whether to add the tile matrix to the AnnData object or return a new AnnData object.
    chunk_size
        Increasing the chunk_size speeds up I/O but uses more memory.
    exclude_chroms
        A list of chromosomes to exclude.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.
    file
        File name of the output file used to store the result. If provided, result will
        be saved to a backed AnnData, otherwise an in-memory AnnData is used.
        This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.
    
    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to bins. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    make_peak_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.pp.add_tile_matrix(data, bin_size=500)
    >>> print(data)
    AnnData object with n_obs × n_vars = 585 × 6062095
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
        uns: 'reference_sequences'
        obsm: 'fragment_paired'
    """
    if isinstance(exclude_chroms, str):
        exclude_chroms = [exclude_chroms]

    if inplace:
        if isinstance(adata, list):
            snapatac2._utils.anndata_par(
                adata,
                lambda x: internal.mk_tile_matrix(x, bin_size, chunk_size, counting_strategy, exclude_chroms, None),
                n_jobs=n_jobs,
            )
        else:
            internal.mk_tile_matrix(adata, bin_size, chunk_size, counting_strategy, exclude_chroms, min_frag_size, max_frag_size, None)
    else:
        if file is None:
            if adata.isbacked:
                out = AnnData(obs=adata.obs[:].to_pandas())
            else:
                out = AnnData(obs=adata.obs[:])
        else:
            out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
        internal.mk_tile_matrix(adata, bin_size, chunk_size, counting_strategy, exclude_chroms, min_frag_size, max_frag_size, out)
        return out

def make_peak_matrix(
    adata: internal.AnnData | internal.AnnDataSet,
    *,
    use_rep: str | list[str] | None = None,
    inplace: bool = False,
    file: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    peak_file: Path | None = None,
    chunk_size: int = 500,
    use_x: bool = False,
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'insertion',
) -> internal.AnnData:
    """Generate cell by peak count matrix.

    This function will generate a cell by peak count matrix and store it in a 
    new .h5ad file.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    use_rep
        This is used to read peak information from `.uns[use_rep]`.
        The peaks can also be provided by a list of strings:
        ["chr1:1-100", "chr2:2-200"].
    inplace
        Whether to add the tile matrix to the AnnData object or return a new AnnData object.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used. This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    peak_file
        Bed file containing the peaks. If provided, peak information will be read
        from this file.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` as raw counts.
        Otherwise the `.obsm['insertion']` is used.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.

    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to peaks. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    add_tile_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> peak_mat = snap.pp.make_peak_matrix(data, peak_file=snap.datasets.cre_HEA())
    >>> print(peak_mat)
    AnnData object with n_obs × n_vars = 585 × 1154611
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
    """
    import gzip

    if peak_file is not None and use_rep is not None:
        raise RuntimeError("'peak_file' and 'use_rep' cannot be both set") 

    if use_rep is None and peak_file is None:
        use_rep = "peaks"

    if isinstance(use_rep, str):
        df = adata.uns[use_rep]
        peaks = df[df.columns[0]]
    else:
        peaks = use_rep

    if peak_file is not None:
        if Path(peak_file).suffix == ".gz":
            with gzip.open(peak_file, 'rt') as f:
                peaks = [line.strip() for line in f]
        else:
            with open(peak_file, 'r') as f:
                peaks = [line.strip() for line in f]

    if inplace:
        out = None
    elif file is None:
        if adata.isbacked:
            out = AnnData(obs=adata.obs[:].to_pandas())
        else:
            out = AnnData(obs=adata.obs[:])
    else:
        out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
    internal.mk_peak_matrix(adata, peaks, chunk_size, use_x, counting_strategy, min_frag_size, max_frag_size, out)
    return out

def make_gene_matrix(
    adata: internal.AnnData | internal.AnnDataSet,
    gene_anno: Genome | Path,
    *,
    inplace: bool = False,
    file: Path | None = None,
    backend: Literal['hdf5'] | None = 'hdf5',
    chunk_size: int = 500,
    use_x: bool = False,
    id_type: Literal['gene', 'transcript'] = "gene",
    transcript_name_key: str = "transcript_name",
    transcript_id_key: str = "transcript_id",
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'insertion',
) -> internal.AnnData:
    """Generate cell by gene activity matrix.

    Generate cell by gene activity matrix by counting the TN5 insertions in gene
    body regions. The result will be stored in a new file and a new AnnData object
    will be created.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    gene_anno
        Either a Genome object or the path of a gene annotation file in GFF or GTF format.
    inplace
        Whether to add the gene matrix to the AnnData object or return a new AnnData object.
    file
        File name of the h5ad file used to store the result. This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` to compute the gene activity.
        Otherwise the `.obsm['insertion']` is used.
    id_type
        "gene" or "transcript".
    transcript_name_key
        The key of the transcript name in the gene annotation file.
    transcript_id_key
        The key of the transcript id in the gene annotation file.
    gene_name_key
        The key of the gene name in the gene annotation file.
    gene_id_key
        The key of the gene id in the gene annotation file.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.

    Returns
    -------
    AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to genes. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    add_tile_matrix
    make_peak_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> gene_mat = snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38)
    >>> print(gene_mat)
    AnnData object with n_obs × n_vars = 585 × 60606
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
    """
    if isinstance(gene_anno, Genome):
        gene_anno = gene_anno.annotation

    if inplace:
        out = None
    elif file is None:
        if adata.isbacked:
            out = AnnData(obs=adata.obs[:].to_pandas())
        else:
            out = AnnData(obs=adata.obs[:])
    else:
        out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
    internal.mk_gene_matrix(adata, gene_anno, chunk_size, use_x, id_type,
        transcript_name_key, transcript_id_key, gene_name_key, gene_id_key,
        counting_strategy, min_frag_size, max_frag_size, out)
    return out

def filter_cells(
    data: internal.AnnData | list[internal.AnnData],
    min_counts: int | None = 1000,
    min_tsse: float | None = 5.0,
    max_counts: int | None = None,
    max_tsse: float | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | None:
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `data` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_tsse
        Minimum TSS enrichemnt score required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_tsse
        Maximum TSS enrichment score expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `data` is a list.

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return 
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.
    """
    if isinstance(data, list):
        result = snapatac2._utils.anndata_par(
            data,
            lambda x: filter_cells(x, min_counts, min_tsse, max_counts, max_tsse, inplace=inplace),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    if inplace:
        if data.isbacked:
            data.subset(selected_cells)
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells

def _find_most_accessible_features(
    feature_count,
    filter_lower_quantile,
    filter_upper_quantile,
    total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]
 
def select_features(
    adata: internal.AnnData | internal.AnnDataSet | list[internal.AnnData],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    max_iter: int = 1,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection by selecting the most accessibile features across
    all cells unless `max_iter` > 1.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.
    Features that are zero in all cells will be always removed regardless of the
    filtering criteria.
    For more discussion about feature selection, see: https://github.com/kaizhang/SnapATAC2/discussions/116.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    n_features
        Number of features to keep. Note that the final number of features
        may be smaller than this number if there is not enough features that pass
        the filtering criteria.
    filter_lower_quantile
        Lower quantile of the feature count distribution to filter out.
        For example, 0.005 means the bottom 0.5% features with the lowest counts will be removed.
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
        For example, 0.005 means the top 0.5% features with the highest counts will be removed.
        Be aware that when the number of feature is very large, the default value of 0.005 may
        risk removing too many features.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
        If a feature is present in both whitelist and blacklist, it will be kept.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    max_iter
        If greater than 1, this function will perform iterative clustering and feature selection
        based on variable features found using previous clustering results.
        This is similar to the procedure implemented in ArchR, but we do not recommend it,
        see https://github.com/kaizhang/SnapATAC2/issues/111.
        Default value is 1, which means no iterative clustering is performed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: select_features(x, n_features, filter_lower_quantile,
                                      filter_upper_quantile, whitelist,
                                      blacklist, max_iter, inplace, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        count += np.ravel(batch.sum(axis = 0))
    adata.var['count'] = count

    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features)

    if blacklist is not None:
        blacklist = np.array(internal.intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = snapatac2.tl.spectral(adata, features=selected_features, inplace=False)[1]
        clusters = snapatac2.tl.leiden(snapatac2.pp.knn(embedding, inplace=False))
        rpm = snapatac2.tl.aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        # Apply blacklist to the result
        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    # Finally, apply whitelist to the result
    if whitelist is not None:
        whitelist = np.array(internal.intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist
    
    if verbose:
        logging.info(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result
