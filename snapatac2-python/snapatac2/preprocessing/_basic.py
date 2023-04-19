from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np
import anndata as ad
import logging

import snapatac2
from snapatac2._snapatac2 import AnnData, AnnDataSet, PyFlagStat
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome

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
) -> PyFlagStat:
    """
    Convert a BAM file to a fragment file.

    Convert a BAM file to a fragment file by performing the following steps:

        1. Filtering: remove reads that are unmapped, not primary alignment, mapq < 30,
           fails platform/vendor quality checks, or optical duplicate.
           For paired-end sequencing, it also removes reads that are not properly aligned.
        2. Deduplicate: Sort the reads by cell barcodes and remove duplicated reads
           for each unique cell barcode.
        3. Output: Convert BAM records to fragments (if paired-end) or single-end reads.

    Note the bam file needn't be sorted or filtered.

    Parameters
    ----------
    bam_file
        File name of the BAM file.
    output_file
        File name of the output fragment file.
    is_paired
        Indicate whether the BAM file contain paired-end reads
    barcode_tag
        Extract barcodes from TAG fields of BAM records, e.g., `barcode_tag = "CB"`.
    barcode_regex
        Extract barcodes from read names of BAM records using regular expressions.
        Reguler expressions should contain exactly one capturing group 
        (Parentheses group the regex between them) that matches
        the barcodes. For example, `barcode_regex = "(..:..:..:..):\w+$"`
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

    Returns
    -------
    PyFlagStat
        Various statistics.
    """
    if barcode_tag is None and barcode_regex is None:
        raise ValueError("Either barcode_tag or barcode_regex must be set.")
    if barcode_tag is not None and barcode_regex is not None:
        raise ValueError("Only one of barcode_tag or barcode_regex can be set.")

    return internal.make_fragment_file(
        bam_file, output_file, is_paired, shift_left, shift_right, chunk_size,
        barcode_tag, barcode_regex, umi_tag, umi_regex, min_mapq,
    )

def import_data(
    fragment_file: Path,
    *,
    file: Path | None = None,
    genome: Genome | None = None,
    gene_anno: Path | None = None,
    chrom_size: dict[str, int] | None = None,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    sorted_by_barcode: bool = True,
    low_memory: bool = True,
    whitelist: Path | list[str] | None = None,
    shift_left: int = 0,
    shift_right: int = 0,
    chunk_size: int = 2000,
    tempdir: Path | None = None,
    backend: str | None = None,
) -> AnnData:
    """Import dataset and compute QC metrics.

    This function will store fragments as base-resolution TN5 insertions in the
    resulting h5ad file (in `.obsm['insertion']`), along with the chromosome
    sizes (in `.uns['reference_sequences']`). Various QC metrics, including TSSe,
    number of unique fragments, duplication rate, fraction of mitochondrial DNA
    reads, will be computed.

    Parameters
    ----------
    fragment_file
        File name of the fragment file.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
    genome
        A Genome object, providing gene annotation and chromosome sizes.
        If not set, `gff_file` and `chrom_size` must be provided.
        `genome` has lower priority than `gff_file` and `chrom_size`.
    gene_anno
        File name of the gene annotation file in GFF or GTF format.
        This is required if `genome` is not set.
        Setting `gene_anno` will override the annotations from the `genome` parameter.
    chrom_size
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
        This is required if `genome` is not set.
        Setting `chrom_size` will override the chrom_size from the `genome` parameter.
    min_num_fragments
        Number of unique fragments threshold used to filter cells
    min_tsse
        TSS enrichment threshold used to filter cells
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes.
        If `sorted_by_barcode == True`, this function makes use of small fixed amout of 
        memory. If `sorted_by_barcode == False` and `low_memory == False`,
        all data will be kept in memory. See `low_memory` for more details.
    low_memory
        Whether to use the low memory mode when `sorted_by_barcode == False`.
        It does this by first sort the records by barcodes and then process them
        in batch. The parameter has no effect when `sorted_by_barcode == True`.
    whitelist
        File name or a list of barcodes. If it is a file name, each line
        must contain a valid barcode. When provided, only barcodes in the whitelist
        will be retained.
    shift_left
        Insertion site correction for the left end.
    shift_right
        Insertion site correction for the right end. Note this has no effect on single-end reads.
        For single-end reads, `shift_right` will be set using the value of `shift_left`.
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
        if gene_anno is None:
            gene_anno = genome.fetch_annotations()

    if whitelist is not None:
        if isinstance(whitelist, str) or isinstance(whitelist, Path):
            with open(whitelist, "r") as fl:
                whitelist = set([line.strip() for line in fl])
        else:
            whitelist = set(whitelist)
        
    adata = ad.AnnData() if file is None else AnnData(filename=file, backend=backend)
    internal.import_fragments(
        adata, fragment_file, gene_anno, chrom_size, min_num_fragments,
        min_tsse, sorted_by_barcode, low_memory, shift_left, shift_right,
        chunk_size, whitelist, tempdir,
    )
    return adata

def add_tile_matrix(
    adata: AnnData | list[AnnData],
    *,
    bin_size: int = 500,
    inplace: bool = True,
    chunk_size: int = 500,
    exclude_chroms: list[str] | str | None = ["chrM", "chrY", "M", "Y"],
    file: Path | None = None,
    backend: str | None = None,
    n_jobs: int = 8,
) -> AnnData | None:
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
    file
        File name of the output file used to store the result. If provided, result will
        be saved to a backed AnnData, otherwise an in-memory AnnData is used.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.
    """
    if isinstance(exclude_chroms, str):
        exclude_chroms = [exclude_chroms]

    if inplace:
        if isinstance(adata, list):
            snapatac2._utils.anndata_par(
                adata,
                lambda x: internal.mk_tile_matrix(x, bin_size, chunk_size, exclude_chroms, None),
                n_jobs=n_jobs,
            )
        else:
            internal.mk_tile_matrix(adata, bin_size, chunk_size, exclude_chroms, None)
    else:
        if file is None:
            if adata.isbacked:
                out = ad.AnnData(obs=adata.obs[:].to_pandas())
            else:
                out = ad.AnnData(obs=adata.obs[:])
        else:
            out = AnnData(filename=file, backend=backend, obs=adata.obs[:])
        internal.mk_tile_matrix(adata, bin_size, chunk_size, exclude_chroms, out)
        return out

def make_peak_matrix(
    adata: AnnData | AnnDataSet,
    *,
    use_rep: str | list[str] | None = None,
    inplace: bool = False,
    file: Path | None = None,
    backend: str | None = None,
    peak_file: Path | None = None,
    chunk_size: int = 500,
    use_x: bool = False,
) -> AnnData:
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
        is used.
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

    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to peaks. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
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
        internal.mk_peak_matrix(adata, peaks, chunk_size, use_x, None)
    else:
        if file is None:
            if adata.isbacked:
                out = ad.AnnData(obs=adata.obs[:].to_pandas())
            else:
                out = ad.AnnData(obs=adata.obs[:])
        else:
            out = AnnData(filename=file, backend=backend, obs=adata.obs[:])
        internal.mk_peak_matrix(adata, peaks, chunk_size, use_x, out)
        return out

def make_gene_matrix(
    adata: AnnData | AnnDataSet,
    gene_anno: Genome | Path,
    *,
    inplace: bool = False,
    file: Path | None = None,
    backend: str | None = None,
    chunk_size: int = 500,
    use_x: bool = False,
    id_type: Literal['gene', 'transcript'] = "gene",
) -> AnnData:
    """Generate cell by gene activity matrix.

    Generate cell by gene activity matrix by counting the TN5 insertions in gene
    body regions. The result will be stored in a new file and a new AnnData object
    will be created.

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
        File name of the h5ad file used to store the result.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` to compute the gene activity.
        Otherwise the `.obsm['insertion']` is used.
    id_type
        "gene" or "transcript".

    Returns
    -------
    AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to genes. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    if isinstance(gene_anno, Genome):
        gene_anno = gene_anno.fetch_annotations()

    if inplace:
        internal.mk_gene_matrix(adata, gene_anno, chunk_size, use_x, id_type, None)
    else:
        if file is None:
            if adata.isbacked:
                out = ad.AnnData(obs=adata.obs[:].to_pandas())
            else:
                out = ad.AnnData(obs=adata.obs[:])
        else:
            out = AnnData(filename=file, backend=backend, obs=adata.obs[:])
        internal.mk_gene_matrix(adata, gene_anno, chunk_size, use_x, id_type, out)
        return out

def filter_cells(
    data: AnnData,
    min_counts: int | None = 1000,
    min_tsse: float | None = 5.0,
    max_counts: int | None = None,
    max_tsse: float | None = None,
    inplace: bool = True,
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

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return 
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.
    """
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
    adata: AnnData | AnnDataSet | list[AnnData],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    max_iter: int = 1,
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.

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
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    
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
            lambda x: select_features(x, n_features, filter_lower_quantile, filter_upper_quantile, whitelist, blacklist, max_iter, inplace),
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
        blacklist = np.logical_not(internal.intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = snapatac2.tl.spectral(adata, features=selected_features, inplace=False)[1]
        clusters = snapatac2.tl.leiden(snapatac2.pp.knn(embedding, inplace=False))
        rpm = snapatac2.tl.aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    if whitelist is not None:
        whitelist = np.array(internal.intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist
    
    logging.info(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result
