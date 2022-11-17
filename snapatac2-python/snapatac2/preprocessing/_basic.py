from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np
import math

from snapatac2._snapatac2 import AnnData, AnnDataSet, PyFlagStat
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome
from snapatac2._utils import is_anndata 

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
        Insertion site correction for the left end.
    shift_right
        Insertion site correction for the right end.
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
    return internal.make_fragment_file(
        bam_file, output_file, is_paired, barcode_tag, barcode_regex,
        umi_tag, umi_regex, shift_left, shift_right, min_mapq, chunk_size
    )

def import_data(
    fragment_file: Path,
    *,
    file: Path | None = None,
    genome: Genome | None = None,
    gff_file: Path | None = None,
    chrom_size: dict[str, int] | None = None,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    sorted_by_barcode: bool = True,
    low_memory: bool = True,
    whitelist: Path | list[str] | None = None,
    chunk_size: int = 2000,
    tempdir: Path | None = None,
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
        A Genome object. If not set, `gff_file` and `chrom_size` must be provided.
    gff_file
        File name of the gene annotation file in GFF format.
        This is required if `genome` is not set.
    chrom_size
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`.
        This is required if `genome` is not set.
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
    chunk_size
        Increasing the chunk_size speeds up I/O but uses more memory.
    tempdir
        Location to store temporary files. If `None`, system temporary directory
        will be used.

    Returns
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to regions. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    if genome is not None:
        chrom_size = genome.chrom_sizes
        gff_file = genome.fetch_annotations()

    if whitelist is not None:
        if isinstance(whitelist, str) or isinstance(whitelist, Path):
            with open(whitelist, "r") as fl:
                whitelist = set([line.strip() for line in fl])
        else:
            whitelist = set(whitelist)
    return internal.import_fragments(
        file, fragment_file, gff_file, chrom_size, min_num_fragments,
        min_tsse, sorted_by_barcode, low_memory, whitelist, chunk_size, tempdir
    )

def add_tile_matrix(
    adata: AnnData,
    bin_size: int = 500,
    chunk_size: int = 500,
    n_jobs: int = 4
) -> None:
    """Generate cell by bin count matrix.

    This function is used to generate and add a cell by bin count matrix to the AnnData
    object.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    bin_size
        The size of consecutive genomic regions used to record the counts.
    chunk_size
        Increasing the chunk_size speeds up I/O but uses more memory.
    n_jobs
        Number of CPUs to use.
    """
    internal.mk_tile_matrix(adata, bin_size, chunk_size, n_jobs)

def make_peak_matrix(
    adata: AnnData | AnnDataSet,
    file: Path | None = None,
    use_rep: str | list[str] = "peaks",
    peak_file: Path | None = None,
    chunk_size: int = 500,
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
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used.
    use_rep
        This is used to read peak information from `.uns[use_rep]`.
        The peaks can also be provided by a list of strings:
        ["chr1:1-100", "chr2:2-200"].
    peak_file
        Bed file containing the peaks. If provided, peak information will be read
        from this file.
    chunk_size
        Chunk size

    Returns
    -------
    AnnData | ad.AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to peaks. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.
    """
    import gzip

    if peak_file is not None and use_rep is not None:
        raise RuntimeError("'peak_file' and 'use_rep' cannot be both set") 

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

    anndata = internal.mk_peak_matrix(adata, peaks, file, chunk_size)
    if file is None and adata.isbacked: # anndata accepts only pandas DataFrame
        anndata.obs = adata.obs[:].to_pandas()
    else:
        anndata.obs = adata.obs[:]
    return anndata

def make_gene_matrix(
    adata: AnnData | AnnDataSet,
    gff_file: Genome | Path,
    file: Path | None = None,
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
    gff_file
        Either a Genome object or the path of a gene annotation file in GFF format.
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
        A new AnnData object, where rows correspond to cells and columns to genes.
    """
    if isinstance(gff_file, Genome):
        gff_file = gff_file.fetch_annotations()

    anndata = internal.mk_gene_matrix(adata, gff_file, file, chunk_size, use_x, id_type)
    if file is None and adata.isbacked: # anndata accepts only pandas DataFrame
        anndata.obs = adata.obs[:].to_pandas()
    else:
        anndata.obs = adata.obs[:]
    return anndata

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
 
def select_features(
    adata: AnnData | AnnDataSet,
    min_cells: int = 1,
    most_variable: int | float | None = 1000000,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
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
    min_cells
        Minimum number of cells.
    most_variable
        If None, do not perform feature selection using most variable features
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        Features that are overlapped with these regions will be retained.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """
    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))

    selected_features = count >= min_cells

    if whitelist is not None:
        selected_features &= internal.intersect_bed(adata.var_names, str(whitelist))
    if blacklist is not None:
        selected_features &= np.logical_not(internal.intersect_bed(adata.var_names, str(blacklist)))

    if most_variable is not None and len(count[selected_features]) > most_variable:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        zscores = np.absolute((count - mean) / std)
        cutoff = np.sort(zscores)[most_variable - 1]
        selected_features &= zscores <= cutoff

    if inplace:
        adata.var["selected"] = selected_features
    else:
        return selected_features