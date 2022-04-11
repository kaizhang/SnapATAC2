import numpy as np
import math
from typing import Optional, Union, Literal, Mapping

from snapatac2._snapatac2 import AnnData
import snapatac2._snapatac2 as internal

def import_data(
    fragment_file: str,
    gff_file: str,
    chrom_size: Mapping[str, int],
    file: str,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    sorted_by_barcode: bool = True,
    n_jobs: int = 4,
) -> AnnData:
    """
    Import dataset and compute QC metrics.

    Parameters
    ----------
    fragment_file
        File name of the fragment file
    gff_file
        File name of the gene annotation file in GFF format
    chrom_size
        A dictionary containing chromosome sizes, for example,
        `{"chr1": 2393, "chr2": 2344, ...}`
    file
        File name of the output h5ad file used to store the result
    min_num_fragments
        Threshold used to filter cells
    min_tsse
        Threshold used to filter cells
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes. Pre-sort the
        fragment file will speed up the processing and require far less memory.
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    return internal.import_fragments(
        file, fragment_file, gff_file, chrom_size,
        min_num_fragments, min_tsse, sorted_by_barcode, n_jobs
    )

def make_tile_matrix(
    adata: AnnData,
    bin_size: int = 500,
    n_jobs: int = 4
):
    """
    Generate cell by bin count matrix.

    Parameters
    ----------
    adata
        AnnData
    bin_size
        The size of consecutive genomic regions used to record the counts
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    """
    internal.mk_tile_matrix(adata, bin_size, n_jobs)

def make_gene_matrix(
    adata: AnnData,
    gff_file: str,
    file: str,
    use_x: bool = False,
    n_jobs: int = 4,
) -> AnnData:
    """
    Generate cell by gene activity matrix.

    Parameters
    ----------
    adata
        An anndata instance or the file name of a h5ad file containing the
        cell by bin count matrix
    gff_file
        File name of the gene annotation file in GFF format
    file
        File name of the h5ad file used to store the result
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    anndata = internal.mk_gene_matrix(adata, gff_file, file, use_x, n_jobs)
    anndata.obs = adata.obs[...]
    return anndata

def filter_cells(
    data: AnnData,
    min_counts: Optional[int] = 1000,
    min_tsse: Optional[float] = 5.0,
    max_counts: Optional[int] = None,
    max_tsse: Optional[float] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
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
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:
    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    """
    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    if inplace:
        data.subset(selected_cells)
    else:
        raise NameError("Not implement")
 
def select_features(
    adata: AnnData,
    variable_feature: bool = True,
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Perform feature selection.

    Parameters
    ----------
    adata
        AnnData object
    variable_feature
        Whether to perform feature selection using most variable features
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
    Boolean index mask that does filtering. True means that the cell is kept.
    False means the cell is removed.
    """
    count = np.zeros(adata.shape[1])
    for batch in adata.X.chunked(5000):
        count += np.ravel(batch.sum(axis = 0))

    selected_features = count != 0

    if whitelist is not None:
        selected_features &= internal.intersect_bed(list(adata.var_names), whitelist)
    if blacklist is not None:
        selected_features &= np.logical_not(internal.intersect_bed(list(adata.var_names), blacklist))

    if variable_feature:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        selected_features &= np.absolute((count - mean) / std) < 1.65

    if inplace:
        adata.var["selected"] = selected_features
    else:
        return selected_features