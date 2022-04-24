from pathlib import Path
import numpy as np
import math
from typing import Optional, Union, Mapping, Set

from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def import_data(
    fragment_file: Path,
    gff_file: Path,
    chrom_size: Mapping[str, int],
    file: Path,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    sorted_by_barcode: bool = True,
    whitelist: Optional[Union[Path, Set[str]]] = None,
    n_jobs: int = 4,
) -> AnnData:
    """
    Import dataset and compute QC metrics.

    This function will store fragments as base-resolution TN5 insertions in the
    resulting h5ad file (in `.obsm['insertion']`), along with the chromosome
    sizes (in `.uns['reference_sequences']`). Various QC metrics, including TSSe,
    number of unique fragments, duplication rate, fraction of mitochondrial DNA
    reads, will be computed.

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
        Number of unique fragments threshold used to filter cells
    min_tsse
        TSS enrichment threshold used to filter cells
    sorted_by_barcode
        Whether the fragment file has been sorted by cell barcodes. Pre-sort the
        fragment file will speed up the processing and require far less memory.
    whitelist
        File name or a set of strings. If it is a file name, each line
        must contain a valid barcode. 
    n_jobs
        number of CPUs to use

    Returns
    -------
    An annotated data matrix of shape `n_obs` x `n_vars`.
    Rows correspond to cells and columns to regions.
    """
    if isinstance(whitelist, str) or isinstance(whitelist, Path):
        with open(whitelist, "r") as fl:
            whitelist = set([line.strip() for line in fl])
    return internal.import_fragments(
        str(file), str(fragment_file), str(gff_file), chrom_size,
        min_num_fragments, min_tsse, sorted_by_barcode, whitelist, n_jobs
    )

def make_tile_matrix(
    adata: AnnData,
    bin_size: int = 500,
    n_jobs: int = 4
):
    """
    Generate cell by bin count matrix.

    This function will generate and add a cell by bin count matrix to the AnnData
    object in place.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    bin_size
        The size of consecutive genomic regions used to record the counts.
    n_jobs
        number of CPUs to use.
    """
    internal.mk_tile_matrix(adata, bin_size, n_jobs)

def make_peak_matrix(
    adata: AnnData,
    peak_file: Path,
    n_jobs: int = 4
):
    """
    Add a cell by peak count matrix to the AnnData object.

    This function will generate and add a cell by peak count matrix to
    the AnnData object in place. If there is already a matrix in `.X`, it will
    replace it.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    peak_file
        Bed file containing the peaks
    n_jobs
        number of CPUs to use
    """
    internal.mk_peak_matrix(adata, str(peak_file), n_jobs)

def make_gene_matrix(
    adata: Union[AnnData, AnnDataSet],
    gff_file: Path,
    file: Path,
    use_x: bool = False,
) -> AnnData:
    """
    Generate cell by gene activity matrix.

    Generate cell by gene activity matrix by counting the TN5 insertions in gene
    body regions. The result will be stored in a new file and a new AnnData object
    will be created.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    gff_file
        File name of the gene annotation file in GFF format.
    file
        File name of the h5ad file used to store the result.
    use_x
        If True, use the matrix stored in `.X` to compute the gene activity.
        Otherwise the base-resolution TN5 insertions are used.

    Returns
    -------
    A new AnnData object, where rows correspond to cells and columns to genes.
    """
    anndata = internal.mk_gene_matrix(adata, str(gff_file), str(file), use_x)
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
        data.subset(selected_cells)
    else:
        return selected_cells
 
def select_features(
    adata: Union[AnnData, AnnDataSet],
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
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
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
    If `inplace = False`, return a boolean index mask that does filtering,
    where `True` means that the feature is kept, `False` means the feature is removed.
    Otherwise, store this index mask directly to `.var['selected']`.

    Notes
    -----
    This function doesn't perform actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.
    """
    count = np.zeros(adata.shape[1])
    for batch in adata.X.chunked(2000):
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