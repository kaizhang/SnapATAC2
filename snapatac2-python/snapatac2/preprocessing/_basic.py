import numpy as np
import anndata as ad
import math
from typing import Optional, Union, Literal
from anndata.experimental import AnnCollection

import snapatac2._snapatac2 as internal

def make_tile_matrix(
    output: str,
    fragment_file: str,
    gff_file: str,
    chrom_size,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    bin_size: int = 500,
    sorted_by_barcode: bool = True,
    backed: Optional[Literal["r", "r+"]] = None,
    n_jobs: int = 4,
) -> ad.AnnData:
    """
    Generate cell by bin count matrix.

    Parameters
    ----------
    output
        file name for saving the result
    fragment_file
        fragment file
    gff_file
        annotation file in GFF format
    chrom_size
        chromosome sizes
    min_num_fragments
        threshold used to filter cells
    min_tsse
        threshold used to filter cells
    bin_size
        the size of consecutive genomic regions used to record the counts
    sorted_by_barcode
        whether the fragment file has been sorted by cell barcodes. Pre-sort the
        fragment file will speed up the processing and require far less memory.
    backed
        Whether to return the resulting anndata in backed mode.
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    internal.mk_tile_matrix(
        output, fragment_file, gff_file, chrom_size,
        bin_size, min_num_fragments, min_tsse, sorted_by_barcode, n_jobs
    )
    return ad.read(output) if backed is None else ad.read(output, backed=backed)

def make_gene_matrix(
    adata: Union[ad.AnnData, str],
    gff_file: str,
    copy_obs: bool = True,
    copy_obsm: bool = True,
    output: Optional[str] = "gene_matrix.h5ad",
    backed: Optional[Literal["r", "r+"]] = None,
) -> ad.AnnData:
    """
    Generate cell by gene activity matrix.

    Parameters
    ----------
    adata
        input anndata containing the cell by bin count matrix
    gff_file
        annotation file in GFF format
    backed
        Whether to return the resulting anndata in backed mode.
    
    Returns
    -------
    AnnData
    """
   
    if isinstance(adata, ad.AnnData):
        if adata.filename is None:
            raise ValueError("anndata need to be in backed mode!")
        else:
            input_file = str(adata.filename)
    else:
        input_file = adata
    internal.mk_gene_matrix(output, str(input_file), gff_file)
    gene_mat = ad.read(output) if backed is None else ad.read(output, backed=backed)
    if copy_obs: gene_mat.obs = adata.obs
    if copy_obsm: gene_mat.obsm = adata.obsm
    return gene_mat

def filter_cells(
    data: ad.AnnData,
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
        data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells
 
def select_features(
    adata: Union[ad.AnnData, AnnCollection],
    variable_feature: bool = True,
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
) -> np.ndarray:
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
    
    Returns
    -------
    Boolean index mask that does filtering. True means that the cell is kept.
    False means the cell is removed.
    """
    if isinstance(adata, ad.AnnData):
        count = np.ravel(adata.X[...].sum(axis = 0))
    elif isinstance(adata, AnnCollection):
        count = np.zeros(adata.shape[1])
        for batch, _ in adata.iterate_axis(5000):
            count += np.ravel(batch.X[...].sum(axis = 0))
    else:
        raise ValueError(
            '`pp.highly_variable_genes` expects an `AnnData` argument, '
            'pass `inplace=False` if you want to return a `pd.DataFrame`.'
        )

    selected_features = count != 0

    if whitelist is not None:
        selected_features &= internal.intersect_bed(list(adata.var_names), whitelist)
    if blacklist is not None:
        selected_features &= not internal.intersect_bed(list(adata.var_names), blacklist)

    if variable_feature:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        selected_features &= np.absolute((count - mean) / std) < 1.65

    return selected_features