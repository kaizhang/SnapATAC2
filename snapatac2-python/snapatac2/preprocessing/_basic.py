import numpy as np
import anndata as ad
import math
from typing import Optional, Union
from anndata.experimental import AnnCollection

import snapatac2._snapatac2 as internal

def make_tile_matrix(
    output: str,
    fragment_file: str,
    gtf_file: str,
    chrom_size,
    min_num_fragments: int = 200,
    min_tsse: float = 1,
    bin_size: int = 500,
    sorted_by_barcode: bool = True,
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
    gtf_file
        annotation file in GTF format
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
    n_jobs
        number of CPUs to use
    
    Returns
    -------
    AnnData
    """
    if sorted_by_barcode:
        internal.mk_tile_matrix(
            output, fragment_file, gtf_file, chrom_size,
            bin_size, min_num_fragments, min_tsse, n_jobs
        )
    else:
        internal.mk_tile_matrix_unsorted(
            output, fragment_file, gtf_file, chrom_size,
            bin_size, min_num_fragments, min_tsse
        )
    return ad.read(output, backed='r+')

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