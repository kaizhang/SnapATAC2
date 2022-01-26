import numpy as np
import anndata as ad
from typing import Optional, Union
from anndata.experimental import AnnCollection
from scipy.stats import zscore

import snapatac2._snapatac2

def make_tile_matrix(
    output: str,
    fragment_file: str,
    gtf_file: str,
    chrom_size,
    min_num_fragments: int = 100,
    min_tsse: float = 0,
    bin_size: int = 500,
    n_jobs: int = 4,
) -> ad.AnnData:
    """
    Generate cell by bin count matrix.

    Parameters
    ----------
    output
        file name for saving the result
    fragment_file
        gzipped fragment file
    gtf_file
        gzipped annotation file in GTF format
    chrom_size
        chromosome sizes
    min_num_fragments
        jsdkf
    
    Returns
    -------
    AnnData
    """
    _snapatac2.mk_tile_matrix(output, fragment_file, gtf_file, chrom_size, bin_size, min_num_fragments, min_tsse, n_jobs)
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
        count = np.ravel(adata.X.sum(axis = 0))
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
    if variable_feature:
        # TODO: exclude 0 from zscore calculation
        selected_features &= zscore(count) < 1.65
    if whitelist is not None:
        selected_features &= _snapatac2.intersect_bed(list(adata.var_names), whitelist)
    if blacklist is not None:
        selected_features &= not _snapatac2.intersect_bed(list(adata.var_names), blacklist)
    return selected_features