import pandas as pd
import numpy as np
import anndata as ad
from typing import Optional

from .snapatac2 import *
from ._spectral import Spectral
from .utils import read_as_binarized

hg38 = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415
}

def make_tile_matrix(
    output: str,
    fragment_file: str,
    gtf_file: str,
    chrom_size,
    min_num_fragments: int = 100,
    bin_size: int = 500,
) -> ad.AnnData:
    """Generate cell by bin count matrix.

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
    min_num_fragments: 
    
    Returns
    -------
    """
    mk_tile_matrix(output, fragment_file, gtf_file, chrom_size, bin_size, min_num_fragments)
    return ad.read(output, backed='r+')

# FIXME: random state
def spectral(
    data: ad.AnnData,
    n_comps: Optional[int] = None,
    random_state: int = 0,
    sample_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    data
        AnnData object
    n_comps
        Number of dimensions to keep
    random_state
        Seed of the random state generator
    sample_size
        Sample size used in the Nystrom method
    chunk_size
        Chunk size used in the Nystrom method
    Returns
    -------
    """
    if n_comps is None:
        min_dim = min(data.n_vars, data.n_obs)
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50

    if data.isbacked:
        if data.is_view:
            raise ValueError(
                "View of AnnData object in backed mode is not supported."
                "To save the object to file, use `.copy(filename=myfilename.h5ad)`."
                "To load the object into memory, use `.to_memory()`.")
        else:
            X = read_as_binarized(data)
    else:
        X = data.X[...]
        X.data = np.ones(X.indices.shape, dtype=np.float64)

    model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=1)
    model.fit(X)
    data.obsm['X_spectral'] = model.evecs[:, 1:]

def umap(
    data: ad.AnnData,
    n_comps: int = 2,
    random_state: int = 0,
) -> None:
    """
    Parameters
    ----------
    Returns
    -------
    """
    from umap import UMAP
    data.obsm["X_umap"] = UMAP(random_state=random_state, n_components=n_comps).fit_transform(data.obsm["X_spectral"])