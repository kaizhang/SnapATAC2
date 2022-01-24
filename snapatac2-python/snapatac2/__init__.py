import pandas as pd
import numpy as np
import anndata as ad
from typing import Optional, Union
from anndata.experimental import AnnCollection
from scipy.stats import zscore

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

GRCm39 = {
    "chr1": 195154279,
    "chr2": 181755017,
    "chr3": 159745316,
    "4": 156860686,
    "5": 151758149,
    "6": 149588044,
    "7": 144995196,
    "8": 130127694,
    "9": 124359700,
    "10": 130530862,
    "11": 121973369,
    "12": 120092757,
    "13": 120883175,
    "14": 125139656,
    "15": 104073951,
    "16": 98008968,
    "17": 95294699,
    "18": 90720763,
    "19": 61420004,
    "X": 169476592,
    "Y": 91455967,
}

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
    mk_tile_matrix(output, fragment_file, gtf_file, chrom_size, bin_size, min_num_fragments, min_tsse, n_jobs)
    return ad.read(output, backed='r+')

def find_variable_features(
    adata: Union[ad.AnnData, AnnCollection]
) -> np.ndarray:
    """
    cells_subsetndarray
    Boolean index mask that does filtering. True means that the cell is kept. False means the cell is removed.
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
    # TODO: exclude 0 from zscore calculation
    selected_features = np.logical_and(zscore(count) < 1.65, count != 0)
    return selected_features

# FIXME: random state
def spectral(
    data: ad.AnnData,
    n_comps: Optional[int] = None,
    features: Optional[np.ndarray] = None,
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
    None
    """
    if n_comps is None:
        min_dim = min(data.n_vars, data.n_obs)
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50
    (n_sample, _) = data.shape

    if sample_size is None or sample_size >= n_sample:
        if isinstance(data, AnnCollection):
            X, _ = next(data.iterate_axis(n_sample))
            X = X.X[...]
            X.data = np.ones(X.indices.shape, dtype=np.float64)
        else:
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

        if features is not None: X = X[:, features]

        model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=1)
        model.fit(X)
        data.obsm['X_spectral'] = model.evecs[:, 1:]

    else:
        if isinstance(data, AnnCollection):
            S, sample_indices = next(data.iterate_axis(sample_size, shuffle=True))
            S = S.X[...]
            S.data = np.ones(S.indices.shape, dtype=np.float64)
        else:
            pass

        if features is not None: S = S[:, features]

        model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=sample_size / n_sample)
        model.fit(S)
        if chunk_size is None: chunk_size = 2000

        from tqdm import tqdm
        import math
        result = []
        print("Perform Nystrom extension")
        for batch, _ in tqdm(data.iterate_axis(chunk_size), total = math.ceil(n_sample / chunk_size)):
            batch = batch.X[...]
            batch.data = np.ones(batch.indices.shape, dtype=np.float64)
            if features is not None: batch = batch[:, features]
            result.append(model.predict(batch)[:, 1:])
        data.obsm['X_spectral'] = np.concatenate(result, axis=0)

def umap(
    data: ad.AnnData,
    n_comps: int = 2,
    random_state: int = 0,
) -> None:
    """
    Parameters
    ----------
    data
        AnnData

    Returns
    -------
    None
    """
    from umap import UMAP
    data.obsm["X_umap"] = UMAP(random_state=random_state, n_components=n_comps).fit_transform(data.obsm["X_spectral"])