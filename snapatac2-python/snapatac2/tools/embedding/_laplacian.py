import numpy as np

from typing import Optional, Union
from scipy.sparse.linalg import svds

from snapatac2._snapatac2 import AnnData
from ._spectral import JaccardNormalizer, jaccard_similarity

def laplacian(
    data: AnnData,
    n_comps: Optional[int] = None,
    features: Optional[Union[str, np.ndarray]] = "selected",
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Convert chromatin accessibility profiles of cells into lower dimensional representations.

    Convert chromatin accessibility profiles of cells into lower dimensional representations
    using the spectrum of the normalized graph Laplacian defined by pairwise similarity
    between cells.

    Parameters
    ----------
    data
        AnnData object
    n_comps
        Number of dimensions to keep
    features
        Boolean index mask. True means that the feature is kept.
        False means the feature is removed.
    random_state
        Seed of the random state generator

    Returns
    -------
    if `inplace=True` it stores Spectral embedding of data in the field
    `adata.obsm["X_spectral"]`, otherwise it returns the result as a numpy array.
    """
    features = data.var[features] if isinstance(features, str) else features
    if n_comps is None:
        min_dim = min(data.n_vars, data.n_obs)
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50
    (n_sample, _) = data.shape

    X = data.X[...]
    X.data = np.ones(X.indices.shape, dtype=np.float64)

    if features is not None: X = X[:, features]

    result = _laplacian(X, n_comps, distance="jaccard")
    if inplace:
        data.obsm['X_spectral'] = result
    else:
        return result

def _laplacian(mat, n_dim, distance="jaccard"):
    from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

    if distance == "jaccard":
        compute_similarity = jaccard_similarity
    elif distance == "cosine":
        compute_similarity = cosine_similarity
    else:
        compute_similarity = rbf_kernel

    dim = mat.shape[1]
    coverage = mat.sum(axis=1) / dim
    print("Compute similarity matrix")
    A = compute_similarity(mat)
    if distance == "jaccard":
        print("Normalization")
        normalizer = JaccardNormalizer(A, coverage)
        normalizer.normalize(A, coverage, coverage)
        np.fill_diagonal(A, 0)
        # Remove outlier
        normalizer.outlier = np.quantile(np.asarray(A), 0.999)
        np.clip(A, a_min=0, a_max=normalizer.outlier, out=A)
    else:
        np.fill_diagonal(A, 0)

    np.divide(A, A.sum(axis=1), out=A)

    print("Perform decomposition")
    u, s, _ = svds(A, k = n_dim, return_singular_vectors="u")
    ix = s.argsort()[::-1]
    s = s[ix]
    u = u[:, ix]
    return np.multiply(u, s)