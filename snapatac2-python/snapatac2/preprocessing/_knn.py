from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from snapatac2._utils import is_anndata 
from snapatac2._snapatac2 import AnnData, AnnDataSet, approximate_nearest_neighbors

# TODO: add random state
def knn(
    adata: AnnData | AnnDataSet | np.ndarray,
    n_neighbors: int = 50,
    use_dims: int | list[int] | None = None,
    use_rep: str | None = None,
    use_approximate_search: bool = True,
    n_jobs: int = -1,
    inplace: bool = True,
) -> csr_matrix | None:
    """
    Compute a neighborhood graph of observations.

    Parameters
    ----------
    adata
        Annotated data matrix or numpy array.
    n_neighbors
        The number of nearest neighbors to be searched.
    use_dims
        The dimensions used for computation.
    use_rep
        The key for the matrix
    use_approximate_search
        Whether to use approximate nearest neighbor search
    n_jobs
        number of CPUs to use
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    csr_matrix | None
        if `inplace=True`, store KNN in `.obsp['distances']`.
        Otherwise, return a sparse matrix.
    """
    if is_anndata(adata):
        if use_rep is None: use_rep = "X_spectral"
        data = adata.obsm[use_rep]
    else:
        inplace = False
        data = adata
    if data.size == 0:
        raise ValueError("matrix is empty")

    if use_dims is not None:
        if isinstance(use_dims, int):
            data = data[:, :use_dims]
        else:
            data = data[:, use_dims]

    n = data.shape[0]
    if use_approximate_search:
        (d, indices, indptr) = approximate_nearest_neighbors(data.astype(np.float32), n_neighbors)
        adj = csr_matrix((d, indices, indptr), shape=(n, n))
    else:
        from sklearn.neighbors import kneighbors_graph
        adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
    
    if inplace:
        adata.obsp['distances'] = adj
    else:
        return adj