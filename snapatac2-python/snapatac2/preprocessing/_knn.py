from __future__ import annotations
from typing_extensions import Literal

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
    method: Literal['hora', 'pynndescent', 'exact'] = "hora",
    n_jobs: int = -1,
    inplace: bool = True,
    random_state: int = 0,
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
    method
        'hora', 'pynndescent', or 'exact'
    n_jobs
        number of CPUs to use
    inplace
        Whether to store the result in the anndata object.
    random_state
        Random seed for approximate nearest neighbor search.

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
    if method == 'hora':
        (d, indices, indptr) = approximate_nearest_neighbors(data.astype(np.float32), n_neighbors)
        adj = csr_matrix((d, indices, indptr), shape=(n, n))
    elif method == 'pynndescent':
        import pynndescent
        index = pynndescent.NNDescent(data, n_neighbors=max(50, n_neighbors), random_state=random_state)
        adj, distances = index.neighbor_graph
        indices = np.ravel(adj[:, :n_neighbors])
        distances = np.ravel(distances[:, :n_neighbors]) 
        indptr = np.arange(0, distances.size + 1, n_neighbors)
        adj = csr_matrix((distances, indices, indptr), shape=(n, n))
    else:
        from sklearn.neighbors import kneighbors_graph
        adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
        adj.sort_indices()
    
    if inplace:
        adata.obsp['distances'] = adj
    else:
        return adj