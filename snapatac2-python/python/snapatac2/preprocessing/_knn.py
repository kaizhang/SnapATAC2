from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.sparse import csr_matrix

from snapatac2._utils import is_anndata
import snapatac2._snapatac2 as internal

def knn(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    n_neighbors: int = 50,
    use_dims: int | list[int] | None = None,
    use_rep: str = 'X_spectral',
    method: Literal['kdtree', 'hora', 'pynndescent'] = "kdtree",
    inplace: bool = True,
    random_state: int = 0,
) -> csr_matrix | None:
    """
    Compute a neighborhood graph of observations.

    Computes a neighborhood graph of observations stored in `adata` using
    the method specified by `method`. The distance metric used is Euclidean.

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
        Can be one of the following:
        - 'kdtree': use the kdtree algorithm to find the nearest neighbors.
        - 'hora': use the HNSW algorithm to find the approximate nearest neighbors.
        - 'pynndescent': use the pynndescent algorithm to find the approximate nearest neighbors.
    inplace
        Whether to store the result in the anndata object.
    random_state
        Random seed for approximate nearest neighbor search.
        Note that this is only used when `method='pynndescent'`.
        Currently 'hora' does not support random seed, so the result of 'hora' is not reproducible.

    Returns
    -------
    csr_matrix | None
        if `inplace=True`, store KNN in `.obsp['distances']`.
        Otherwise, return a sparse matrix.
    """
    if is_anndata(adata):
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
        adj = internal.approximate_nearest_neighbour_graph(
            data.astype(np.float32), n_neighbors)
    elif method == 'pynndescent':
        import pynndescent
        index = pynndescent.NNDescent(data, n_neighbors=max(50, n_neighbors), random_state=random_state)
        adj, distances = index.neighbor_graph
        indices = np.ravel(adj[:, :n_neighbors])
        distances = np.ravel(distances[:, :n_neighbors]) 
        indptr = np.arange(0, distances.size + 1, n_neighbors)
        adj = csr_matrix((distances, indices, indptr), shape=(n, n))
        adj.sort_indices()
    elif method == 'kdtree':
        adj = internal.nearest_neighbour_graph(data, n_neighbors)
    else:
        raise ValueError("method must be one of 'hora', 'pynndescent', 'kdtree'")
    
    if inplace:
        adata.obsp['distances'] = adj
    else:
        return adj