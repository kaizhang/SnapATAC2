from typing import Optional, Union, List
import numpy as np
from scipy.sparse import csr_matrix

from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def knn(
    adata: Union[AnnData, AnnDataSet, np.ndarray],
    n_neighbors: int = 50,
    use_dims: Optional[Union[int, List[int]]] = None,
    use_rep: Optional[str] = None,
    use_approximate_search: bool = True,
    n_jobs: int = -1,
    inplace: bool = True,
) -> Optional[csr_matrix]:
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
    if `inplace`, store KNN in `.obsp['distances']`. Otherwise, return a sparse
    matrix.
    """
    if isinstance(adata, AnnData) or isinstance(adata, AnnDataSet):
        if use_rep is None: use_rep = "X_spectral"
        data = adata.obsm[use_rep]
    else:
        data = adata

    if use_dims is not None:
        if isinstance(use_dims, int):
            data = data[:, :use_dims]
        else:
            data = data[:, use_dims]

    n = data.shape[0]
    if use_approximate_search:
        (d, indices, indptr) = internal.approximate_nearest_neighbors(data.astype(np.float32), n_neighbors)
        adj = csr_matrix((d, indices, indptr), shape=(n, n))
    else:
        from sklearn.neighbors import kneighbors_graph
        adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
    
    if inplace:
        adata.obsp['distances'] = adj
    else:
        return adj