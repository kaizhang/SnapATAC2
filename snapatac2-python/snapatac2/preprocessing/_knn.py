from typing import Optional, Union, Type
import numpy as np
from sklearn.neighbors import kneighbors_graph
from anndata.experimental import AnnCollection
from scipy.sparse import csr_matrix
import anndata as ad

import snapatac2._snapatac2 as internal

def knn(
    adata: Union[ad.AnnData, AnnCollection],
    n_neighbors: int = 15,
    use_rep: Optional[str] = None,
    use_approximate_search: bool = False,
    n_jobs: int = -1,
) -> None:
    """
    Compute a neighborhood graph of observations.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The number of nearest neighbors to be searched.
    use_rep
    use_approximate_search
        Whether to use approximate nearest neighbor search
    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:
    See `key_added` parameter description for the storage path of
    connectivities and distances.
    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    """
    if use_rep is None: use_rep = "X_spectral"
    data = adata.obsm[use_rep]
    n = data.shape[0]
    if use_approximate_search:
        (d, indices, indptr) = internal.approximate_nearest_neighbors(data.astype(np.float32), n_neighbors)
        adj = csr_matrix((d, indices, indptr), shape=(n, n))
    else:
        adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
    adata.obsp['distances'] = adj