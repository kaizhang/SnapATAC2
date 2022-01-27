from typing import Optional, Union, Type
import pandas as pd
import scipy.sparse as ss
import numpy as np
import leidenalg as la
from sklearn.neighbors import kneighbors_graph
from anndata.experimental import AnnCollection
import anndata as ad
from leidenalg.VertexPartition import MutableVertexPartition
from .. import _utils

def knn(
    adata: Union[ad.AnnData, AnnCollection],
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
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
    key_added
        If not specified, connectivities .obsp['connectivities'].
        connectivities in .obsp[key_added+'_connectivities'].

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:
    See `key_added` parameter description for the storage path of
    connectivities and distances.
    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """
    if use_rep is None:
        data = adata.obsm["X_spectral"]
    else:
        data = adata.obsm[use_rep]

    adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
    np.reciprocal(adj.data, out=adj.data)
    adata.obsp['connectivities'] = adj

