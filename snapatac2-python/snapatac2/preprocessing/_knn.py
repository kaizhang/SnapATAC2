from typing import Optional, Union, Type
import numpy as np
from sklearn.neighbors import kneighbors_graph
from anndata.experimental import AnnCollection
import anndata as ad

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
    """
    if use_rep is None: use_rep = "X_spectral"
    data = adata.obsm[use_rep]
    n_sample, n_dim = data.shape

    if use_approximate_search:
        ''' TODO: Implement this Rust
        from horapy import HNSWIndex
        index = HNSWIndex(n_dim, "usize")
        for i in range(n_sample): index.add(np.float32(data[i]), i)
        index.build("euclidean")

        target = np.random.randint(0, n)
        # 410 in Hora ANNIndex <HNSWIndexUsize> (dimension: 50, dtype: usize, max_item: 1000000, n_neigh: 32, n_neigh0: 64, ef_build: 20, ef_search: 500, has_deletion: False)
        # has neighbors: [410, 736, 65, 36, 631, 83, 111, 254, 990, 161]
        print("{} in {} \nhas neighbors: {}".format(
            target, index, index.search(samples[target], 10)))  # search
        '''
        pass
    else:
        adj = kneighbors_graph(data, n_neighbors, mode='distance', n_jobs=n_jobs)
        np.reciprocal(adj.data, out=adj.data)
        adata.obsp['connectivities'] = adj

