from typing import Optional, Union, Type
import pandas as pd
import scipy.sparse as ss
import numpy as np
import leidenalg as la
from sklearn.neighbors import kneighbors_graph
from anndata.experimental import AnnCollection
import igraph as ig
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

# TODO: leiden function in igraph C library is much faster
def leiden(
    adata: Union[ad.AnnData, AnnCollection],
    resolution: float = 1,
    partition_type: MutableVertexPartition = la.RBConfigurationVertexPartition,
    use_weights: bool = True,
    n_iterations: int = -1,
    random_state: int = 0,
    key_added: str = 'leiden',
    adjacency: Optional[ss.spmatrix] = None,
):
    """
    Cluster cells into subgroups [Traag18]_.

    Cluster cells using the Leiden algorithm [Traag18]_,
    an improved version of the Louvain algorithm [Blondel08]_.
    It has been proposed for single-cell analysis by [Levine15]_.
    This requires having ran :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    random_state
        Change the initialization of the optimization.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain `(obs_key, list_of_categories)`.
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
    directed
        Whether to treat the graph as directed or undirected.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    partition_type
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    neighbors_key
        Use neighbors connectivities as adjacency.
        If not specified, leiden looks .obsp['connectivities'] for connectivities
        (default storage place for pp.neighbors).
        If specified, leiden looks
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    obsp
        Use .obsp[obsp] as adjacency. You can't specify both
        `obsp` and `neighbors_key` at the same time.
    copy
        Whether to copy `adata` or modify it inplace.
    **partition_kwargs
        Any further arguments to pass to `~leidenalg.find_partition`
        (which in turn passes arguments to the `partition_type`).

    Returns
    -------
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    `adata.uns['leiden']['params']`
        A dict with the values for the parameters `resolution`, `random_state`,
        and `n_iterations`.
    """

    if adjacency is None:
        adjacency = adata.obsp["connectivities"]
    gr = _utils.get_igraph_from_adjacency(adjacency)
    if use_weights:
        weights = gr.es["weight"]
    else:
        weights = None

    partition = la.find_partition(gr, partition_type, n_iterations=n_iterations,
        seed=random_state, resolution_parameter=resolution, weights=weights
    )
    groups = np.array(partition.membership)

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )
    # store information on the clustering parameters
    adata.uns['leiden'] = {}
    adata.uns['leiden']['params'] = dict(
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
    )