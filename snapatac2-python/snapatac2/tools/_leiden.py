from typing import Optional, Union
import pandas as pd
import scipy.sparse as ss
import numpy as np
import leidenalg as la
from anndata.experimental import AnnCollection
import anndata as ad
from leidenalg.VertexPartition import MutableVertexPartition
from .. import _utils

# TODO: leiden function in igraph C library is much faster
def leiden(
    adata: Union[ad.AnnData, AnnCollection],
    resolution: float = 1,
    partition_type: MutableVertexPartition = la.RBConfigurationVertexPartition,
    use_weights: bool = False,
    n_iterations: int = -1,
    random_state: int = 0,
    key_added: str = 'leiden',
    adjacency: Optional[ss.spmatrix] = None,
) -> None:
    """
    Cluster cells into subgroups [Traag18]_.

    Cluster cells using the Leiden algorithm [Traag18]_,
    an improved version of the Louvain algorithm [Blondel08]_.
    It has been proposed for single-cell analysis by [Levine15]_.
    This requires having ran :func:`~snapatac2.pp.knn`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    partition_type
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    random_state
        Change the initialization of the optimization.
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.

    Returns
    -------
    adds fields to `adata`:
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
