from typing import Optional, Union, Type
import pandas as pd
import numpy as np
from anndata.experimental import AnnCollection
import anndata as ad
import snapatac2._snapatac2 as _snapatac2

def kmeans(
    adata: Union[ad.AnnData, AnnCollection],
    n_clusters: int,
    n_iterations: int = -1,
    random_state: int = 0,
    use_rep: Optional[str] = None,
    key_added: str = 'kmeans',
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
        to one that doesn't accept a `resolution_parameter`.
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

    if use_rep is None:
        data = adata.obsm["X_spectral"]
    else:
        data = adata.obsm[use_rep]

    groups = _snapatac2.kmeans(n_clusters, data)
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )
    # store information on the clustering parameters
    adata.uns['kmeans'] = {}
    adata.uns['kmeans']['params'] = dict(
        n_clusters=n_clusters,
        random_state=random_state,
        n_iterations=n_iterations,
    )