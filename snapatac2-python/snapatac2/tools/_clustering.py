from __future__ import annotations
from typing_extensions import Literal

import scipy.sparse as ss
import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2._utils import get_igraph_from_adjacency, is_anndata 

def leiden(
    adata: AnnData | AnnDataSet | ss.spmatrix,
    resolution: float = 1,
    objective_function: Literal['CPM', 'modularity', 'RBConfiguration'] = 'modularity',
    min_cluster_size: int = 5,
    n_iterations: int = -1,
    random_state: int = 0,
    key_added: str = 'leiden',
    use_leidenalg: bool = False,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Cluster cells into subgroups [Traag18]_.

    Cluster cells using the Leiden algorithm [Traag18]_,
    an improved version of the Louvain algorithm [Blondel08]_.
    It has been proposed for single-cell analysis by [Levine15]_.
    This requires having ran :func:`~snapatac2.pp.knn`.

    Parameters
    ----------
    adata
        The annotated data matrix or sparse adjacency matrix of the graph,
        defaults to neighbors connectivities.
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn't accept a `resolution_parameter`.
    objective_function
        whether to use the Constant Potts Model (CPM) or modularity.
        Must be either "CPM", "modularity" or "RBConfiguration".
    min_cluster_size
        The minimum size of clusters.
    n_iterations
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    random_state
        Change the initialization of the optimization.
    key_added
        `adata.obs` key under which to add the cluster labels.
    use_leidenalg
        If `True`, `leidenalg` package is used. Otherwise, `python-igraph` is used.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        If `inplace=True`, update `adata.obs[key_added]` to store an array of
        dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell. Otherwise, returns the array directly.
    """
    from collections import Counter
    import polars

    if is_anndata(adata):
        adjacency = adata.obsp["distances"]
    else:
        inplace = False
        adjacency = adata
    
    gr = get_igraph_from_adjacency(adjacency)

    if use_leidenalg or objective_function == "RBConfiguration":
        import leidenalg
        from leidenalg.VertexPartition import MutableVertexPartition

        if objective_function == "modularity":
            partition_type = leidenalg.ModularityVertexPartition
        elif objective_function == "CPM":
            partition_type = leidenalg.CPMVertexPartition
        elif objective_function == "RBConfiguration":
            partition_type = leidenalg.RBConfigurationVertexPartition
        else:
            raise ValueError(
                'objective function is not supported: ' + partition_type
            )

        partition = leidenalg.find_partition(
            gr, partition_type, n_iterations=n_iterations,
            seed=random_state, resolution_parameter=resolution, weights=None
        )
    else:
        from igraph import set_random_number_generator
        import random
        random.seed(random_state)
        set_random_number_generator(random)
        partition = gr.community_leiden(
            objective_function=objective_function,
            weights=None,
            resolution_parameter=resolution,
            beta=0.01,
            initial_membership=None,
            n_iterations=n_iterations,
        )

    groups = partition.membership

    new_cl_id = dict([(cl, i) if count >= min_cluster_size else (cl, -1) for (i, (cl, count)) in enumerate(Counter(groups).most_common())])
    for i in range(len(groups)): groups[i] = new_cl_id[groups[i]]

    groups = np.array(groups, dtype=np.compat.unicode)
    if inplace:
        adata.obs[key_added] = polars.Series(
            groups,
            dtype=polars.datatypes.Categorical,
        )
        # store information on the clustering parameters
        #adata.uns['leiden'] = {}
        #adata.uns['leiden']['params'] = dict(
        #    resolution=resolution,
        #    random_state=random_state,
        #    n_iterations=n_iterations,
        #)
    else:
        return groups

def kmeans(
    adata: AnnData | AnnDataSet | np.ndarray,
    n_clusters: int,
    n_iterations: int = -1,
    random_state: int = 0,
    use_rep: str = "X_spectral",
    key_added: str = 'kmeans',
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Cluster cells into subgroups using the K-means algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_clusters
        Number of clusters to return.
    n_iterations
        How many iterations of the kmeans clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
    random_state
        Change the initialization of the optimization.
    use_rep
        Which data in `adata.obsm` to use for clustering. Default is "X_spectral".
    key_added
        `adata.obs` key under which to add the cluster labels.

    Returns
    -------
    adds fields to `adata`:
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    `adata.uns['kmeans']['params']`
        A dict with the values for the parameters `n_clusters`, `random_state`,
        and `n_iterations`.
    """
    import polars

    if is_anndata(adata):
        data = adata.obsm[use_rep]
    else:
        data = adata
    groups = _snapatac2.kmeans(n_clusters, data)
    groups = np.array(groups, dtype=np.compat.unicode)
    if inplace:
        adata.obs[key_added] = polars.Series(
            groups,
            dtype=polars.datatypes.Categorical,
        )
        # store information on the clustering parameters
        #adata.uns['kmeans'] = {}
        #adata.uns['kmeans']['params'] = dict(
        #    n_clusters=n_clusters,
        #    random_state=random_state,
        #    n_iterations=n_iterations,
        #)

    else:
        return groups

def hdbscan(
    adata: AnnData,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    cluster_selection_method: str = "eom",
    random_state: int = 0,
    use_rep: str = "X_spectral",
    key_added: str = 'hdbscan',
    **kwargs,
) -> None:
    """
    Cluster cells into subgroups using the HDBSCAN algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    min_cluster_size
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of
        a cluster rather than a cluster splitting into two new clusters.
    min_samples
        The number of samples in a neighbourhood for a point to be considered a core point.
    cluster_selection_epsilon
        A distance threshold. Clusters below this value will be merged.
    alpha
        A distance scaling parameter as used in robust single linkage.
    cluster_selection_method
        The method used to select clusters from the condensed tree.
        The standard approach for HDBSCAN* is to use an Excess of Mass
        algorithm to find the most persistent clusters.
        Alternatively you can instead select the clusters at the leaves of
        the tree - this provides the most fine grained and homogeneous clusters.
        Options are: "eom" or "leaf".
    random_state
        Change the initialization of the optimization.
    use_rep
        Which data in `adata.obsm` to use for clustering. Default is "X_spectral".
    key_added
        `adata.obs` key under which to add the cluster labels.

    Returns
    -------
    adds fields to `adata`:
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    """
    import pandas as pd
    import hdbscan

    data = adata.obsm[use_rep][...]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples = min_samples,
        cluster_selection_epsilon = cluster_selection_epsilon,
        alpha = alpha,
        cluster_selection_method = cluster_selection_method,
        **kwargs
    )
    clusterer.fit(data)
    groups = clusterer.labels_
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )

def dbscan(
    adata: AnnData,
    eps: float = 0.5,
    min_samples: int = 5,
    leaf_size: int = 30,
    n_jobs: int | None = None,
    use_rep: str = "X_spectral",
    key_added: str = 'dbscan',
) -> None:
    """
    Cluster cells into subgroups using the DBSCAN algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    eps
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. This is the most important
        DBSCAN parameter to choose appropriately for your data set and distance function.
    min_samples
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    leaf_size
        Leaf size passed to BallTree or cKDTree. This can affect the speed of
        the construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.
    n_jobs
        The number of parallel jobs to run. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    use_rep
        Which data in `adata.obsm` to use for clustering. Default is "X_spectral".
    key_added
        `adata.obs` key under which to add the cluster labels.

    Returns
    -------
    adds fields to `adata`:
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    """
    from sklearn.cluster import DBSCAN
    import pandas as pd

    data = adata.obsm[use_rep][...]

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean',
        leaf_size=leaf_size,
        n_jobs=n_jobs).fit(data)
    groups = clustering.labels_
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )