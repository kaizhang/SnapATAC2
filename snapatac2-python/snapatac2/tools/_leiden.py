from logging import raiseExceptions
from typing import Optional, Union
import pandas as pd
import scipy.sparse as ss
import numpy as np
from anndata.experimental import AnnCollection
import anndata as ad
from .. import _utils

def leiden(
    adata: Union[ad.AnnData, AnnCollection],
    resolution: float = 1,
    objective_function: str = "modularity",
    min_cluster_size: int = 5,
    n_iterations: int = -1,
    random_state: int = 0,
    key_added: str = 'leiden',
    adjacency: Optional[ss.spmatrix] = None,
    use_leidenalg: bool = False,
    inplace: bool = True,
) -> Optional[np.ndarray]:
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
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
    use_leidenalg
        If `True`, `leidenalg` package is used. Otherwise, `python-igraph` is used.
    inplace
        Whether to store the result in the anndata object.

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
    from collections import Counter

    if adjacency is None:
        adjacency = adata.obsp["distances"]
    gr = _utils.get_igraph_from_adjacency(adjacency)

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
    groups = np.array(groups, dtype=np.int64)

    if inplace:
        adata.obs[key_added] = pd.Categorical(
            values=groups.astype('U'),
            categories=map(str, sorted(np.unique(groups))),
        )
        # store information on the clustering parameters
        adata.uns['leiden'] = {}
        adata.uns['leiden']['params'] = dict(
            resolution=resolution,
            random_state=random_state,
            n_iterations=n_iterations,
        )
    else:
        return groups