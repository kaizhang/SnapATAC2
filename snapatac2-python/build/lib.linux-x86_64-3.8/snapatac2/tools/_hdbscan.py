from typing import Optional, Union, Type
import pandas as pd
import numpy as np
from anndata.experimental import AnnCollection
import anndata as ad

def hdbscan(
    adata: Union[ad.AnnData, AnnCollection],
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    cluster_selection_method: str = "eom",
    random_state: int = 0,
    use_rep: Optional[str] = None,
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
    import hdbscan
    if use_rep is None: use_rep = "X_spectral"
    data = adata.obsm[use_rep]
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