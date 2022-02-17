from typing import Optional, Union, Type
import pandas as pd
from typing import Optional
import numpy as np
from anndata.experimental import AnnCollection
import anndata as ad

def dbscan(
    adata: Union[ad.AnnData, AnnCollection],
    eps: float = 0.5,
    min_samples: int = 5,
    leaf_size: int = 30,
    n_jobs: Optional[int] = None,
    use_rep: Optional[str] = None,
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
    if use_rep is None: use_rep = "X_spectral"
    data = adata.obsm[use_rep]

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