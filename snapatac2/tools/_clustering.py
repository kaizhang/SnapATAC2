from __future__ import annotations

from typing import Literal
import scipy.sparse as ss
import numpy as np

import snapatac2
import snapatac2._snapatac2 as internal
from snapatac2._utils import get_igraph_from_adjacency, is_anndata

def leiden(
    adata: internal.AnnData | internal.AnnDataSet | ss.spmatrix,
    resolution: float = 1,
    objective_function: Literal["CPM", "modularity"] = "modularity",
    min_cluster_size: int = 5,
    n_iterations: int = -1,
    random_state: int = 0,
    key_added: str = "leiden",
    weighted: bool = False,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Cluster cells with the Leiden community detection algorithm.

    Use this function after building a nearest-neighbor graph with
    :func:`snapatac2.pp.knn`, or pass a sparse adjacency matrix directly.

    Anti-Patterns
    -------------
    - Do NOT pass raw embeddings as `adata`; pass an AnnData object with
      `adata.obsp["distances"]` or a sparse graph matrix.
    - Do NOT expect clusters smaller than `min_cluster_size` to keep their
      original labels; they are relabeled as `"-1"`.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | scipy.sparse.spmatrix
        Annotated data object containing `adata.obsp["distances"]`, or a sparse
        graph adjacency matrix.
    resolution : float
        Clustering resolution. Larger values usually produce more clusters.
    objective_function : {"CPM", "modularity"}
        Leiden objective function.
    min_cluster_size : int
        Minimum retained cluster size. Smaller clusters are labeled `"-1"`.
    n_iterations : int
        Number of Leiden optimization iterations. Use `-1` to run until
        convergence.
    random_state : int
        Seed for Leiden initialization.
    key_added : str
        Key in `adata.obs` used to store cluster labels.
    weighted : bool
        If True, transform graph distances into edge weights before clustering.
    inplace : bool
        If True, store labels in `adata.obs[key_added]`; if False, return them.

    Returns
    -------
    np.ndarray | None
        If `inplace=True`, stores categorical labels in `adata.obs[key_added]`
        and returns None. If `inplace=False`, returns a string array of labels.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.pp.knn(adata, use_rep="X_spectral")
    >>> snap.tl.leiden(adata, resolution=1.0)
    >>> "leiden" in adata.obs
    True
    """
    try:
        from igraph import set_random_number_generator
    except ImportError:
        raise ImportError(
            "tl.leiden requires python-igraph, an optional dependency of SnapATAC2. "
            "Install it with:  pip install 'snapatac2[clustering]'  (or: pip install igraph). "
            "It is optional because python-igraph is GPL-licensed while SnapATAC2 is MIT, so "
            "users who never call tl.leiden, tl.leiden_sweep or tl.pagerank do not have to "
            "take on a copyleft dependency."
        ) from None
    from collections import Counter
    import polars
    import random

    random.seed(random_state)
    set_random_number_generator(random)

    if is_anndata(adata):
        adjacency = adata.obsp["distances"]
    else:
        inplace = False
        adjacency = adata

    gr = get_igraph_from_adjacency(adjacency)

    if weighted:
        weights = np.array(gr.es["weight"])
        weights = np.exp(-weights)
    else:
        weights = None

    groups = gr.community_leiden(
        objective_function=objective_function,
        weights=weights,
        resolution=resolution,
        beta=0.01,
        initial_membership=None,
        n_iterations=n_iterations,
    ).membership

    new_cl_id = dict(
        [
            (cl, i) if count >= min_cluster_size else (cl, -1)
            for (i, (cl, count)) in enumerate(Counter(groups).most_common())
        ]
    )
    for i in range(len(groups)):
        groups[i] = new_cl_id[groups[i]]

    groups = np.array(groups, dtype=np.str_)
    if inplace:
        adata.obs[key_added] = polars.Series(
            groups,
            dtype=polars.datatypes.Categorical,
        )
    else:
        return groups


def leiden_sweep(
    adata: internal.AnnData | internal.AnnDataSet | ss.spmatrix,
    resolutions: list[float],
    use_rep: str | np.ndarray = "X_spectral",
    objective_function: Literal["CPM", "modularity", "RBConfiguration"] = "modularity",
    min_cluster_size: int = 5,
    n_iterations: int = -1,
    random_state: int = 0,
    weighted: bool = False,
    n_jobs: int = 16,
):
    """
    Score Leiden clustering across multiple resolutions.

    Use this function to choose a Leiden resolution by comparing silhouette
    scores computed from an embedding while clustering the existing graph.

    Anti-Patterns
    -------------
    - Do NOT pass an AnnData object without `adata.obsp["distances"]`; run
      :func:`snapatac2.pp.knn` first.
    - Do NOT interpret the highest silhouette score as a guaranteed biological
      optimum; inspect marker accessibility and cluster sizes as well.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | scipy.sparse.spmatrix
        Annotated data object containing `adata.obsp["distances"]`, or a sparse
        graph adjacency matrix.
    resolutions : list[float]
        Resolution values to evaluate.
    use_rep : str | np.ndarray
        Embedding used for silhouette scoring. If `adata` is a graph matrix,
        pass the embedding array directly.
    objective_function : {"CPM", "modularity", "RBConfiguration"}
        Objective function passed to :func:`leiden`.
    min_cluster_size : int
        Minimum retained cluster size.
    n_iterations : int
        Number of Leiden optimization iterations. Use `-1` to run until
        convergence.
    random_state : int
        Seed for Leiden initialization.
    weighted : bool
        If True, use graph edge weights.
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    list[dict]
        One dictionary per resolution with keys `"resolution"`, `"n_clusters"`,
        and `"silhouette_score"`.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.pp.knn(adata, use_rep="X_spectral")
    >>> scores = snap.tl.leiden_sweep(adata, [0.5, 1.0], n_jobs=1)
    >>> sorted(scores[0])
    ['n_clusters', 'resolution', 'silhouette_score']
    """
    from sklearn.metrics import silhouette_score
    from multiprocess import get_context

    if is_anndata(adata):
        distances = adata.obsp["distances"]
        mat = adata.obsm[use_rep]
    else:
        mat = use_rep
        distances = adata

    def _func(resolution):
        groups = leiden(
            distances,
            resolution,
            objective_function=objective_function,
            min_cluster_size=min_cluster_size,
            n_iterations=n_iterations,
            random_state=random_state,
            weighted=weighted,
            inplace=False,
        )
        if len(set(groups)) > 1:
            score = silhouette_score(
                mat,
                groups,
                sample_size=20000,
            )
        else:
            score = 0
        return  {
            "resolution": resolution,
            "n_clusters": len(set(groups)),
            "silhouette_score": score,
        }

    with get_context("spawn").Pool(n_jobs) as p:
        return list(p.imap(_func, resolutions))

def kmeans(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    n_clusters: int,
    n_iterations: int = -1,
    random_state: int = 0,
    use_rep: str = "X_spectral",
    key_added: str = "kmeans",
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Cluster cells with k-means.

    Use this function on a dense embedding such as `adata.obsm["X_spectral"]`,
    or pass a NumPy array directly and set `inplace=False`.

    Anti-Patterns
    -------------
    - Do NOT pass a raw count matrix unless k-means on counts is intended; use a
      normalized embedding for typical single-cell workflows.
    - Do NOT rely on `n_iterations` or `random_state` to alter the current Rust
      backend call; they are retained in the API but not forwarded here.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | np.ndarray
        Annotated data object containing `adata.obsm[use_rep]`, or a numeric
        matrix with cells as rows.
    n_clusters : int
        Number of clusters to compute.
    n_iterations : int
        API parameter reserved for k-means iteration control.
    random_state : int
        API parameter reserved for initialization control.
    use_rep : str
        Key in `adata.obsm` containing the input embedding.
    key_added : str
        Key in `adata.obs` used to store cluster labels.
    inplace : bool
        If True, store labels in `adata.obs[key_added]`; if False, return them.

    Returns
    -------
    np.ndarray | None
        If `inplace=True`, stores categorical labels in `adata.obs[key_added]`
        and returns None. If `inplace=False`, returns a string array of labels.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> X = np.random.default_rng(0).normal(size=(12, 3))
    >>> labels = snap.tl.kmeans(X, n_clusters=3, inplace=False)
    >>> labels.shape
    (12,)
    """
    import polars

    if is_anndata(adata):
        data = adata.obsm[use_rep]
    else:
        data = adata
    groups = internal.kmeans(n_clusters, data)
    groups = np.array(groups, dtype=np.str_)
    if inplace:
        adata.obs[key_added] = polars.Series(
            groups,
            dtype=polars.datatypes.Categorical,
        )
        # store information on the clustering parameters
        # adata.uns['kmeans'] = {}
        # adata.uns['kmeans']['params'] = dict(
        #    n_clusters=n_clusters,
        #    random_state=random_state,
        #    n_iterations=n_iterations,
        # )

    else:
        return groups


def hdbscan(
    adata: internal.AnnData,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    cluster_selection_method: str = "eom",
    random_state: int = 0,
    use_rep: str = "X_spectral",
    key_added: str = "hdbscan",
    **kwargs,
) -> None:
    """
    Cluster cells with HDBSCAN.

    Use this function to detect variable-density clusters and label noise cells
    from an embedding stored in `adata.obsm[use_rep]`.

    Anti-Patterns
    -------------
    - Do NOT expect every cell to receive a cluster label; HDBSCAN labels noise
      cells as `-1`.
    - Do NOT pass raw fragment counts as `use_rep`; use a low-dimensional
      embedding for typical workflows.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing `adata.obsm[use_rep]`.
    min_cluster_size : int
        Minimum cluster size; single linkage splits that contain
        fewer points than this will be considered points "falling out" of
        a cluster rather than a cluster splitting into two new clusters.
    min_samples : int | None
        Number of samples in a neighborhood for a point to be considered a core
        point. If None, HDBSCAN chooses its default from `min_cluster_size`.
    cluster_selection_epsilon : float
        A distance threshold. Clusters below this value will be merged.
    alpha : float
        A distance scaling parameter as used in robust single linkage.
    cluster_selection_method : str
        Cluster extraction method, usually `"eom"` or `"leaf"`.
    random_state : int
        API parameter reserved for consistency with other clustering functions.
    use_rep : str
        Key in `adata.obsm` containing the input embedding.
    key_added : str
        Key in `adata.obs` used to store cluster labels.
    **kwargs
        Additional keyword arguments passed to `hdbscan.HDBSCAN`.

    Returns
    -------
    None
        Stores categorical labels in `adata.obs[key_added]`.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.tl.hdbscan(adata, min_cluster_size=20)
    >>> "hdbscan" in adata.obs
    True
    """
    import pandas as pd
    import hdbscan

    data = adata.obsm[use_rep][...]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        cluster_selection_method=cluster_selection_method,
        **kwargs,
    )
    clusterer.fit(data)
    groups = clusterer.labels_
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=sorted(map(str, np.unique(groups))),
    )


def dbscan(
    adata: internal.AnnData,
    eps: float = 0.5,
    min_samples: int = 5,
    leaf_size: int = 30,
    n_jobs: int | None = None,
    use_rep: str = "X_spectral",
    key_added: str = "dbscan",
) -> None:
    """
    Cluster cells with DBSCAN.

    Use this function to identify density-connected groups and noise cells from
    an embedding stored in `adata.obsm[use_rep]`.

    Anti-Patterns
    -------------
    - Do NOT expect DBSCAN to assign every cell to a cluster; noise cells are
      labeled as `-1`.
    - Do NOT reuse `eps` across embeddings with different scales; tune it for
      the representation passed through `use_rep`.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing `adata.obsm[use_rep]`.
    eps : float
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. This is the most important
        DBSCAN parameter to choose appropriately for your data set and distance function.
    min_samples : int
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    leaf_size : int
        Leaf size passed to BallTree or cKDTree. This can affect the speed of
        the construction and query, as well as the memory required to store the
        tree. The optimal value depends on the nature of the problem.
    n_jobs : int | None
        The number of parallel jobs to run. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    use_rep : str
        Key in `adata.obsm` containing the input embedding.
    key_added : str
        Key in `adata.obs` used to store cluster labels.

    Returns
    -------
    None
        Stores categorical labels in `adata.obs[key_added]`.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.tl.dbscan(adata, eps=0.5, min_samples=5)
    >>> "dbscan" in adata.obs
    True
    """
    from sklearn.cluster import DBSCAN
    import pandas as pd

    data = adata.obsm[use_rep][...]

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        leaf_size=leaf_size,
        n_jobs=n_jobs,
    ).fit(data)
    groups = clustering.labels_
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=sorted(map(str, np.unique(groups))),
    )
