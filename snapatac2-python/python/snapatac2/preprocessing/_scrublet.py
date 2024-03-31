""" Implementation of the scrublet algorithm for single-cell ATAC-seq data
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as ss
import logging
from anndata import AnnData

from .._utils import chunks, anndata_par
import snapatac2._snapatac2 as internal
from snapatac2.tools._embedding import spectral

def scrublet(
    adata: internal.AnnData | list[internal.AnnData],
    features: str | np.ndarray | None = "selected",
    n_comps: int = 15,
    sim_doublet_ratio: float = 2.0,
    expected_doublet_rate: float = 0.1,
    n_neighbors: int | None = None,
    use_approx_neighbors=False,
    random_state: int = 0,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> None:
    """
    Compute probability of being a doublet using the scrublet algorithm.

    This function identifies doublets by generating simulated doublets using
    randomly pairing chromatin accessibility profiles of individual cells.
    The simulated doublets are then embedded alongside the original cells using
    the spectral embedding algorithm in this package.
    A k-nearest-neighbor classifier is trained to distinguish between the simulated
    doublets and the authentic cells.
    This trained classifier produces a "doublet score" for each cell.
    The doublet scores are then converted into probabilities using a Gaussian mixture model.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    features
        Boolean index mask, where `True` means that the feature is kept, and
        `False` means the feature is removed.
    n_comps
        Number of components. 15 is usually sufficient. The algorithm is not sensitive
        to this parameter.
    sim_doublet_ratio
        Number of doublets to simulate relative to the number of observed cells.
    expected_doublet_rate
        Expected doublet rate.
    n_neighbors
        Number of neighbors used to construct the KNN graph of observed
        cells and simulated doublets. If `None`, this is 
        set to round(0.5 * sqrt(n_cells))
    use_approx_neighbors
        Whether to use approximate search.
    random_state
        Random state.
    inplace
        Whether update the AnnData object inplace
    n_jobs
        Number of jobs to run in parallel.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None:
        if ``inplace = True``, it updates adata with the following fields:
            - ``adata.obs["doublet_probability"]``: probability of being a doublet
            - ``adata.obs["doublet_score"]``: doublet score
    """
    if isinstance(adata, list):
        result = anndata_par(
            adata,
            lambda x: scrublet(x, features, n_comps, sim_doublet_ratio,
                               expected_doublet_rate, n_neighbors,
                               use_approx_neighbors, random_state,
                               inplace, n_jobs, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    if isinstance(features, str):
        if features in adata.var:
            features = adata.var[features]
        else:
            raise NameError("Please call `select_features` first or explicitly set `features = None`")

    if features is None:
        count_matrix = adata.X[:]
    else:
        count_matrix = adata.X[:, features]

    if min(count_matrix.shape) == 0: raise NameError("Matrix is empty")

    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(count_matrix.shape[0])))

    doublet_scores_obs, doublet_scores_sim, _, _ = scrub_doublets_core(
        count_matrix, n_neighbors, sim_doublet_ratio, expected_doublet_rate,
        n_comps=n_comps,
        use_approx_neighbors = use_approx_neighbors,
        random_state=random_state,
        verbose=verbose,
    )
    probs = get_doublet_probability(
        doublet_scores_sim, doublet_scores_obs, random_state,
    )
 
    if inplace:
        adata.obs["doublet_probability"] = probs
        adata.obs["doublet_score"] = doublet_scores_obs
        adata.uns["scrublet_sim_doublet_score"] = doublet_scores_sim
    else:
        return probs, doublet_scores_obs

def filter_doublets(
    adata: internal.AnnData | list[internal.AnnData],
    probability_threshold: float | None = 0.5,
    score_threshold: float | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> np.ndarray | None:
    """Remove doublets according to the doublet probability or doublet score.

    The user can choose to remove doublets by either the doublet probability or the doublet score.
    :func:`~snapatac2.pp.scrublet` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    probability_threshold
        Threshold for doublet probability. Doublet probability greater than
        this threshold will be removed. The default value is 0.5. Using a lower
        threshold will remove more cells.
    score_threshold
        Threshold for doublet score. Doublet score greater than this threshold
        will be removed. Only one of `probability_threshold` and `score_threshold`
        can be set. Using `score_threshold` is not recommended for most cases.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of jobs to run in parallel.
    verbose
        Whether to print progress messages.

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return 
        a boolean index mask that does filtering, where `True` means that the
        cell is kept, `False` means the cell is removed.

    See Also
    --------
    scrublet
    """
    if isinstance(adata, list):
        result = anndata_par(
            adata,
            lambda x: filter_doublets(x, probability_threshold, score_threshold,
                                      inplace, n_jobs, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    if probability_threshold is not None and score_threshold is not None:
        raise ValueError("Only one of `probability_threshold` and `score_threshold` can be set.")
    if probability_threshold is not None:
        probs = adata.obs["doublet_probability"].to_numpy()
        is_doublet = probs > probability_threshold
    if score_threshold is not None:
        scores = adata.obs["doublet_score"].to_numpy()
        is_doublet = scores > score_threshold

    doublet_rate = np.mean(is_doublet)
    if verbose: logging.info(f"Detected doublet rate = {doublet_rate*100:.3f}%")

    if inplace:
        adata.uns["doublet_rate"] = doublet_rate
        if adata.isbacked:
            adata.subset(~is_doublet)
        else:
            adata._inplace_subset_obs(~is_doublet)
    else:
        return ~is_doublet

def scrub_doublets_core(
    count_matrix: ss.spmatrix,
    n_neighbors: int,
    sim_doublet_ratio: float,
    expected_doublet_rate: float,
    synthetic_doublet_umi_subsampling: float =1.0,
    n_comps: int = 30,
    use_approx_neighbors: bool = False,
    random_state: int = 0,
    verbose: bool = False,
) -> None:
    """
    Modified scrublet pipeline for single-cell ATAC-seq data.

    Automatically sets a threshold for calling doublets, but it's best to check 
    this by running plot_histogram() afterwards and adjusting threshold 
    with call_doublets(threshold=new_threshold) if necessary.

    Arguments
    ---------
    synthetic_doublet_umi_subsampling
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified
        rate.
    n_comps
        Number of principal components used to embed the transcriptomes prior
        to k-nearest-neighbor graph construction.

    Returns
    -------
    doublet_scores_obs_, doublet_errors_obs_,
    doublet_scores_sim_, doublet_errors_sim_,
    predicted_doublets_, z_scores_ 
    threshold_, detected_doublet_rate_,
    detectable_doublet_fraction_, overall_doublet_rate_,
    doublet_parents_, doublet_neighbor_parents_ 
    """
    import gc

    total_counts_obs = count_matrix.sum(1).A.squeeze()

    if verbose: logging.info('Simulating doublets...')
    (count_matrix_sim, total_counts_sim, _) = simulate_doublets(
        count_matrix, total_counts_obs, sim_doublet_ratio,
        synthetic_doublet_umi_subsampling, random_state
    )

    if verbose: logging.info('Spectral embedding ...')
    n = count_matrix.shape[0]
    merged_matrix = ss.vstack([count_matrix, count_matrix_sim])
    del count_matrix_sim
    gc.collect()
    _, evecs = spectral(
        AnnData(X=merged_matrix),
        features=None,
        n_comps=n_comps,
        inplace=False,
    )
    manifold = np.asanyarray(evecs)
    manifold_obs = manifold[0:n, ]
    manifold_sim = manifold[n:, ]

    if verbose: logging.info('Calculating doublet scores...')
    doublet_scores_obs, doublet_scores_sim = calculate_doublet_scores(
        manifold_obs, manifold_sim, k = n_neighbors,
        exp_doub_rate = expected_doublet_rate,
        use_approx_neighbors = use_approx_neighbors,
        random_state = random_state,
    )

    return (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim)


def simulate_doublets(
    count_matrix: ss.spmatrix,
    total_counts: np.ndarray,
    sim_doublet_ratio: int = 2,
    synthetic_doublet_umi_subsampling: float = 1.0,
    random_state: int = 0,
) -> tuple[ss.spmatrix, np.ndarray, np.ndarray]:
    """
    Simulate doublets by adding the counts of random cell pairs.

    Parameters
    ----------
    count_matrix
    total_counts
        Total insertion counts in each cell
    sim_doublet_ratio
        Number of doublets to simulate relative to the number of cells.
    synthetic_doublet_umi_subsampling
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified rate.

    Returns
    -------
    count_matrix_sim

    """
    n_obs = count_matrix.shape[0]
    n_sim = int(n_obs * sim_doublet_ratio)

    np.random.seed(random_state)
    pair_ix = np.random.randint(0, n_obs, size=(n_sim, 2))

    count_matrix_sim = count_matrix[pair_ix[:,0],:] + count_matrix[pair_ix[:,1],:]
    total_counts_sim = total_counts[pair_ix[:,0]] + total_counts[pair_ix[:,1]]

    if synthetic_doublet_umi_subsampling < 1:
        pass
        #count_matrix_sim, total_counts_sim = subsample_counts(
        #    count_matrix_sim, synthetic_doublet_umi_subsampling, total_counts_sim,
        #    random_seed=random_state
        #)
    return (count_matrix_sim, total_counts_sim, pair_ix)

def calculate_doublet_scores(
    manifold_obs: np.ndarray,
    manifold_sim: np.ndarray,
    k: int = 40,
    exp_doub_rate: float = 0.1,
    stdev_doub_rate: float = 0.03,
    use_approx_neighbors=False,
    random_state: int = 0,
) -> None:
    """
    Parameters
    ----------
    manifold_obs
        Manifold of observations
    manifold_sim
        Manifold of simulated doublets
    k
        Number of nearest neighbors
    exp_doub_rate
    stdev_doub_rate
    random_state
    """
    from sklearn.neighbors import NearestNeighbors

    n_obs = manifold_obs.shape[0]
    n_sim = manifold_sim.shape[0]

    manifold = np.vstack((manifold_obs, manifold_sim))
    doub_labels = np.concatenate(
        (np.zeros(n_obs, dtype=int), np.ones(n_sim, dtype=int))
    )

    # Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
    k_adj = int(round(k * (1 + n_sim / float(n_obs))))
    
    # Find k_adj nearest neighbors
    if use_approx_neighbors:
        knn = internal.approximate_nearest_neighbour_graph(
            manifold.astype(np.float32), k_adj)
    else:
        knn = internal.nearest_neighbour_graph(manifold, k_adj)
    indices = knn.indices
    indptr = knn.indptr
    neighbors = np.vstack(
        [indices[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]
    )
    
    # Calculate doublet score based on ratio of simulated cell neighbors vs. observed cell neighbors
    doub_neigh_mask = doub_labels[neighbors] == 1
    n_sim_neigh = doub_neigh_mask.sum(1)
    n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh
    
    rho = exp_doub_rate
    r = n_sim / float(n_obs)
    nd = n_sim_neigh.astype(float)
    ns = n_obs_neigh.astype(float)
    N = float(k_adj)
    
    # Bayesian
    q=(nd+1)/(N+2)
    Ld = q*rho/r/(1-rho-q*(1-rho-rho/r))

    se_q = np.sqrt(q*(1-q)/(N+3))
    se_rho = stdev_doub_rate

    se_Ld = q*rho/r / (1-rho-q*(1-rho-rho/r))**2 * np.sqrt((se_q/q*(1-rho))**2 + (se_rho/rho*(1-q))**2)

    doublet_scores_obs = Ld[doub_labels == 0]
    doublet_scores_sim = Ld[doub_labels == 1]
    doublet_errors_obs = se_Ld[doub_labels==0]
    doublet_errors_sim = se_Ld[doub_labels==1]

    return (doublet_scores_obs, doublet_scores_sim)

def get_doublet_probability(
    doublet_scores_sim: np.ndarray,
    doublet_scores: np.ndarray,
    random_state: int = 0,
    verbose: bool = False,
):
    from sklearn.mixture import BayesianGaussianMixture

    X = doublet_scores_sim.reshape((-1, 1))
    gmm = BayesianGaussianMixture(
        n_components=2, n_init=10, max_iter=1000, random_state=random_state
    ).fit(X)

    if verbose:
        logging.info("GMM means: {}".format(gmm.means_))

    i = np.argmax(gmm.means_)
    return gmm.predict_proba(doublet_scores.reshape((-1, 1)))[:,i]
