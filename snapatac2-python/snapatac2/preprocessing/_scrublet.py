""" Implementation of the scrublet algorithm for single-cell ATAC-seq data
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as ss
import logging

from .._utils import chunks
from snapatac2._snapatac2 import AnnData, approximate_nearest_neighbors
from snapatac2.tools.embedding._spectral import Spectral

def scrublet(
    adata: AnnData,
    features: str | np.ndarray | None = "selected",
    n_comps: int = 15,
    sim_doublet_ratio: float = 2.0,
    expected_doublet_rate: float = 0.1,
    n_neighbors: int | None = None,
    use_approx_neighbors=True,
    random_state: int = 0,
) -> None:
    """
    Compute probability of being a doublet using the scrublet algorithm.

    Parameters
    ----------
    adata
        AnnData object
    features
        Boolean index mask, where `True` means that the feature is kept, and
        `False` means the feature is removed.
    n_comps
        Number of PCs
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
    
    Returns
    -------
    None
        It updates adata with the following fields:
            - ``adata.obs["doublet_score"]``: scrublet doublet score
            - ``adata.uns["scrublet"]["sim_doublet_score"]``: doublet scores of simulated doublets 
    """
    #- adata.uns["scrublet_cell_embedding"]: embedding of cells
    #- adata.uns["scrublet_sim_doublet_embedding"]: embedding of simulated doublets

    if isinstance(features, str):
        if features in adata.var:
            features = adata.var[features]
        else:
            raise NameError("Please call `select_features` first or explicitly set `features = None`")

    if features is None:
        count_matrix = adata.X[...]
    else:
        count_matrix = adata.X[:, features]

    if min(count_matrix.shape) == 0: raise NameError("Matrix is empty")

    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(count_matrix.shape[0])))

    (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim) = scrub_doublets_core(
        count_matrix, n_neighbors, sim_doublet_ratio, expected_doublet_rate,
        n_comps=n_comps,
        use_approx_neighbors = use_approx_neighbors,
        random_state=random_state
        )
    adata.obs["doublet_score"] = doublet_scores_obs
    adata.uns["scrublet_sim_doublet_score"] = doublet_scores_sim

def call_doublets(
    adata: AnnData,
    threshold: str | float = "gmm",
    random_state: int = 0,
    inplace: bool = True,
) -> tuple[np.ndarray, float] | None:
    """
    Find doublet score threshold for calling doublets.
    `pp.scrublet` must be run first.

    Parameters
    ----------
    adata
        AnnData object
    threshold
        Mannually specify a threshold or use one of the default methods to
        automatically identify a threshold:

        - 'gmm': fit a 2-component gaussian mixture model.
        - 'scrublet': the method used in the scrublet paper (not implemented).
    random_state
        Random state
    inplace
        Whether update the AnnData object inplace
    
    Returns
    -------
    tuple[np.ndarray, float] | None:
        if ``inplace = True``, it updates adata with the following fields:
            - ``adata.obs["is_doublet"]``: boolean mask
            - ``adata.uns["scrublet"]["threshold"]``: saved threshold
    """

    if 'scrublet_sim_doublet_score' not in adata.uns:
        raise NameError("Please call `scrublet` first")

    doublet_scores_sim = adata.uns["scrublet_sim_doublet_score"]
    doublet_scores_obs = adata.obs["doublet_score"].to_numpy()

    if isinstance(threshold, float):
        thres = threshold
    elif threshold == "gmm":
        thres, _ = get_doublet_probability(
            doublet_scores_sim, doublet_scores_obs, random_state
        )
    elif threshold == "scrublet":
        from skimage.filters import threshold_minimum
        thres = threshold_minimum(doublet_scores_sim)
    else:
        raise NameError("Invalid value for the `threshold` argument")

    doublets = doublet_scores_obs >= thres
    if inplace:
        adata.uns["scrublet_threshold"] = thres
        adata.obs["is_doublet"] = doublets
    else:
        return (doublets, thres)

def scrub_doublets_core(
    count_matrix: ss.spmatrix,
    n_neighbors: int,
    sim_doublet_ratio: float,
    expected_doublet_rate: float,
    synthetic_doublet_umi_subsampling: float =1.0,
    n_comps: int = 30,
    use_approx_neighbors: bool = True,
    random_state: int = 0,
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
    total_counts_obs = count_matrix.sum(1).A.squeeze()

    logging.info('Simulating doublets...')
    (count_matrix_sim, total_counts_sim, _) = simulate_doublets(
        count_matrix, total_counts_obs, sim_doublet_ratio,
        synthetic_doublet_umi_subsampling, random_state
    )

    logging.info('Spectral embedding ...')
    (manifold_obs, manifold_sim) = get_manifold(count_matrix, count_matrix_sim, n_comps=n_comps)

    logging.info('Calculating doublet scores...')
    (doublet_scores_obs, doublet_scores_sim) = calculate_doublet_scores(
        manifold_obs, manifold_sim, k = n_neighbors,
        exp_doub_rate = expected_doublet_rate,
        use_approx_neighbors = use_approx_neighbors,
        random_state = random_state,
    )

    return (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim)

def get_manifold(obs_norm, sim_norm, n_comps=30, random_state=0):
    model = Spectral(n_dim=n_comps, distance="jaccard").fit(obs_norm, verbose=0)
    for c in chunks(obs_norm, 2000):
        model.extend(c)
    for c in chunks(sim_norm, 2000):
        model.extend(c)
    manifold = np.asanyarray(model.transform()[1])
    n = obs_norm.shape[0]
    manifold_obs = manifold[0:n, ]
    manifold_sim = manifold[n:, ]
    return (manifold_obs, manifold_sim)

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
    use_approx_neighbors=True,
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
        _, indices, indptr = approximate_nearest_neighbors(manifold.astype(np.float32), k_adj)
        neighbors = np.vstack(
            [indices[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]
        )
    else:
        neighbors = NearestNeighbors(
            n_neighbors=k_adj, metric="euclidean"
            ).fit(manifold).kneighbors(return_distance=False)
    
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
):
    from sklearn.mixture import BayesianGaussianMixture

    X = doublet_scores_sim.reshape((-1, 1))
    gmm = BayesianGaussianMixture(
        n_components=2, n_init=10, max_iter=1000, random_state=random_state
    ).fit(X)
    i = np.argmax(gmm.means_)

    probs_sim = gmm.predict_proba(X)[:,i]
    vals = X[probs_sim > 0.5]
    threshold = X.max() if vals.size == 0 else vals.min()

    return (threshold, gmm.predict_proba(doublet_scores.reshape((-1, 1)))[:,i])
