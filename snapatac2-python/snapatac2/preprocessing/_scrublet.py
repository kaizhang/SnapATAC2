""" Implementation of the scrublet algorithm for single-cell ATAC-seq data
"""
import numpy as np
import scipy.sparse as ss
import anndata as ad
from typing import Optional, Union, Type, Tuple
from sklearn.neighbors import NearestNeighbors

from .._utils import get_binarized_matrix
from snapatac2.tools._spectral import Spectral

def scrublet(
    adata: ad.AnnData,
    features: Optional[np.ndarray] = None,
    sim_doublet_ratio: float = 3.0,
    expected_doublet_rate: float = 0.1,
    n_neighbors: Optional[int] = None,
    use_approx_neighbors=True,
    random_state: int = 0,
) -> None:
    """
    Compute probability of being a doublet using the scrublet algorithm.

    Parameters
    ----------
    adata
        AnnData object
    sim_doublet_ratio
        Number of doublets to simulate relative to the number of observed cells.
    features
        Boolean index mask. True means that the feature is kept.
        False means the feature is removed.
    n_neighbors
        Number of neighbors used to construct the KNN graph of observed
        cells and simulated doublets. If `None`, this is 
        set to round(0.5 * sqrt(n_cells))
    use_approx_neighbors
        Whether to use 
    random_state
        Random state
    
    Returns
    -------
    """
    if features is None:
        count_matrix = adata.X[...]
    else:
        count_matrix = adata.X[:, features]

    if n_neighbors is None:
        n_neighbors = int(round(0.5 * np.sqrt(count_matrix.shape[0])))

    (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim) = scrub_doublets_core(
        count_matrix, n_neighbors, sim_doublet_ratio, expected_doublet_rate,
        random_state=random_state
        )

    adata.obs["doublet_score"] = doublet_scores_obs
    adata.uns["scrublet_cell_embedding"] = manifold_obs
    adata.uns["scrublet_sim_doublet_embedding"] = manifold_sim
    adata.uns["scrublet_sim_doublet_score"] = doublet_scores_sim

def scrub_doublets_core(
    count_matrix: ss.spmatrix,
    n_neighbors: int,
    sim_doublet_ratio: float,
    expected_doublet_rate: float,
    synthetic_doublet_umi_subsampling: float =1.0,
    n_comps: int = 30,
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

    print('Simulating doublets...')
    (count_matrix_sim, total_counts_sim, _) = simulate_doublets(
        count_matrix, total_counts_obs, sim_doublet_ratio,
        synthetic_doublet_umi_subsampling, random_state
    )

    E_obs_norm = get_binarized_matrix(count_matrix)
    E_sim_norm = get_binarized_matrix(count_matrix_sim)

    print('Spectral embedding ...')
    (manifold_obs, manifold_sim) = get_manifold(E_obs_norm, E_sim_norm, n_comps=n_comps)

    print('Calculating doublet scores...')
    (doublet_scores_obs, doublet_scores_sim) = calculate_doublet_scores(
        manifold_obs, manifold_sim, k = n_neighbors,
        exp_doub_rate = expected_doublet_rate,
        random_state = random_state,
    )

    return (doublet_scores_obs, doublet_scores_sim, manifold_obs, manifold_sim)

def get_manifold(E_obs_norm, E_sim_norm, n_comps=30, random_state=0):
    model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=1).fit(E_obs_norm)
    manifold_obs = np.asarray(model.transform()[:, 1:])
    manifold_sim = np.asarray(model.transform(E_sim_norm)[:, 1:])
    return (manifold_obs, manifold_sim)

def simulate_doublets(
    count_matrix: ss.spmatrix,
    total_counts: np.adarray,
    sim_doublet_ratio: int = 2,
    synthetic_doublet_umi_subsampling: float = 1.0,
    random_state: int = 0,
) -> Tuple[ss.spmatrix, np.ndarray, np.ndarray]:
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
    n_obs = manifold_obs.shape[0]
    n_sim = manifold_sim.shape[0]

    manifold = np.vstack((manifold_obs, manifold_sim))
    doub_labels = np.concatenate(
        (np.zeros(n_obs, dtype=int), np.ones(n_sim, dtype=int))
    )

    # Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
    k_adj = int(round(k * (1 + n_sim / float(n_obs))))
    
    # Find k_adj nearest neighbors
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