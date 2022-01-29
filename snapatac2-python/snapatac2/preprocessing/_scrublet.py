""" Implementation of the scrublet algorithm for single-cell ATAC-seq data
"""

import numpy as np
import anndata as ad
from typing import Optional, Union, Type

from .._utils import get_binarized_matrix
from snapatac2.tools._spectral import Spectral

def scrublet(
    adata: ad.AnnData,
    sim_doublet_ratio: float = 3.0,
    features: Optional[np.ndarray] = None,
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
    import scrublet as scr

    if features is None:
        counts_matrix = adata.X[...]
    else:
        counts_matrix = adata.X[:, features]
    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate = 0.06,
        sim_doublet_ratio=sim_doublet_ratio, n_neighbors=n_neighbors,
        random_state=random_state)

    doublet_scores, _ = scrub_doublets_core(scrub, use_approx_neighbors=use_approx_neighbors)
    adata.obs["doublet_score"] = doublet_scores
    adata.uns["scrublet_cell_embedding"] = scrub.manifold_obs_
    adata.uns["scrublet_sim_doublet_embedding"] = scrub.manifold_sim_
    adata.uns["scrublet_sim_doublet_score"] = scrub.doublet_scores_sim_

def scrub_doublets_core(
    self,
    synthetic_doublet_umi_subsampling=1.0,
    use_approx_neighbors=True,
    distance_metric='euclidean',
    get_doublet_neighbor_parents=False,
    n_comps=30,
    verbose=True
) -> None:
    """
    Modified scrublet pipeline for single-cell ATAC-seq data.

    Automatically sets a threshold for calling doublets, but it's best to check 
    this by running plot_histogram() afterwards and adjusting threshold 
    with call_doublets(threshold=new_threshold) if necessary.

    Arguments
    ---------
    synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified
        rate.
    use_approx_neighbors : bool, optional (default: True)
        Use approximate nearest neighbor method (annoy) for the KNN 
        classifier.
    distance_metric : str, optional (default: 'euclidean')
        Distance metric used when finding nearest neighbors. For list of
        valid values, see the documentation for annoy (if `use_approx_neighbors`
        is True) or sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors`
        is False).
    get_doublet_neighbor_parents : bool, optional (default: False)
        If True, return the parent transcriptomes that generated the 
        doublet neighbors of each observed transcriptome. This information can 
        be used to infer the cell states that generated a given 
        doublet state.
    n_comps
        Number of principal components used to embed the transcriptomes prior
        to k-nearest-neighbor graph construction.
    verbose : bool, optional (default: True)
        If True, print progress updates.

    Returns
    -------
    doublet_scores_obs_, doublet_errors_obs_,
    doublet_scores_sim_, doublet_errors_sim_,
    predicted_doublets_, z_scores_ 
    threshold_, detected_doublet_rate_,
    detectable_doublet_fraction_, overall_doublet_rate_,
    doublet_parents_, doublet_neighbor_parents_ 
    """
    self._E_sim = None
    self._E_obs_norm = None
    self._E_sim_norm = None

    print('Simulating doublets...')
    self.simulate_doublets(sim_doublet_ratio=self.sim_doublet_ratio,
        synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling)

    self._E_obs_norm = get_binarized_matrix(self._E_obs)
    self._E_sim_norm = get_binarized_matrix(self._E_sim)

    print('Spectral embedding ...')
    pipeline_spectral(self, n_comps=n_comps)

    print('Calculating doublet scores...')
    self.calculate_doublet_scores(
        use_approx_neighbors=use_approx_neighbors,
        distance_metric=distance_metric,
        get_doublet_neighbor_parents=get_doublet_neighbor_parents
        )
    self.call_doublets(verbose=verbose)

    return self.doublet_scores_obs_, self.predicted_doublets_

def pipeline_spectral(self, n_comps=30, random_state=0):
    model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=1).fit(self._E_obs_norm)
    self.set_manifold(np.asarray(model.transform()[:, 1:]),
        np.asarray(model.transform(self._E_sim_norm)[:, 1:])
    )
