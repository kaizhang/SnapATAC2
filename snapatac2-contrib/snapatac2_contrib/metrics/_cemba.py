from __future__ import annotations

import random
from typing import Tuple, Union
import numpy as np
import itertools

from snapatac2 import AnnData
import snapatac2 as snap

__all__ = ["cemba"]

def cemba(
    adata: AnnData | np.ndarray,
    resolution: float,
    objective_function='modularity',
    n_repeat: int = 5,
    random_state: int = 0,
) -> Tuple[float, float]:
    """CEMBA metrics for the selection of resolution of leiden algorithm.

    Detailes in "Cell clustering" of "Methods" in
    https://www.nature.com/articles/s41586-021-03604-1

    The metrics are all based on the so-called connectivity matrix M.
    - it's shape is (n_cell, n_cell)
    - M_ij \in (0, 1) is the fraction of cell i and cell j are in the same cluster
    during multiple round of repeats of leiden algorithms.

    In the original implementation, we use different random seeds for leiden
    algorithm as the multiple rounds of repeats. But we can also apply some random
    modifications of k-nearest neighbor (knn) graph, and then use leiden algorithm on the
    modified knn graph.

    Ideally, two cells are either in the same clusters or not. So most of M should be around
    zeros and ones. But if clustering is not that good, then M_{ij} may be around 0.5.

    Here we include three metrics:
    1. PAC, proportion of ambiguous clustering
    - it's \in (0,1)
    - the lower, the better
    - (\sum (M_{ij} < 0.95) - \sum (M_{ij} < 0.05) ) / n_cell ^2
    2. Dispersion coefficient (disp)
    - it's \in (0,1)
    - the higher, the better
    - \sum 4 * (M_{ij} - 0.5)^2 / n_cell ^2.

    3. Cumulative distribution function (cdf) curve 
    - draw the cdf based on the cdf of M.
    - since M_{ij} should be around 0 or 1 ideally. So the cdf curve would
        be have a relative flat region in the middle,
        and sharp increase around 0.0 and 1.0.

    During analysis, we use PAC and disp to choose the resolution, which
    will be the values of cemba_metrics.
    """
    np.random.seed(random_state)
    # FIXME: random numbers may not be unique.
    random_states = np.random.randint(0, 1000000, size=n_repeat)
    membership = []
    for r in random_states:
        membership.append(snap.tl.leiden(
            adata, objective_function=objective_function, resolution=resolution,
            random_state=r, inplace=False
        ))
    partitions = np.array(membership).T
    return compute_metrics(partitions, nsample=None, u1=0.05, u2=0.95)

def compute_metrics(
    partitions: np.ndarray,
    nsample: Union[int, None] = None,
    u1: float = 0.05,
    u2: float = 0.95
) -> Tuple[float, float]:
    """
    Parameters
    ----------
    partitions
        numpy.ndarray, dtype as np.unit, shape n_cell x n_times
    nsample
        int or None, used for downsampling cells, default is None.
    u1
        float, lower-bound for PAC, default is 0.05.
    u2
        float, upper-bound for PAC, default is 0.95.
    
    Returns
    -------
    Tuple[float, float]:
        (disp, PAC) in order.
    """
    
    # * check partitions
    ndim:int = partitions.ndim
    if ndim != 2:
        raise RuntimeError(
            f"partitions should have 2 instead of {ndim} dims.")
    # * to unsigned int64
    p = partitions.astype(np.int32)
    n_cell, n_times = p.shape
    # * downsample partitions if needed.
    if nsample and n_cell > nsample:
        print(f"Down sampling {n_cell} to {nsample}")
        random.seed(0)
        index = random.sample(range(n_cell), nsample)
        p = p[index, :]
        n_cell, n_times = p.shape
    consensus = np.zeros((n_cell, n_cell), dtype = np.float16)
    for i in range(n_times):
        print(f"{i+1} / {n_times} for consensus matrix")
        conn = cal_connectivity(p[:,i])
        consensus += conn
    consensus /= n_times
    disp: float = cal_dispersion(consensus)
    pac: float = cal_PAC(
        consensus, u1 = u1, u2 = u2)
    return (disp, pac)

def cal_connectivity(partition: list[int]) -> np.ndarray:
    """calculate connectivity matrix"""
    connectivity_mat = np.zeros((len(partition), len(partition)), dtype = bool)
    classN = max(partition)
    ## TODO: accelerate this
    for cls in range(int(classN + 1)):
        xidx = [i for i, x in enumerate(partition) if x == cls]
        iterables = [xidx, xidx]
        for t in itertools.product(*iterables):
            connectivity_mat[t[0], t[1]] = True
    """connectivity_mat = csr_matrix(connectivity_mat)"""
    return connectivity_mat

def cal_dispersion(consensus) -> float:
    """calculate dispersion coefficient

    Parameters
    ----------
    consensus
        Consensus matrix, shape (n_sample, n_sample). Each entry in the matrix is
        the fraction of times that two cells are clustered together.
    """
    n = consensus.shape[1]
    corr_disp = np.sum(
        4 * np.square(consensus - 0.5), dtype = np.float64) / (np.square(n))
    return corr_disp

def cal_PAC(consensus, u1, u2) -> float:
    """calculate PAC (proportion of ambiguous clustering)

    Parameters
    ----------
    consensus
        Consensus matrix, shape (n_sample, n_sample). Each entry in the matrix is
        the fraction of times that two cells are clustered together.
    """
    n = consensus.shape[0] ** 2
    PAC = ((consensus < u2).sum() - (consensus < u1).sum()) / n
    return PAC