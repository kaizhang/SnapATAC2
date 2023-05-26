"""
Add CEMBA metrics for the selection of resolution of leiden algorithm.

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

import os
import scipy
import random
from typing import Tuple, Union
from scipy.sparse import csr_matrix
import numpy as np
import itertools

__all__ = ["cemba_metric", "cal_connectivity", "reorder", "plot_CDF"]


def cal_connectivity(P):
    """calculate connectivity matrix"""
    connectivity_mat = np.zeros((len(P), len(P)), dtype = bool)
    classN = max(P)
    ## TODO: accelerate this
    for cls in range(int(classN + 1)):
        xidx = [i for i, x in enumerate(P) if x == cls]
        iterables = [xidx, xidx]
        for t in itertools.product(*iterables):
            connectivity_mat[t[0], t[1]] = True
    """connectivity_mat = csr_matrix(connectivity_mat)"""
    return connectivity_mat


def reorder(C):
    """
    Reorder consensus matrix.

    :param C: Consensus matrix.
    :type C: `numpy.ndarray`
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list
    Y = 1 - C
    Z = linkage(squareform(Y), method="average")
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]

def plot_CDF(prefix, C, u1, u2, num_bins=100):
    import matplotlib.pyplot as plt
    counts, bin_edges = np.histogram(C, bins=num_bins, density=True)
    cdf = np.cumsum(counts)
    fig = plt.figure(dpi=100)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel("Consensus index value")
    plt.ylabel("CDF")
    plt.axvline(x=u1, color="grey", linestyle="--")
    plt.axvline(x=u2, color="grey", linestyle="--")
    fileN = [prefix, "cdf", "png"]
    fileN = ".".join(fileN)
    fig.savefig(fileN)
    outBinEdges = ".".join([prefix, "cdf.txt"])
    with open(outBinEdges, "w") as fo:
        fo.write("\t".join(str(i) for i in cdf) + "\n")


def cal_dispersion(C):
    """calculate dispersion coefficient"""
    n = C.shape[1]
    corr_disp = np.sum(
        4 * np.square(C - 0.5), dtype = np.float64) / (np.square(n))
    return corr_disp


def cal_PAC(C, u1, u2):
    """calculate PAC (proportion of ambiguous clustering)"""
    n = C.shape[0] ** 2
    PAC = ((C<u2).sum() - (C<u1).sum()) / n
    return PAC

def cal_stab(x) -> np.ndarray:
    """calculate stability for every cell"""
    s = np.sum(abs(x - 0.5)) / (0.5 * x.shape[0])
    return s


def cemba_metric(
        partitions: np.ndarray,
        nsample: Union[int, None] = None,
        u1: float = 0.05,
        u2: float = 0.95
)->Tuple[float, float]:
    """
    Parameters
    ----------
    partitions:
        numpy.ndarray, dtype as np.unit, shape n_cell x n_times
    nsample:
        int or None, used for downsampling cells, default is None.
    u1:
        float, lower-bound for PAC, default is 0.05.
    u2:
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
    p = partitions.astype(np.uint)
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
    # stabs per cell
    # stabs: np.ndarray = np.apply_along_axis(cal_stab, 0, consensus)
    return (disp, pac)

if __name__ == '__main__':
    # * test
    # a temp data
    knn = scipy.io.mmread(os.path.join(
        "test_data", "GLUTL3", "GLUT_13",
        "GLUT_13.k-50.km-RANN.mmtx"
    ))
    knn = knn.tocsr()
    reso = 0.8
    dimN = knn.shape[0]
    vcount = max(knn.shape)
    sources, targets = knn.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))

    import igraph as ig
    g = ig.Graph(vcount, edgelist)

    # generate consensus matrix
    dims = knn.shape[0]
    consensus = csr_matrix((dims, dims))
    N = 2
    
    import leidenalg as la
    def do_leiden(seed):
        partition = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            resolution_parameter = reso,
            seed = seed)
        part_membership = partition.membership
        return(part_membership)

    partitions = list(map(do_leiden, range(N)))
    partitions = np.transpose(np.array(partitions))
    cemba_metric(partitions, nsample = 100, u1 = 0.05, u2 = 0.95)


