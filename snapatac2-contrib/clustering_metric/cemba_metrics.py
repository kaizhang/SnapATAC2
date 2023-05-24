# envs from CEMBA consensus analysis.
import os
import scipy
import random
from typing import Tuple, Union
from scipy.sparse import csr_matrix
import numpy as np
import itertools
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import leidenalg as la
import igraph as ig
import matplotlib.pyplot as plt

def cal_connectivity(P):
    """calculate connectivity matrix"""
    # print("=== calculate connectivity matrix ===")
    # connectivity_mat = lil_matrix((len(P), len(P)))
    # connectivity_mat = csr_matrix((len(P), len(P)))
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
    Y = 1 - C
    Z = linkage(squareform(Y), method="average")
    ivl = leaves_list(Z)
    ivl = ivl[::-1]
    return C[:, ivl][ivl, :]

def plot_CDF(prefix, C, u1, u2, num_bins=100):
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
        nsample: Union[int, None] = None
)->Tuple[float, float]:
    """
    Parameters
    ----------
    partitions
        numpy.ndarray, shape n_cell x n_times
    Returns
    -------
    float
         cemba metric for clustering
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
    consensus = np.zeros((n_cell, n_cell), dtype = np.float16)
    for i in range(n_times):
        print(f"{i} / {n_times} for consensus matrix")
        conn = cal_connectivity(p[:,i])
        consensus += conn
    consensus /= n_times
    disp: float = cal_dispersion(consensus)
    pac: float = cal_PAC(
        consensus, u1 = 0.05, u2 = 0.95)
    # stabs per cell
    # stabs: np.ndarray = np.apply_along_axis(cal_stab, 0, consensus)
    return (disp, pac)

if __name__ == '__main__':
    # * test
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
    g = ig.Graph(vcount, edgelist)

    # generate consensus matrix
    dims = knn.shape[0]
    consensus = csr_matrix((dims, dims))
    N = 2
    def do_leiden(seed):
        partition = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            resolution_parameter = reso,
            seed = seed)
        part_membership = partition.membership
        return(part_membership)

    partitions = list(map(do_leiden, range(N)))
    partitions = np.transpose(np.array(partitions))
    cemba_metric(partitions, nsample = N)

