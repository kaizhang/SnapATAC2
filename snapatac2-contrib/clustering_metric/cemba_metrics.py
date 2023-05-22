# envs from CEMBA consensus analysis.
import os
import scipy
import leidenalg as la
import igraph as ig
from scipy.io import mmread
from scipy import sparse
from time import perf_counter as pc
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import save_npz
import numpy as np
import itertools
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, imshow, set_cmap
from multiprocessing import Pool
plt.switch_backend("agg")
import fastcluster as fc

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
    print("=== calculate dispersion coefficient ===")
    start_t = pc()
    n = C.shape[1]
    corr_disp = np.sum(4 * np.square(np.concatenate(C - 1 / 2))) / (np.square(n))
    end_t = pc()
    print("Used (secs): ", end_t - start_t)
    return corr_disp


def cal_PAC(C, u1, u2):
    """calculate PAC (proportion of ambiguous clustering)"""
    print("=== calculate PAC (proportion of ambiguous clustering) ===")
    start_t = pc()
    totalN = C.shape[0] * C.shape[0]
    u1_fraction = (C.ravel() < u1).sum() / totalN
    u2_fraction = (C.ravel() < u2).sum() / totalN
    PAC = u2_fraction - u1_fraction
    end_t = pc()
    print("Used (secs): ", end_t - start_t)
    return PAC

def cal_stab(x):
    """calculate stability for every cell"""
    s = np.sum(abs(x - 0.5)) / (0.5 * x.shape[0])
    return s



def run(knn :csr_matrix,
        reso: float = 1.0,
        left_cutoff: float = 0.05,
        right_cutoff:float = 0.95,
        niter: int = 10,
        nsample: int = 50000) -> float:
    """
    Parameters
    ----------
    knn
        k-nearst neighbor graph
    Returns
    -------
    
    """
    return 1.0


# * test
knn = scipy.io.mmread(os.path.join(
        "test_data", "GLUTL3", "GLUT_13",
        "GLUT_13.k-50.km-RANN.mmtx"
))
knn = knn.tocsr()

# * run in 02.consensusAnalysis.py
def run():
    """Run and count to peak"""
    start_time = pc()
    """ init input files """
    inf = args.input
    reso = args.resolution
    N = args.N
    sampleN = args.sample
    u1 = args.u1
    u2 = args.u2
    outf = args.output

    knn = mmread(inf)
    knn = knn.tocsr()
    dimN = knn.shape[0]

    vcount = max(knn.shape)
    sources, targets = knn.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(vcount, edgelist)

    # generate consensus matrix
    dims = knn.shape[0]
    if dims > sampleN:
        dims = sampleN
        import random

        random.seed(2022)
        idxy = random.sample(range(dimN), sampleN)
        idxy = sorted(idxy)
    consensus = csr_matrix((dims, dims))

    print("=== calculate connectivity matrix ===")

    for seed in range(N):
        start_t =pc()
        partition = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter = reso, seed = seed)
        part_membership = partition.membership
        part_membership = np.array(part_membership)
        if len(part_membership) > sampleN:
            part_membership = part_membership[idxy] # downsample (10,000 observations)
        outs = cal_connectivity(part_membership)
        consensus += outs
        end_t = pc()
        print('seed ', seed, ' used (secs): ', end_t - start_t)
    consensus /= N

    # save consensus matrix
    consensus_sp = sparse.csr_matrix(consensus)
    outfname = ".".join([outf, "consensus", "npz"])
    save_npz(outfname, consensus_sp)

    # plotting
    order_consensus = reorder(consensus)
    plotC(outf, order_consensus)
    plot_CDF(outf, consensus, u1, u2)

    # cal measurement
    #    o_cophcor = cal_cophenetic(consensus)
    o_disp = cal_dispersion(consensus)
    o_PAC = cal_PAC(consensus, u1, u2)
    print("=== calculate stability for every cell ===")
    o_stab = np.apply_along_axis(cal_stab, 0, consensus)

    # write stat
    out_list = [outf, reso, o_disp, o_PAC]
    outstat = ".".join([outf, "stat.txt"])
    with open(outstat, "w") as fo:
        fo.write("\t".join(str(i) for i in out_list) + "\n")

    outStab = ".".join([outf, "stab.txt"])
    with open(outStab, "w") as fo:
        fo.write("\t".join(str(i) for i in o_stab) + "\n")

    end_time = pc()
    print("Total used (secs): ", end_time - start_time)
