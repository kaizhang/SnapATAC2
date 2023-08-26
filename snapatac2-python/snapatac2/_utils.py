import numpy as np
import anndata as ad
import logging

from snapatac2._snapatac2 import AnnData, AnnDataSet, read

def is_anndata(data) -> bool:
    return isinstance(data, ad.AnnData) or isinstance(data, AnnData) or isinstance(data, AnnDataSet)

def anndata_par(adatas, func, n_jobs=4):
    exist_in_memory_adata = False
    for adata in adatas:
        if isinstance(adata, ad.AnnData):
            exist_in_memory_adata = True
            break
    if exist_in_memory_adata:
        logging.warn(("Input contains in-memory AnnData objects. "
                      "Multiprocessing will not be used. "
                      "To enable multiprocessing, use backed AnnData objects"))
        return [func(adata) for adata in adatas]
    else:
        from multiprocess import get_context

        def _func(fl):
            adata = read(fl)
            result = func(adata)
            adata.close() 
            return result

        files = []
        for adata in adatas:
            files.append(adata.filename)
            adata.close()

        with get_context("spawn").Pool(n_jobs) as p:
            result = p.map(_func, files)
        
        # Reopen the files if they were closed
        for data in adatas:
            data.open()
        
        return result

def get_igraph_from_adjacency(adj):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = list(zip(list(sources), list(targets)))
    weights = np.ravel(adj[(sources, targets)])
    gr = ig.Graph(n=vcount, edges=edgelist, directed=False, edge_attrs={"weight": weights})
    return gr

def chunks(mat, chunk_size: int):
    """
    Return chunks of the input matrix
    """
    n = mat.shape[0]
    for i in range(0, n, chunk_size):
        j = max(i + chunk_size, n)
        yield mat[i:j, :]

def find_elbow(x, saturation=0.01):
    accum_gap = 0
    for i in range(1, len(x)):
        gap = x[i-1] - x[i]
        accum_gap = accum_gap + gap
        if gap < saturation * accum_gap:
            return i
    return None

def fetch_seq(fasta, region):
    chr, x = region.split(':')
    start, end = x.split('-')
    start = int(start)
    end = int(end)
    seq = fasta[chr][start:end].seq
    l1 = len(seq)
    l2 = end - start
    if l1 != l2:
        raise NameError(
            "sequence fetch error: expected length: {}, but got {}.".format(l2, l1)
        )
    else:
        return seq

def pcorr(A, B):
    """Compute pairwsie correlation between two matrices.

    A
        n_sample x n_feature
    B
        n_sample x n_feature
    """
    N = B.shape[0]

    sA = A.sum(0)
    sB = B.sum(0)

    p1 = N * np.einsum('ij,ik->kj', A, B)
    p2 = sA * sB[:,None]
    p3 = N * ((B**2).sum(0)) - (sB**2)
    p4 = N * ((A**2).sum(0)) - (sA**2)

    return (p1 - p2) / np.sqrt(p4 * p3[:,None])