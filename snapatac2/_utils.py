import numpy as np
import logging

from anndata import AnnData
import snapatac2._snapatac2 as internal

def is_anndata(data) -> bool:
    """Check whether an object is an AnnData-like object supported by SnapATAC2.

    Use this helper before dispatching code paths that require either in-memory
    AnnData, backed SnapATAC2 AnnData, or AnnDataSet objects.
    """
    return isinstance(data, AnnData) or isinstance(data, internal.AnnData) or isinstance(data, internal.AnnDataSet)

def anndata_par(adatas, func, n_jobs=4):
    """Apply a function to multiple AnnData objects with safe parallel dispatch.

    Use this helper when a preprocessing or metrics function receives a list of
    AnnData objects. In-memory AnnData objects are processed serially; backed
    SnapATAC2 AnnData objects are closed, processed in worker processes, and
    reopened after completion.
    """
    return anndata_ipar(list(enumerate(adatas)), lambda x: func(x[1]), n_jobs=n_jobs)

def anndata_ipar(inputs, func, n_jobs=4):
    """Apply an indexed function to multiple AnnData objects.

    Use this helper when the worker function needs both the object index and the
    AnnData object. It preserves input order and reopens backed objects after
    multiprocessing finishes.
    """
    from tqdm import tqdm
    
    exist_in_memory_adata = False
    for _, adata in inputs:
        if isinstance(adata, AnnData):
            exist_in_memory_adata = True
            break
    if exist_in_memory_adata:
        logging.warn(("Input contains in-memory AnnData objects. "
                      "Multiprocessing will not be used. "
                      "To enable multiprocessing, use backed AnnData objects"))
        return [func((i, adata)) for i, adata in tqdm(inputs)]
    else:
        from multiprocess import get_context

        def _func(x):
            adata = internal.read(x[1], backend=x[2])
            result = func((x[0], adata))
            adata.close() 
            return result

        # Close the AnnData objects and return the filenames
        files = []
        for i, adata in inputs:
            files.append((i, adata.filename, adata.backend))
            adata.close()

        with get_context("spawn").Pool(n_jobs) as p:
            result = list(tqdm(p.imap(_func, files), total=len(files)))
        
        # Reopen the files if they were closed
        for _, adata in inputs:
            adata.open(mode='r+')
        
        return result

def get_file_format(suffix):
    """Infer genomic track format and compression from a file suffix.

    Use this helper before exporting coverage or fragments to determine whether
    a suffix denotes bigWig, bedGraph, gzip compression, or zstandard
    compression.
    """
    suffix = suffix.lower()
    _suffix = suffix

    if suffix.endswith(".gz"):
        compression = "gzip"
        _suffix = suffix[:-3]
    elif suffix.endswith(".zst"):
        compression = "zstandard"
        _suffix = suffix[:-4]
    else:
        compression = None
    
    if suffix.endswith(".bw") or suffix.endswith(".bigwig"):
        format = "bigwig"
    elif _suffix.endswith(".bedgraph") or _suffix.endswith(".bg") or _suffix.endswith(".bdg"):
        format = "bedgraph"
    else:
        format = None
    
    return format, compression

def get_igraph_from_adjacency(adj):
    """Get igraph graph from adjacency matrix."""
    try:
        import igraph as ig
    except ImportError:
        raise ImportError(
            "tl.leiden requires python-igraph, an optional dependency of SnapATAC2. "
            "Install it with:  pip install 'snapatac2[clustering]'  (or: pip install igraph). "
            "It is optional because python-igraph is GPL-licensed while SnapATAC2 is MIT, so "
            "users who never call tl.leiden, tl.leiden_sweep or tl.pagerank do not have to "
            "take on a copyleft dependency."
        ) from None
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
    """Find the first saturation point in a decreasing sequence.

    Use this helper to locate the elbow where incremental gains fall below a
    fraction of accumulated gain.
    """
    accum_gap = 0
    for i in range(1, len(x)):
        gap = x[i-1] - x[i]
        accum_gap = accum_gap + gap
        if gap < saturation * accum_gap:
            return i
    return None

def fetch_seq(fasta, region):
    """Fetch a genomic sequence from a FASTA object for one region string.

    Use this helper with region strings formatted as `chrom:start-end`. The
    function raises an error if the retrieved sequence length does not match the
    requested interval length.
    """
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
