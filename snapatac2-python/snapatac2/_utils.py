import chunk
import numpy as np

def binarize_inplace(X):
    """Binarize sparse matrix in-place"""
    X.data = np.ones(X.indices.shape, dtype=np.float64)

def get_binarized_matrix(X):
    """Return a copy of binarize sparse matrix"""
    X_ = X.copy()
    binarize_inplace(X_)
    return X_

def get_igraph_from_adjacency(adj):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = list(zip(list(sources), list(targets)))
    weights = np.ravel(adj[(sources, targets)])
    gr = ig.Graph(n=vcount, edges=edgelist, edge_attrs={"weight": weights})
    return gr

def chunks(mat, chunk_size: int):
    """
    Return chunks of the input matrix
    """
    n = mat.shape[0]
    for i in range(0, n, chunk_size):
        j = max(i + chunk_size, n)
        yield mat[i:j, :]