from typing import Union, Sequence
import numpy as np
import collections.abc as cabc
import scipy.sparse as ss
import anndata as ad

def binarize_inplace(X):
    """Binarize sparse matrix in-place"""
    X.data = np.ones(X.indices.shape, dtype=np.float64)

def get_binarized_matrix(X):
    """Return a copy of binarize sparse matrix"""
    X_ = X.copy()
    binarize_inplace(X_)
    return X_

def read_as_binarized(adata: ad.AnnData) -> ss.spmatrix:
    grp = adata.file["X"]
    mtx = ss.csr_matrix(adata.shape, dtype=np.float64)
    mtx.indices = grp["indices"][...]
    mtx.indptr = grp["indptr"][...]
    mtx.data = np.ones(mtx.indices.shape, dtype=np.float64)
    return mtx

def get_igraph_from_adjacency(adj):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = list(zip(list(sources), list(targets)))
    weights = np.ravel(adj[(sources, targets)])
    gr = ig.Graph(n=vcount, edges=edgelist, edge_attrs={"weight": weights})
    return gr

def binarized_chunk_X(
    adata: ad.AnnData,
    select: Union[int, Sequence[int], np.ndarray] = 1000,
    replace: bool = False,
) -> ss.spmatrix:
    """
    Return a chunk of the data matrix :attr:`X` with random or specified indices.

    Parameters
    ----------
    select
        Depending on the type:
        :class:`int`
            A random chunk with `select` rows will be returned.
        :term:`sequence` (e.g. a list, tuple or numpy array) of :class:`int`
            A chunk with these indices will be returned.
    replace
        If `select` is an integer then `True` means random sampling of
        indices with replacement, `False` without replacement.
    """
    if isinstance(select, int):
        select = select if select < adata.n_obs else adata.n_obs
        choice = np.random.choice(adata.n_obs, select, replace)
    elif isinstance(select, (np.ndarray, cabc.Sequence)):
        choice = np.asarray(select)
    else:
        raise ValueError("select should be int or array")

    reverse = None
    if adata.isbacked:
        # h5py can only slice with a sorted list of unique index values
        # so random batch with indices [2, 2, 5, 3, 8, 10, 8] will fail
        # this fixes the problem
        indices, reverse = np.unique(choice, return_inverse=True)
        selection = adata.X[indices.tolist()]
    else:
        selection = adata.X[choice]

    binarize_inplace(selection)
    return selection if reverse is None else selection[reverse]

def inplace_init_view_as_actual(data):
    """
    Replace view of backed AnnData with actual data
    """
    if data.isbacked and data.is_view:
        filename = str(data.filename)
        data.write()
        data.file.close()
        new_data = ad.read(filename, backed="r+")
        new_data.file.close()
        data._init_as_actual(
            obs=new_data.obs,
            var=new_data.var,
            uns=new_data.uns,
            obsm=new_data.obsm,
            varm=new_data.varm,
            varp=new_data.varp,
            obsp=new_data.obsp,
            raw=new_data.raw,
            layers=new_data.layers,
            shape=new_data.shape,
            filename=new_data.filename,
            filemode="r+",
        )