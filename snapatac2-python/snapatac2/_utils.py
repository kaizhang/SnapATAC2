from ntpath import join
from typing import Union, Sequence, Literal, Optional, Dict
import numpy as np
import collections.abc as cabc
import scipy.sparse as ss
import anndata as ad
from anndata import AnnData
from anndata.experimental import AnnCollection

def create_ann_collection(
    adatas: Union[Sequence[AnnData], Dict[str, AnnData]],
    join_obs: Optional[Literal["inner", "outer"]] = "inner",
    join_obsm: Optional[Literal["inner"]] = None,
    join_vars: Optional[Literal["inner"]] = None,
    label: Optional[str] = None,
    keys: Optional[Sequence[str]] = None,
    index_unique: Optional[str] = '_',
    convert = None,
    harmonize_dtypes: bool = True,
    indices_strict: bool = True,
) -> AnnCollection:
    """
    adatas
        The objects to be lazily concatenated.
        If a Mapping is passed, keys are used for the `keys` argument and values are concatenated.
    join_obs
        If "inner" specified all `.obs` attributes from `adatas` will be inner joined
        and copied to this object.
        If "outer" specified all `.obsm` attributes from `adatas` will be outer joined
        and copied to this object.
        For "inner" and "outer" subset objects will access `.obs` of this object,
        not the original `.obs` attributes of `adatas`.
        If `None`, nothing is copied to this object's `.obs`, a subset object will directly
        access `.obs` attributes of `adatas` (with proper reindexing and dtype conversions).
        For `None`the inner join rule is used to select columns of `.obs` of `adatas`.
    join_obsm
        If "inner" specified all `.obsm` attributes from `adatas` will be inner joined
        and copied to this object. Subset objects will access `.obsm` of this object,
        not the original `.obsm` attributes of `adatas`.
        If `None`, nothing is copied to this object's `.obsm`, a subset object will directly
        access `.obsm` attributes of `adatas` (with proper reindexing and dtype conversions).
        For both options the inner join rule for the underlying `.obsm` attributes is used.
    join_vars
        Specify how to join `adatas` along the var axis. If `None`, assumes all `adatas`
        have the same variables. If "inner", the intersection of all variables in
        `adatas` will be used.
    label
        Column in `.obs` to place batch information in.
        If it's None, no column is added.
    keys
        Names for each object being added. These values are used for column values for
        `label` or appended to the index if `index_unique` is not `None`. Defaults to
        incrementing integer labels.
    index_unique
        Whether to make the index unique by using the keys. If provided, this
        is the delimeter between "{orig_idx}{index_unique}{key}". When `None`,
        the original indices are kept.
    convert
        You can pass a function or a Mapping of functions which will be applied
        to the values of attributes (`.obs`, `.obsm`, `.layers`, `.X`) or to specific
        keys of these attributes in the subset object.
        Specify an attribute and a key (if needed) as keys of the passed Mapping
        and a function to be applied as a value.
    harmonize_dtypes
        If `True`, all retrieved arrays from subset objects will have the same dtype.
    indices_strict
        If  `True`, arrays from the subset objects will always have the same order
        of indices as in selection used to subset.
        This parameter can be set to `False` if the order in the returned arrays
        is not important, for example, when using them for stochastic gradient descent.
        In this case the performance of subsetting can be a bit better.
    """
    import pandas as pd
    from anndata._core.aligned_mapping import AxisArrays

    data = AnnCollection(
        adatas,
        join_obs,
        join_obsm,
        join_vars,
        label,
        keys,
        index_unique,
        convert, 
        harmonize_dtypes,
        indices_strict,
    )
    data.var = pd.DataFrame()
    if join_obsm is None:
        data._obsm = AxisArrays(data, axis=0) 
        data._view_attrs_keys.pop("obsm", None)

    data.uns = {}
    return data

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