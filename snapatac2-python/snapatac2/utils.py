import numpy as np
import scipy.sparse as ss
import anndata as ad

def read_as_binarized(adata: ad.AnnData) -> ss.spmatrix:
    grp = adata.file["X"]
    mtx = ss.csr_matrix(adata.shape, dtype=np.float64)
    mtx.indices = grp["indices"][...]
    mtx.indptr = grp["indptr"][...]
    mtx.data = np.ones(mtx.indices.shape, dtype=np.float64)
    return mtx

'''
def binarized_chunk_X(
    adata: ad.AnnData
    select: Union[int, Sequence[int], np.ndarray] = 1000,
    replace: bool = False,
):
    """ Return a chunk of the data matrix :attr:`X` with random or specified indices.
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
        select = select if select < self.n_obs else self.n_obs
        choice = np.random.choice(self.n_obs, select, replace)
    elif isinstance(select, (np.ndarray, cabc.Sequence)):
        choice = np.asarray(select)
    else:
        raise ValueError("select should be int or array")

    reverse = None
    if self.isbacked:
        # h5py can only slice with a sorted list of unique index values
        # so random batch with indices [2, 2, 5, 3, 8, 10, 8] will fail
        # this fixes the problem
        indices, reverse = np.unique(choice, return_inverse=True)
        selection = self.X[indices.tolist()]
    else:
        selection = self.X[choice]

    selection = selection.toarray() if issparse(selection) else selection
    return selection if reverse is None else selection[reverse]

def binarized_chunked_X(self, chunk_size: Optional[int] = None):
    """
    Return an iterator over the rows of the data matrix :attr:`X`.
    Parameters
    ----------
    chunk_size
        Row size of a single chunk.
    """
    if chunk_size is None:
        # Should be some adaptive code
        chunk_size = 6000
    start = 0
    n = self.n_obs
    for _ in range(int(n // chunk_size)):
        end = start + chunk_size
        yield (self.X[start:end], start, end)
        start = end
    if start < n:
        yield (self.X[start:n], start, n)


'''