import numpy as np
import scipy.sparse as ss
import anndata as ad

hg38 = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415
}

GRCm39 = {
    "chr1": 195154279,
    "chr2": 181755017,
    "chr3": 159745316,
    "4": 156860686,
    "5": 151758149,
    "6": 149588044,
    "7": 144995196,
    "8": 130127694,
    "9": 124359700,
    "10": 130530862,
    "11": 121973369,
    "12": 120092757,
    "13": 120883175,
    "14": 125139656,
    "15": 104073951,
    "16": 98008968,
    "17": 95294699,
    "18": 90720763,
    "19": 61420004,
    "X": 169476592,
    "Y": 91455967,
}



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