from typing import Optional, Union
from scipy.sparse import vstack
import numpy as np
import scipy.sparse as ss

from snapatac2._snapatac2 import AnnData, AnnDataSet

def smooth(
    adata: Union[AnnData, AnnDataSet, ss.spmatrix],
    distances: Optional[Union[str, ss.spmatrix]] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Smoothing.

    Parameters
    ----------
    adata
        AnnData or AnnDataSet object.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    if `inplace=True` it stores Spectral embedding of data in the field
    `adata.obsm["X_spectral"]`,
    `adata.uns["spectral_eigenvalue"]`,
    otherwise it returns the result as a numpy array.
    """
 
    if distances is None: distances = "distances"
    if isinstance(str, distances): distances = adata.obsp[distances] 
    data = adata.X[:] if isinstance(AnnData, adata) or isinstance(AnnDataSet, adata) else adata
    data = data * make_diffuse_operator(distances)
    if inplace:
        adata.X = data
    else:
        return data

def make_diffuse_operator(knn_d, t = 3):
    return make_markov_matrix(knn_d)**t

def make_markov_matrix(knn_d):
    """
    Turn a (knn) distance matrix into a markov matrix.
    """
    rows = []
    for i in range(knn_d.shape[0]):
        row = knn_d.getrow(i)
        # local nearest neighbor for estimating local density
        ka = int(row.nnz / 3)
        # set sigma for each cell i to the distance to its kath nearest neighbor
        # FIXME: corner case where sigma == 0
        sigma = np.sort(row.data)[ka - 1]
        # apply guassian kernel to get the affinity matrix
        row.data = np.exp(-np.square(row.data / sigma))
        rows.append(row)
    affinity = vstack(rows, format="csr")

    # symmetrize the affinity matrix
    affinity += affinity.T
    # make stochastic matrix
    s = np.ravel(affinity.sum(axis=1))
    for i in range(affinity.shape[0]):
        affinity.data[affinity.indptr[i] : affinity.indptr[i+1]] /= s[i]
    return affinity