from __future__ import annotations

import numpy as np
import itertools
from scipy.special import logsumexp

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata 

def mnc_correct(
    adata: internal.AnnData | internal.AnnDataSet | np.adarray,
    *,
    batch: str | list[str],
    n_neighbors: int = 5,
    n_clusters: int = 40,
    n_iter: int = 1,
    use_rep: str = "X_spectral",
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | None:
    """
    A modified MNN-Correct algorithm based on cluster centroid.

    Parameters
    ----------
    data
        Matrice or AnnData object. Matrices should be shaped like n_obs x n_vars.
    batch
        Batch labels for cells. If a string, labels will be obtained from `obs`.
    n_neighbors
        Number of mutual nearest neighbors.
    n_clusters
        Number of clusters
    n_iter
        Number of iterations.
    use_rep
        Use the indicated representation in `.obsm`.
    use_dims
        Use these dimensions in `use_rep`.
    groupby
        If specified, split the data into groups and perform batch correction
        on each group separately.
    key_added
        If specified, add the result to ``adata.obsm`` with this key. Otherwise,
        it will be stored in ``adata.obsm[use_rep + "_mnn"]``.
    inplace
        Whether to store the result in the anndata object.
    n_jobs
        Number of jobs to use for parallelization.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it updates adata with the field
        ``adata.obsm[`use_rep`_mnn]``, containing adjusted principal components.
        Otherwise, it returns the result as a numpy array.
    """
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]
    mat = np.asarray(mat)

    if isinstance(batch, str):
        batch = adata.obs[batch]

    if groupby is None:
        mat = _mnc_correct_main(mat, batch, n_iter, n_neighbors, n_clusters)
    else:
        from multiprocess import Pool

        if isinstance(groupby, str): groupby = adata.obs[groupby]

        group_indices = {}
        for i, group in enumerate(groupby):
            if group in group_indices:
                group_indices[group].append(i)
            else:
                group_indices[group] = [i]
        group_indices = [x for x in group_indices.values()]

        inputs = [(mat[group_idx, :], batch[group_idx]) for group_idx in group_indices]
        with Pool(n_jobs) as p:
            results = p.map(lambda x: _mnc_correct_main(x[0], x[1], n_iter, n_neighbors, n_clusters), inputs)
        for idx, result in zip(group_indices, results):
            mat[idx, :] = result

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_mnn"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat

def _mnc_correct_main(
    data_matrix,
    batch_labels,
    n_iter,
    n_neighbors,
    n_clusters,
    random_state=0
):
    label_uniq = list(set(batch_labels))

    if len(label_uniq) > 1:
        for _ in range(n_iter):
            batch_idx = []
            data_by_batch = []
            for label in label_uniq:
                idx = [i for i, x in enumerate(batch_labels) if x == label]
                batch_idx.append(idx)
                data_by_batch.append(data_matrix[idx,:])
            new_matrix = _mnc_correct_multi(data_by_batch, n_neighbors, n_clusters, random_state)

            idx = list(itertools.chain.from_iterable(batch_idx))
            idx = np.argsort(idx)
            data_matrix = new_matrix[idx, :]
    return data_matrix

def _mnc_correct_multi(datas, n_neighbors, n_clusters, random_state):
    data0 = datas[0]
    for i in range(1, len(datas)):
        data1 = datas[i]
        pdata0, pdata1 = _mnc_correct_pair(data0, data1, n_clusters, n_neighbors, random_state)
        ratio = data0.shape[0] / (data0.shape[0] + data1.shape[0])
        data0_ = pdata0.project(data0, weight = 1 - ratio)
        data1_ = pdata1.project(data1, weight = ratio)
        data0 = np.concatenate((data0_, data1_), axis=0)
    return data0

def _mnc_correct_pair(X, Y, n_clusters, n_neighbors, random_state):
    from sklearn.neighbors import KDTree
    from sklearn.cluster import KMeans

    n_X = X.shape[0]
    n_Y = Y.shape[0]
    c_X = KMeans(
        n_clusters=min(n_clusters, n_X), n_init=10, random_state=random_state
    ).fit(X).cluster_centers_
    c_Y = KMeans(
        n_clusters=min(n_clusters, n_Y), n_init=10, random_state=random_state
    ).fit(Y).cluster_centers_

    tree_X = KDTree(c_X)
    tree_Y = KDTree(c_Y)

    # X by Y matrix
    m_X = tree_Y.query(c_X, k=min(n_neighbors, n_Y), return_distance=False)

    # Y by X matrix
    m_Y_ = tree_X.query(c_Y, k=min(n_neighbors, n_X), return_distance=False)
    m_Y = []
    for i in range(m_Y_.shape[0]):
        m_Y.append(set(m_Y_[i,:]))

    i_X = []
    i_Y = []
    for i in range(m_X.shape[0]):
        for j in m_X[i]:
            if i in m_Y[j]:
                i_X.append(i)
                i_Y.append(j)
    a = c_X[i_X,:]
    b = c_Y[i_Y,:]
    return (Projector(a, b), Projector(b, a))

class Projector(object):
    def __init__(self, X, Y):
        self.reference = X
        self.vector = Y - X

    def project(self, X, weight=0.5):
        def project(x):
            P = self.reference
            U = self.vector
            d = np.sqrt(np.sum((P - x)**2, axis=1))
            w = _normalize(-(d/0.005))
            #w = 1/d
            return (x + weight * np.average(U, axis=0, weights=w))
        return np.apply_along_axis(project, 1, X)

# exp transform the weights and then normalize them to sum to 1.
def _normalize(ws):
    s = logsumexp(ws)
    return np.exp(ws - s)