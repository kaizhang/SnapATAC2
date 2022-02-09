import scipy as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import gc

import anndata as ad
from snapatac2._snapatac2 import jm_regress
from typing import Optional, Union
from anndata.experimental import AnnCollection

from .._utils import read_as_binarized, binarized_chunk_X

# FIXME: random state
def spectral(
    data: ad.AnnData,
    n_comps: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    random_state: int = 0,
    sample_size: Optional[Union[int, float]] = None,
    chunk_size: Optional[int] = None,
    distance_metric: str = "jaccard",
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Compute Laplacian Eigenmaps of chromatin accessibility profiles.

    Convert chromatin accessibility profiles of cells into lower dimensional representations
    using the spectrum of the normalized graph Laplacian defined by pairwise similarity
    between cells.

    Parameters
    ----------
    data
        AnnData object
    n_comps
        Number of dimensions to keep
    features
        Boolean index mask. True means that the feature is kept.
        False means the feature is removed.
    random_state
        Seed of the random state generator
    sample_size
        Sample size used in the Nystrom method. It could be either an integer
        indicating the number of cells to sample or a real value from 0 to 1
        indicating the fraction of cells to sample.
    chunk_size
        Chunk size used in the Nystrom method
    distance_metric
        distance metric: "jaccard", "cosine".
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    if `inplace=True` it stores Spectral embedding of data in the field
    `adata.obsm["X_spectral"]`, otherwise it returns the result as a numpy array.
    """
    np.random.seed(random_state)
    if n_comps is None:
        min_dim = min(data.n_vars, data.n_obs)
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50
    if chunk_size is None: chunk_size = 20000

    (n_sample, _) = data.shape
    if sample_size is None:
        sample_size = n_sample
    elif isinstance(sample_size, int):
        if sample_size <= 1:
            raise ValueError("when sample_size is an integer, it should be > 1")
        if sample_size > n_sample:
            sample_size = n_sample
    else:
        if sample_size <= 0.0 or sample_size > 1.0:
            raise ValueError("when sample_size is a float, it should be > 0 and <= 1")
        else:
            sample_size = int(sample_size * n_sample)

    if sample_size >= n_sample:
        if isinstance(data, AnnCollection):
            X, _ = next(data.iterate_axis(n_sample))
            X = X.X[...]
            if distance_metric == "jaccard":
                X.data = np.ones(X.indices.shape, dtype=np.float64)
        elif isinstance(data, ad.AnnData):
            if data.isbacked:
                if data.is_view:
                    raise ValueError(
                        "View of AnnData object in backed mode is not supported."
                        "To save the object to file, use `.copy(filename=myfilename.h5ad)`."
                        "To load the object into memory, use `.to_memory()`.")
                else:
                    X = read_as_binarized(data)
            else:
                X = data.X[...]
                if distance_metric == "jaccard":
                    X.data = np.ones(X.indices.shape, dtype=np.float64)
        else:
            raise ValueError("input should be AnnData or AnnCollection")

        if features is not None: X = X[:, features]

        model = Spectral(n_dim=n_comps, distance=distance_metric)
        model.fit(X)
        data.uns['spectral_eigenvalue'] = model.evals[1:]
        result = model.transform()
    else:
        if isinstance(data, AnnCollection):
            S, sample_indices = next(data.iterate_axis(sample_size, shuffle=True))
            S = S.X[...]
            if distance_metric == "jaccard":
                S.data = np.ones(S.indices.shape, dtype=np.float64)
            chunk_iterator = map(lambda b: b[0].X[...], data.iterate_axis(chunk_size))
        elif isinstance(data, ad.AnnData):
            if distance_metric == "jaccard":
                S = binarized_chunk_X(data, select=sample_size, replace=False)
            else:
                S = ad.chunk_X(data, select=sample_size, replace=False)
            chunk_iterator = map(lambda b: b[0], data.chunked_X(chunk_size))
        else:
            raise ValueError("input should be AnnData or AnnCollection")

        if features is not None: S = S[:, features]

        model = Spectral(n_dim=n_comps, distance=distance_metric)
        model.fit(S)
        data.uns['spectral_eigenvalue'] = model.evals[1:]

        from tqdm import tqdm
        import math
        print("Perform Nystrom extension")
        for batch in tqdm(chunk_iterator, total = math.ceil(n_sample / chunk_size)):
            if distance_metric == "jaccard":
                batch.data = np.ones(batch.indices.shape, dtype=np.float64)
            if features is not None: batch = batch[:, features]
            model.extend(batch)
        result = model.transform()

    if inplace:
        data.obsm['X_spectral'] = result
    else:
        return result

class Spectral:
    def __init__(self, n_dim=30, distance="jaccard"):

        #self.dim = mat.get_shape()[1]
        self.n_dim = n_dim
        self.distance = distance
        if (self.distance == "jaccard"):
            self.compute_similarity = jaccard_similarity
        elif (self.distance == "cosine"):
            self.compute_similarity = cosine_similarity
        else:
            self.compute_similarity = rbf_kernel

    def fit(self, mat):
        self.sample = mat
        self.dim = mat.shape[1]
        self.coverage = mat.sum(axis=1) / self.dim
        print("Compute similarity matrix")
        A = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            print("Normalization")
            self.normalizer = JaccardNormalizer(A, self.coverage)
            self.normalizer.normalize(A, self.coverage, self.coverage)
            np.fill_diagonal(A, 0)
            # Remove outlier
            self.normalizer.outlier = np.quantile(np.asarray(A), 0.999)
            np.clip(A, a_min=0, a_max=self.normalizer.outlier, out=A)
        else:
            np.fill_diagonal(A, 0)
            A = np.matrix(A)

        # M <- D^-1/2 * A * D^-1/2
        D = np.sqrt(A.sum(axis=1))
        np.divide(A, D, out=A)
        np.divide(A, D.T, out=A)

        print("Perform decomposition")
        evals, evecs = sp.sparse.linalg.eigsh(A, self.n_dim + 1, which='LM')
        ix = evals.argsort()[::-1]
        self.evals = np.real(evals[ix])
        self.evecs = np.real(evecs[:, ix])

        B = np.divide(self.evecs, D)
        np.divide(B, self.evals.reshape((1, -1)), out=B)

        self.B = B
        self.Q = []

        return self

    def extend(self, data):
        A = self.compute_similarity(self.sample, data)
        if (self.distance == "jaccard"):
            self.normalizer.normalize(
                A, self.coverage, data.sum(axis=1) / self.dim,
                clip_min=0, clip_max=self.normalizer.outlier
            )
        self.Q.append(A.T @ self.B)

    def transform(self, orthogonalize = True):
        if len(self.Q) > 0:
            Q = np.concatenate(self.Q, axis=0)
            D_ = np.sqrt(np.multiply(Q, self.evals.reshape(1, -1)) @ Q.sum(axis=0).T)
            np.divide(Q, D_.reshape((-1, 1)), out=Q)

            if orthogonalize:
                # orthogonalization
                sigma, V = np.linalg.eig(Q.T @ Q)
                sigma = np.sqrt(sigma)
                B = np.multiply(V.T, self.evals.reshape((1,-1))) @ V
                np.multiply(B, sigma.reshape((-1, 1)), out=B)
                np.multiply(B, sigma.reshape((1, -1)), out=B)
                evals_new, evecs_new = np.linalg.eig(B)

                # reorder
                ix = evals_new.argsort()[::-1]
                self.evals = evals_new[ix]
                evecs_new = evecs_new[:, ix]

                np.divide(evecs_new, sigma.reshape((-1, 1)), out=evecs_new)
                self.evecs = Q @ V @ evecs_new
            else:
                self.evecs = Q
        return self.evecs[:, 1:]

def jaccard_similarity(m1, m2=None):
    """
    Compute pair-wise jaccard index

    Parameters
    ----------
    mat1
        n1 x m
    mat2
        n2 x m
    
    Returns
    -------
        Jaccard similarity matrix
    """
    s1 = m1.sum(axis=1)
    if m2 is None:
        d = m1.dot(m1.T).todense()
        gc.collect()
        t = np.negative(d)
        t += s1
        t += s1.T
        d /= t
    else:
        s2 = m2.sum(axis=1)
        d = m1.dot(m2.T).todense()
        gc.collect()
        d /= -d + s1 + s2.T
    gc.collect()
    return d

class JaccardNormalizer:
    def __init__(self, jm, c):
        (slope, intersect) = jm_regress(jm, c)
        self.slope = slope
        self.intersect = intersect
        self.outlier = None

    def normalize(self, jm, c1, c2, clip_min=None, clip_max=None):
        # jm / (self.slope / (1 / c1 + 1 / c2.T - 1) + self.intersect)
        temp = 1 / c1 + 1 / c2.T
        temp -= 1
        np.reciprocal(temp, out=temp)
        np.multiply(temp, self.slope, out=temp)
        temp += self.intersect
        jm /= temp
        if clip_min is not None or clip_max is not None:
            np.clip(jm, a_min=clip_min, a_max=clip_max, out=jm)
        gc.collect()


"""
def nystrom_full(mat, sample_size, n_dim):
    n, m = mat.shape
    sample_indices = np.random.choice(n, size=sample_size, replace=False)
    mask = np.ones(n, bool)
    mask[sample_indices] = False
    sample = mat[sample_indices, :]
    other_data = mat[mask, :]

    # Compute affinity matrix
    coverage = sample.sum(axis=1) / m
    A = jaccard_similarity(sample)
    normalizer = JaccardNormalizer(A, coverage)
    normalizer.normalize(A, coverage, coverage)
    np.fill_diagonal(A, 0)
    normalizer.outlier = np.quantile(np.asarray(A), 0.999)
    np.clip(A, a_min=0, a_max=normalizer.outlier, out=A)

    # Compute distance matrix B
    B = jaccard_similarity(sample, other_data)
    normalizer.normalize(B, coverage, other_data.sum(axis=1) / m,
        clip_min=0, clip_max=normalizer.outlier)

    # Compute degree
    a_r = A.sum(axis=1)
    b_r = B.sum(axis=1)
    b_c = B.sum(axis=0).reshape((-1, 1))
    d1 = np.sqrt(a_r + b_r)
    d2 = np.sqrt(np.clip(b_c + B.T @ np.linalg.pinv(A) @ b_r, a_min=1e-10, a_max=None))

    # normalization
    np.divide(A, d1, out=A)
    np.divide(A, d1.T, out=A)
    np.divide(B, d1.reshape((-1, 1)), out=B)
    np.divide(B, d2.reshape((1, -1)), out=B)

    # compute eigenvector
    evals, U = sp.sparse.linalg.eigsh(A, n_dim + 1, which='LM')
    U_ = np.divide(B.T @ U, evals.reshape((1, -1)))
    result = np.empty((n, n_dim))
    result[sample_indices] = U[:, 1:]
    result[mask] = U_[:, 1:]
    return result
"""