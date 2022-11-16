from __future__ import annotations

import scipy as sp
import numpy as np
import gc
import logging
import math

from snapatac2._snapatac2 import AnnData, AnnDataSet, jm_regress, jaccard_similarity, cosine_similarity

def idf(data, features=None):
    n, m = data.shape
    count = np.zeros(m)
    for batch, _, _ in data.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))
    if features is not None:
        count = count[features]
    return np.log(n / (1 + count))

# FIXME: random state
def spectral(
    adata: AnnData | AnnDataSet,
    n_comps: int = 50,
    features: str | np.ndarray | None = "selected",
    random_state: int = 0,
    sample_size: int | float | None = None,
    chunk_size: int = 20000,
    distance_metric: str = "jaccard",
    feature_weights: str | np.ndarray | None = "idf",
    inplace: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute Laplacian Eigenmaps of chromatin accessibility profiles.

    Convert chromatin accessibility profiles of cells into lower dimensional representations
    using the spectrum of the normalized graph Laplacian defined by pairwise similarity
    between cells.

    Note
    ----
    The space complexity of this function is :math:`O(N^2)`, where $N$ is the minimum between
    the total of cells and the `sample_size`.
    The memory usage in bytes is given by $N^2 * 8 * 2$. For example,
    when $N = 10,000$ it will use roughly 745 MB memory.
    When `sample_size` is set, the Nystrom algorithm will be used to approximate
    the embedding. For large datasets, try to set the `sample_size` appropriately to
    reduce the memory usage.

    Parameters
    ----------
    adata
        AnnData or AnnDataSet object.
    n_comps
        Number of dimensions to keep.
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
    feature_weights
        Whether to weight features using "IDF".
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        if `inplace=True` it stores Spectral embedding of data in
        `adata.obsm["X_spectral"]` and `adata.uns["spectral_eigenvalue"]`.
        Otherwise, it returns the result as numpy arrays.
    """
    np.random.seed(random_state)

    if isinstance(features, str):
        if features in adata.var:
            features = adata.var[features]
        else:
            raise NameError("Please call `select_features` first or explicitly set `features = None`")

    if feature_weights is None:
        weights = None
    elif isinstance(feature_weights, np.ndarray):
        weights = feature_weights if features is None else feature_weights[features]
    elif feature_weights == "idf":
        weights = idf(adata, features)
    else:
        raise NameError("Invalid feature_weights")

    n_comps = min(adata.n_vars - 1, adata.n_obs - 1, n_comps)

    n_sample, _ = adata.shape
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
        if features is not None:
            X = adata.X[:, features]
        else:
            X = adata.X[...]
        model = Spectral(n_comps, distance_metric, weights)
        model.fit(X)
        result = model.transform()
    else:
        if adata.isbacked:
            S = adata.X.chunk(sample_size, replace=False)
        else:
            S = sp.sparse.csr_matrix(adata.chunk_X(sample_size, replace=False))
        if features is not None: S = S[:, features]

        model = Spectral(n_comps, distance_metric, weights)
        model.fit(S)

        from tqdm import tqdm
        logging.info("Perform Nystrom extension")
        for batch, _, _ in tqdm(adata.chunked_X(chunk_size), total=math.ceil(adata.n_obs/chunk_size)):
            if distance_metric == "jaccard":
                batch.data = np.ones(batch.indices.shape, dtype=np.float64)
            if features is not None: batch = batch[:, features]
            model.extend(batch)
        result = model.transform()

    if inplace:
        adata.uns['spectral_eigenvalue'] = result[0]
        adata.obsm['X_spectral'] = result[1]
    else:
        return result


class Spectral:
    def __init__(self, n_dim: int = 30, distance: str = "jaccard", feature_weights = None):

        #self.dim = mat.get_shape()[1]
        self.n_dim = n_dim
        self.distance = distance
        if (self.distance == "jaccard"):
            self.compute_similarity = lambda x, y=None: jaccard_similarity(x, y, feature_weights)
        elif (self.distance == "cosine"):
            self.compute_similarity = lambda x, y=None: cosine_similarity(x, y, feature_weights)
        else:
            from sklearn.metrics.pairwise import rbf_kernel
            self.compute_similarity = rbf_kernel

    def fit(self, mat, verbose: int = 1):
        """
        mat
            Sparse matrix, note that if `distance == jaccard`, the matrix will be
            interpreted as a binary matrix.
        """
        self.sample = mat
        self.dim = mat.shape[1]
        self.coverage = mat.sum(axis=1) / self.dim
        if verbose > 0:
            logging.info("Compute similarity matrix")
        A = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            if verbose > 0:
                logging.info("Normalization")
            self.normalizer = JaccardNormalizer(A, self.coverage)
            self.normalizer.normalize(A, self.coverage, self.coverage)
            np.fill_diagonal(A, 0)
            # Remove outlier
            self.normalizer.outlier = np.quantile(A, 0.999)
            np.clip(A, a_min=0, a_max=self.normalizer.outlier, out=A)
        else:
            np.fill_diagonal(A, 0)

        # M <- D^-1/2 * A * D^-1/2
        D = np.sqrt(A.sum(axis=1)).reshape((-1, 1))
        np.divide(A, D, out=A)
        np.divide(A, D.T, out=A)

        if verbose > 0:
            logging.info("Perform decomposition")
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
        return (self.evals[1:], self.evecs[:, 1:])

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