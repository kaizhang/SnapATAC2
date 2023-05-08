from __future__ import annotations
from typing_extensions import Literal

import scipy as sp
import numpy as np
import gc
import logging
import math

from snapatac2._utils import get_igraph_from_adjacency, is_anndata 
from snapatac2._snapatac2 import (AnnData, AnnDataSet, jm_regress, jaccard_similarity,
    cosine_similarity, spectral_embedding, multi_spectral_embedding, spectral_embedding_nystrom)

__all__ = ['umap2', 'umap', 'spectral', 'multi_spectral']

def umap2(
    adata: AnnData | AnnDataSet | np.ndarray,
    n_comps: int = 2,
    key_added: str = 'umap',
    random_state: int = 0,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        The number of dimensions of the embedding.
    key_added
        `adata.obs` key under which to add the cluster labels.
    random_state
        Random seed.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it stores UMAP embedding in
        `adata.obsm["X_`key_added`"]`.
        Otherwise, it returns the result as a numpy array.
    """
    from igraph import set_random_number_generator
    import random
    random.seed(random_state)
    set_random_number_generator(random)

    if is_anndata(adata):
        adjacency = adata.obsp["distances"]
    else:
        inplace = False
        adjacency = adata
    gr = get_igraph_from_adjacency(adjacency)
    dist = np.array(gr.es["weight"])
    umap = gr.layout_umap(dist=dist, dim=n_comps, min_dist=0.01, epochs=500)
    umap = np.array(umap.coords)

    if inplace:
        adata.obsm["X_" + key_added] = umap
    else:
        return umap

def umap(
    adata: AnnData | AnnDataSet | np.ndarray,
    n_comps: int = 2,
    use_dims: int | list[int] | None = None,
    use_rep: str = "X_spectral",
    key_added: str = 'umap',
    random_state: int = 0,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        The number of dimensions of the embedding.
    use_dims
        Use these dimensions in `use_rep`.
    use_rep
        Use the indicated representation in `.obsm`.
    key_added
        `adata.obs` key under which to add the cluster labels.
    random_state
        Random seed.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it stores UMAP embedding in
        `adata.obsm["X_`key_added`"]`.
        Otherwise, it returns the result as a numpy array.
    """
    from umap import UMAP

    data = adata.obsm[use_rep] if is_anndata(adata) else adata

    if use_dims is not None:
        data = data[:, :use_dims] if isinstance(use_dims, int) else data[:, use_dims]

    umap = UMAP(random_state=random_state, n_components=n_comps).fit_transform(data)
    if inplace:
        adata.obsm["X_" + key_added] = umap
    else:
        return umap

def idf(data, features=None):
    n, m = data.shape
    count = np.zeros(m)
    for batch, _, _ in data.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))
    if features is not None:
        count = count[features]
    return np.log(n / (1 + count))

def spectral(
    adata: AnnData | AnnDataSet,
    n_comps: int = 30,
    features: str | np.ndarray | None = "selected",
    random_state: int = 0,
    sample_size: int | float | None = None,
    sample_method: Literal["random", "degree"] = "random",
    chunk_size: int = 20000,
    distance_metric: Literal["jaccard", "cosine"] = "cosine",
    weighted_by_sd: bool = True,
    feature_weights: list[float] | None = None,
    inplace: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Perform dimension reduction using Laplacian Eigenmaps.

    Convert the cell-by-feature count matrix into lower dimensional representations
    using the spectrum of the normalized graph Laplacian defined by pairwise similarity
    between cells.

    Note
    ----
    When using the cosine similarity as the similarity metric, the matrix-free
    spectral embedding algorithm is used, which scales linearly with the number of cells. 
    The memory usage is roughly :math:`2 \times input_size`.
    For other types of similarity metrics, the time and space complexity is :math:`O(N^2)`,
    where $N$ is the minimum between the total of cells and the `sample_size`.
    The memory usage in bytes is given by $N^2 * 8 * 2$. For example,
    when $N = 10,000$ it will use roughly 745 MB memory.
    When `sample_size` is set, the Nystrom algorithm will be used to approximate
    the embedding. 

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
    weighted_by_sd
        Whether to weight the result eigenvectors by the square root of eigenvalues.
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
        if distance_metric == "cosine":
            evals, evecs = spectral_embedding(adata, features, n_comps, feature_weights)
        else:
            if feature_weights is None:
                feature_weights = idf(adata, features)
            model = Spectral(n_comps, distance_metric, feature_weights)
            X = adata.X[...] if features is None else adata.X[:, features]
            model.fit(X)
            evals, evecs = model.transform()
    else:
        if distance_metric == "cosine":
            if sample_method == "random":
                weighted_by_degree = False
            else:
                weighted_by_degree = True
            v, u = spectral_embedding_nystrom(adata, features, n_comps, sample_size, weighted_by_degree, chunk_size)
            evals, evecs = orthogonalize(v, u)
        else:
            if feature_weights is None:
                feature_weights = idf(adata, features)
            model = Spectral(n_comps, distance_metric, feature_weights)
            if adata.isbacked:
                S = adata.X.chunk(sample_size, replace=False)
            else:
                S = sp.sparse.csr_matrix(adata.chunk_X(sample_size, replace=False))
            if features is not None: S = S[:, features]

            model.fit(S)

            from tqdm import tqdm
            logging.info("Perform Nystrom extension")
            for batch, _, _ in tqdm(adata.chunked_X(chunk_size), total=math.ceil(adata.n_obs/chunk_size)):
                if distance_metric == "jaccard":
                    batch.data = np.ones(batch.indices.shape, dtype=np.float64)
                if features is not None: batch = batch[:, features]
                model.extend(batch)
            evals, evecs = model.transform()

    if weighted_by_sd:
        idx = [i for i in range(evals.shape[0]) if evals[i] > 0]
        evals = evals[idx]
        evecs = evecs[:, idx] * np.sqrt(evals)

    if inplace:
        adata.uns['spectral_eigenvalue'] = evals
        adata.obsm['X_spectral'] = evecs
    else:
        return (evals, evecs)


class Spectral:
    def __init__(
        self,
        out_dim: int = 30,
        distance: Literal["jaccard", "cosine"] = "jaccard",
        feature_weights = None,
    ):
        self.out_dim = out_dim
        self.distance = distance
        if (self.distance == "jaccard"):
            self.compute_similarity = lambda x, y=None: jaccard_similarity(x, y, feature_weights)
        elif (self.distance == "cosine"):
            self.compute_similarity = lambda x, y=None: cosine_similarity(x, y, feature_weights)
        elif (self.distance == "rbf"):
            from sklearn.metrics.pairwise import rbf_kernel
            self.compute_similarity = lambda x, y=None: rbf_kernel(x, y)
        else:
            raise ValueError("Invalid distance")

    def fit(self, mat, verbose: int = 1):
        """
        mat
            Sparse matrix, note that if `distance == jaccard`, the matrix will be
            interpreted as a binary matrix.
        """
        self.sample = mat
        self.in_dim = mat.shape[1]
        if verbose > 0:
            logging.info("Compute similarity matrix")
        A = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            if verbose > 0:
                logging.info("Normalization")
            self.coverage = mat.sum(axis=1) / self.in_dim
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
        evals, evecs = sp.sparse.linalg.eigsh(A, self.out_dim, which='LM')
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
                A, self.coverage, data.sum(axis=1) / self.in_dim,
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
        return (self.evals, self.evecs)

def orthogonalize(evals, evecs):
    sigma, V = np.linalg.eig(evecs.T @ evecs)
    sigma = np.sqrt(sigma)
    B = np.multiply(V.T, evals.reshape((1,-1))) @ V
    np.multiply(B, sigma.reshape((-1, 1)), out=B)
    np.multiply(B, sigma.reshape((1, -1)), out=B)
    evals_new, evecs_new = np.linalg.eig(B)

    # reorder
    ix = evals_new.argsort()[::-1]
    evals_new = evals_new[ix]
    evecs_new = evecs_new[:, ix]

    np.divide(evecs_new, sigma.reshape((-1, 1)), out=evecs_new)
    evecs_new = evecs @ V @ evecs_new
    return (evals_new, evecs_new)

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

class SpectralMatrixFree:
    """Matrix-free spectral embedding without computing the similarity matrix explicitly.

    Only cosine similarity is supported.
    """
    def __init__(
        self,
        out_dim: int = 30,
        feature_weights = None,
    ):
        self.out_dim = out_dim
        self.feature_weights = feature_weights

    def fit(self, mat, verbose: int = 1):
        if self.feature_weights is not None:
            mat = mat @ sp.sparse.diags(self.feature_weights)
        self.sample = mat
        self.in_dim = mat.shape[1]

        s = 1 / np.sqrt(np.ravel(sp.sparse.csr_matrix.power(mat, 2).sum(axis = 1)))
        X = sp.sparse.diags(s) @ mat

        D = np.ravel(X @ X.sum(axis = 0).T) - 1
        X = sp.sparse.diags(1 / np.sqrt(D)) @ X
        evals, evecs = _eigen(X, 1 / D, k=self.out_dim)

        ix = evals.argsort()[::-1]
        self.evals = evals[ix]
        self.evecs = evecs[:, ix]

        self.Q = []
        return self

    def extend(self, data):
        raise NotImplementedError

    def transform(self, orthogonalize = True):
        if len(self.Q) > 0:
            raise NotImplementedError
        return (self.evals, self.evecs)

def _eigen(X, D, k):
    def f(v):
        return X @ (v.T @ X).T - D * v

    n = X.shape[0]
    A = sp.sparse.linalg.LinearOperator((n, n), matvec=f, dtype=np.float64)
    return sp.sparse.linalg.eigsh(A, k=k)

def multi_spectral(
    adatas: list[AnnData] | list[AnnDataSet], 
    n_comps: int = 30,
    features: str | list[str] | list[np.ndarray] | None = "selected",
    weights: list[float] | None = None,
    random_state: int = 0,
    weighted_by_sd: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Laplacian Eigenmaps simultaneously on multiple modalities, with linear
    space and time complexity.

    This is similar to :func:`~snapatac2.tl.spectral`, but it can work on multiple modalities.

    Parameters
    ----------
    adatas
        A list of AnnData objects, representing single-cell data from different modalities.
    n_comps
        Number of dimensions to keep.
    features
        Boolean index mask. True means that the feature is kept. False means the feature is removed.
    weights
        Weights for each modality. If None, all modalities are weighted equally.
    random_state
        Seed of the random state generator
    weighted_by_sd
        Whether to weight the result eigenvectors by the square root of eigenvalues.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Return the eigenvalues and eigenvectors of the Laplacian matrix.
    """
    np.random.seed(random_state)

    if features is None or isinstance(features, str):
        features = [features] * len(adatas)
    if all(isinstance(f, str) for f in features):
        features = [adata.var[feature] for adata, feature in zip(adatas, features)]

    if weights is None:
        weights = [1.0 for _ in adatas]

    evals, evecs = multi_spectral_embedding(adatas, features, weights, n_comps)

    if weighted_by_sd:
        idx = [i for i in range(evals.shape[0]) if evals[i] > 0]
        evals = evals[idx]
        evecs = evecs[:, idx] * np.sqrt(evals)

    return (evals, evecs)