import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import gc

import anndata as ad
from snapatac2._snapatac2 import jm_regress
from typing import Optional, Union
from anndata.experimental import AnnCollection

from .._utils import read_as_binarized

# FIXME: random state
def spectral(
    data: ad.AnnData,
    n_comps: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    random_state: int = 0,
    sample_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    data
        AnnData object
    n_comps
        Number of dimensions to keep
    random_state
        Seed of the random state generator
    sample_size
        Sample size used in the Nystrom method
    chunk_size
        Chunk size used in the Nystrom method
    Returns
    -------
    None
    """
    if n_comps is None:
        min_dim = min(data.n_vars, data.n_obs)
        if 50 >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = 50
    (n_sample, _) = data.shape

    if sample_size is None or sample_size >= n_sample:
        if isinstance(data, AnnCollection):
            X, _ = next(data.iterate_axis(n_sample))
            X = X.X[...]
            X.data = np.ones(X.indices.shape, dtype=np.float64)
        else:
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
                X.data = np.ones(X.indices.shape, dtype=np.float64)

        if features is not None: X = X[:, features]

        model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=1)
        model.fit(X)
        data.obsm['X_spectral'] = model.evecs[:, 1:]

    else:
        if isinstance(data, AnnCollection):
            S, sample_indices = next(data.iterate_axis(sample_size, shuffle=True))
            S = S.X[...]
            S.data = np.ones(S.indices.shape, dtype=np.float64)
        else:
            pass

        if features is not None: S = S[:, features]

        model = Spectral(n_dim=n_comps, distance="jaccard", sampling_rate=sample_size / n_sample)
        model.fit(S)
        if chunk_size is None: chunk_size = 2000

        from tqdm import tqdm
        import math
        result = []
        print("Perform Nystrom extension")
        for batch, _ in tqdm(data.iterate_axis(chunk_size), total = math.ceil(n_sample / chunk_size)):
            batch = batch.X[...]
            batch.data = np.ones(batch.indices.shape, dtype=np.float64)
            if features is not None: batch = batch[:, features]
            result.append(model.predict(batch)[:, 1:])
        data.obsm['X_spectral'] = np.concatenate(result, axis=0)

class Spectral:
    def __init__(self, n_dim=30, sampling_rate=1, distance="jaccard"):
        self.sampling_rate = sampling_rate
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
        S = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            print("Normalization")
            self.normalizer = JaccardNormalizer(S, self.coverage)
            self.normalizer.normalize(S, self.coverage, self.coverage)

        np.fill_diagonal(S, 0)
        d = 1 / (self.sampling_rate * S.sum(axis=1))
        np.multiply(S, d, out=S)

        print("Perform decomposition")
        evals, evecs = sp.sparse.linalg.eigs(S, self.n_dim + 1, which='LR')
        ix = evals.argsort()[::-1]
        self.evals = np.real(evals[ix])
        self.evecs = np.real(evecs[:, ix])

    def predict(self, data=None):
        if data == None:
            return self.evecs
        S = self.compute_similarity(self.sample, data)

        if (self.distance == "jaccard"):
            self.normalizer.normalize(S, self.coverage, data.sum(axis=1) / self.dim)
        S = S.T

        d = 1 / (self.sampling_rate * S.sum(axis=1))
        np.multiply(S, d, out=S)

        evecs = (S.dot(self.evecs)).dot(np.diag(1/self.evals))
        return evecs

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

    def normalize(self, jm, c1, c2):
        # jm / (self.slope / (1 / c1 + 1 / c2.T - 1) + self.intersect)
        temp = 1 / c1 + 1 / c2.T
        temp -= 1
        np.reciprocal(temp, out=temp)
        np.multiply(temp, self.slope, out=temp)
        temp += self.intersect
        jm /= temp
        gc.collect()

def old_jaccard_similarity(mat1, mat2=None):
    """Compute pair-wise jaccard index

    Parameters
    mat1
        n1 x m
    mat2
        n2 x m
    
    Returns
    -------
        Jaccard similarity matrix
    """
    coverage1 = mat1.sum(axis=1)
    if(mat2 != None):
        coverage2 = mat2.sum(axis=1)
        jm = mat1.dot(mat2.T).todense()
        n1, n2 = jm.shape
        c1 = coverage1.dot(np.ones((1,n2)))
        c2 = coverage2.dot(np.ones((1,n1)))
        jm = jm / (c1 + c2.T - jm)
    else:
        n, _ = mat1.get_shape()
        jm = mat1.dot(mat1.T).todense()
        c = coverage1.dot(np.ones((1,n)))
        jm = jm / (c + c.T - jm)
    return jm

class Old_JaccardNormalizer:
    def __init__(self, jm, c):
        n, _ = jm.shape

        X = 1 / c.dot(np.ones((1,n)))
        X = 1 / (X + X.T - 1)
        X = X[np.triu_indices(n, k = 1)].T
        y = jm[np.triu_indices(n, k = 1)].T

        self.model = LinearRegression().fit(X, y)

    def predict(self, jm, c1, c2):
        X1 = 1 / c1.dot(np.ones((1, c2.shape[1]))) 
        X2 = 1 / c2.dot(np.ones((1, c1.shape[1])))
        X = 1 / (X1 + X2.T - 1)

        y = self.model.predict(X.flatten().T).reshape(jm.shape)
        return np.array(jm / y)

class Old_Spectral:
    def __init__(self, n_dim=30, sampling_rate=1, distance="jaccard"):
        self.sampling_rate = sampling_rate
        #self.dim = mat.get_shape()[1]
        self.n_dim = n_dim
        self.distance = distance
        if (self.distance == "jaccard"):
            print("Use jaccard distance")
            self.compute_similarity = old_jaccard_similarity
        elif (self.distance == "cosine"):
            self.compute_similarity = cosine_similarity
        else:
            self.compute_similarity = rbf_kernel

    def fit(self, mat):
        self.sample = mat
        self.dim = mat.shape[1]
        self.coverage = mat.sum(axis=1) / self.dim
        S = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            self.normalizer = Old_JaccardNormalizer(S, self.coverage)
            S = self.normalizer.predict(S, self.coverage, self.coverage)

        np.fill_diagonal(S, 0)
        D = np.diag(1/(self.sampling_rate * S.sum(axis=1)))
        L = np.matmul(D, S)

        evals, evecs = sp.sparse.linalg.eigs(L, self.n_dim + 1, which='LR')
        ix = evals.argsort()[::-1]
        self.evals = np.real(evals[ix])
        self.evecs = np.real(evecs[:, ix])

    def predict(self, data=None):
        if data == None:
            return self.evecs
        jm = self.compute_similarity(self.sample, data)

        if (self.distance == "jaccard"):
            S_ = self.normalizer.predict(jm, self.coverage, data.sum(axis=1) / self.dim).T
        else:
            S_ = jm.T

        D_ = np.diag(1/(self.sampling_rate * S_.sum(axis=1)))
        L_ = np.matmul(D_, S_)
        evecs = (L_.dot(self.evecs)).dot(np.diag(1/self.evals))
        return evecs