import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import bz2
import _pickle as cPickle
import gc

from .snapatac2 import jm_regress

def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data

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
    """Compute pair-wise jaccard index

    Args:
        mat1: n1 x m
        mat2: n2 x m
    
    Returns:
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

    Args:
        mat1: n1 x m
        mat2: n2 x m
    
    Returns:
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

