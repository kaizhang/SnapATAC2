import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
import bz2
import _pickle as cPickle

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
            print("Use jaccard distance")
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
        jm = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            self.normalizer = Normalizer(jm, self.coverage)
            S = self.normalizer.predict(jm, self.coverage, self.coverage)
        else:
            S = jm

        np.fill_diagonal(S, 0)
        print("Normalization")
        self.D = np.diag(1/(self.sampling_rate * S.sum(axis=1)))
        L = np.matmul(self.D, S)

        print("Reduction")
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

class Normalizer:
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

# regress out a variable
def regress(X, y):
    model = LinearRegression().fit(X, y)
    return model.predict(X)

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
        d /= -d + s1 + s1.T
    else:
        s2 = m2.sum(axis=1)
        d = m1.dot(m2.T).todense()
        d /= -d + s1 + s2.T
    return d