from itertools import product

import numpy as np
from numpy import ma
import pandas as pd
import pytest
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse

from snapatac2.anndata import AnnData

def h5ad():
    return str(uuid.uuid4()) + ".h5ad"

# some test objects that we use below
adata_dense = AnnData(filename = h5ad(), X = np.array([[1, 2], [3, 4]]))
adata_sparse = AnnData(
    filename = h5ad(),
    X = csr_matrix([[0, 2, 3], [0, 5, 6]]),
    obs = dict(obs_names=["s1", "s2"], anno1=["c1", "c2"]),
    var = dict(var_names=["a", "b", "c"]),
)

def test_creation():
    adata = AnnData(filename=h5ad())
    assert adata.n_obs == 0
    adata.obsm =dict(X_pca=np.array([[1, 2], [3, 4]]))
    assert adata.n_obs == 2

    AnnData(X = np.array([[1, 2], [3, 4]]), filename=h5ad())
    AnnData(X = np.array([[1, 2], [3, 4]]), obsm = {}, filename=h5ad())
    #AnnData(X = sp.eye(2), filename="data.h5ad")
    X = np.array([[1, 2, 3], [4, 5, 6]])
    adata = AnnData(
        X=X,
        obs=dict(Obs=["A", "B"]),
        var=dict(Feat=["a", "b", "c"]),
        obsm=dict(X_pca=np.array([[1, 2], [3, 4]])),
        filename=h5ad(),
    )

    adata.var["Count"] = [1,2,3]
    assert list(adata.var["Count"]) == [1,2,3]
    '''
    with pytest.raises(ValueError):
        AnnData(X = np.array([[1, 2], [3, 4]]), obsm = dict(TooLong=[1, 2, 3, 4]))
    '''

    # init with empty data matrix
    #shape = (3, 5)
    #adata = AnnData(None, uns=dict(test=np.array((3, 3))), shape=shape)
    #assert adata.X is None
    #assert adata.shape == shape
    #assert "test" in adata.uns