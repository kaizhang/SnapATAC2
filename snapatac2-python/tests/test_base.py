from snapatac2 import AnnData, AnnDataSet, read

import numpy as np
from numpy import dtype, ma
import pandas as pd
import pytest
from pathlib import Path
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_creation(tmp_path):
    # some test objects that we use below
    adata_dense = AnnData(filename = h5ad(tmp_path), X = np.array([[1, 2], [3, 4]]))
    file = adata_dense.filename
    del adata_dense
    adata_dense = read(file, mode="r+")
    adata_dense.uns['x'] = 0.2

    adata_sparse = AnnData(
        filename = h5ad(tmp_path),
        X = csr_matrix([[0, 2, 3], [0, 5, 6]]),
        obs = dict(obs_names=["s1", "s2"], anno1=["c1", "c2"]),
        var = dict(var_names=["a", "b", "c"]),
    )

    adata = AnnData(filename=h5ad(tmp_path))
    assert adata.n_obs == 0
    adata.obsm =dict(X_pca=np.array([[1, 2], [3, 4]]))
    assert adata.n_obs == 2

    AnnData(X = np.array([[1, 2], [3, 4]]), filename=h5ad(tmp_path))
    AnnData(X = np.array([[1, 2], [3, 4]]), obsm = {}, filename=h5ad(tmp_path))
    #AnnData(X = sp.eye(2), filename="data.h5ad")
    X = np.array([[1, 2, 3], [4, 5, 6]])
    adata = AnnData(
        X=X,
        obs=dict(Obs=["A", "B"]),
        var=dict(Feat=["a", "b", "c"]),
        obsm=dict(X_pca=np.array([[1, 2], [3, 4]])),
        filename=h5ad(tmp_path),
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
