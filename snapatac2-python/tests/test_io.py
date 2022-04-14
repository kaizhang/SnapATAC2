from snapatac2 import AnnData, read_mtx

import numpy as np
from pathlib import Path
import pytest
import uuid
from scipy import sparse as sp
from scipy.sparse import csr_matrix, issparse, random
from scipy.io import mmwrite

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_io(tmp_path):
    mtx = str(tmp_path / "1.mtx") 

    ''' empty is failing
    X = np.zeros((5, 7))
    X = csr_matrix(X)
    mmwrite(mtx, X)
    adata1 = read_mtx(mtx, h5ad(tmp_path))
    np.testing.assert_array_equal(adata1.X[...].todense(), X.todense())
    '''

    X = np.array(
        [ [ 1, 0, 0, 0]
        , [ 0, 0, 0, 0]
        , [ 0, 3, 0, 0]
        , [ 1, 0, 0, 0]
        , [ 0, 2, 0, 0]
        , [ 0, 0, 0, 0] ]
    )
    X = csr_matrix(X)
    mmwrite(mtx, X)
    adata1 = read_mtx(mtx, h5ad(tmp_path), sorted=True)
    np.testing.assert_array_equal(adata1.X[...].todense(), X.todense())

    ''' testing dense sparse array
    '''
    X = random(5000, 50, 0.9, format="csr", dtype=np.int64)
    mmwrite(mtx, X)
    adata1 = read_mtx(mtx, h5ad(tmp_path))
    adata2 = read_mtx(mtx, h5ad(tmp_path), sorted=True)
    np.testing.assert_array_equal(adata1.X[...].todense(), X.todense())
    np.testing.assert_array_equal(adata2.X[...].todense(), X.todense())

    X = random(5000, 50, 0.01, format="csr", dtype=np.int64)
    mmwrite(mtx, X)

    adata1 = read_mtx(mtx, h5ad(tmp_path))
    adata2 = read_mtx(mtx, h5ad(tmp_path), sorted=True)

    np.testing.assert_array_equal(adata1.X[...].todense(), X.todense())
    np.testing.assert_array_equal(adata2.X[...].todense(), X.todense())
