import numpy as np
import scipy.sparse as ss
import anndata as ad

def read_as_binarized(adata: ad.AnnData) -> ss.spmatrix:
    grp = adata.file["X"]
    mtx = ss.csr_matrix(adata.shape, dtype=np.float64)
    mtx.indices = grp["indices"][...]
    mtx.indptr = grp["indptr"][...]
    mtx.data = np.ones(mtx.indices.shape, dtype=np.float64)
    return mtx