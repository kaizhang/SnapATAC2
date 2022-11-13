"""
Use harmony to integrate cells from different experiments.
"""
from __future__ import annotations

import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2._utils import is_anndata 

def harmony(
    adata: AnnData | AnnDataSet | np.ndarray,
    batch: str,
    use_dims: int | list[int] | None = None,
    use_rep: str = "X_spectral",
    inplace: bool = True,
    **kwargs,
) -> np.ndarray | None:
    """
    Use harmonypy to integrate different experiments.

    Harmony is an algorithm for integrating single-cell
    data from multiple experiments. This function uses the python
    port of Harmony, ``harmonypy``, to integrate single-cell data
    stored in an AnnData object. This function should be run after performing
    dimension reduction.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    batch
        The name of the column in ``adata.obs`` that differentiates
        among experiments/batches.
    use_dims
        Use these dimensions in `use_rep`.
    use_rep
        The name of the field in ``adata.obsm`` where the lower dimensional
        representation is stored.
    inplace
        Whether to store the result in the anndata object.
    kwargs
        Any additional arguments will be passed to
        ``harmonypy.run_harmony()``.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it updates adata with the field
        ``adata.obsm[`use_rep`_harmony]``, containing principal components
        adjusted by Harmony such that different experiments are integrated.
        Otherwise, it returns the result as a numpy array.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")

    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]

    harmony_out = harmonypy.run_harmony(mat, adata.obs[...].to_pandas(), batch, **kwargs)
    if inplace:
        adata.obsm[use_rep + "_harmony"] = harmony_out.Z_corr.T
    else:
        return harmony_out.Z_corr.T