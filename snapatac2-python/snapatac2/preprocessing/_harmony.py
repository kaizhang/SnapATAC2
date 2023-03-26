"""
Use harmony to integrate cells from different experiments.
"""
from __future__ import annotations

import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2._utils import is_anndata 

def harmony(
    adata: AnnData | AnnDataSet | np.ndarray,
    *,
    batch: str | list[str],
    use_rep: str = "X_spectral",
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
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
    use_rep
        The name of the field in ``adata.obsm`` where the lower dimensional
        representation is stored.
    use_dims
        Use these dimensions in `use_rep`.
    groupby
        If specified, split the data into groups and perform batch correction
        on each group separately.
    key_added
        If specified, add the result to ``adata.obsm`` with this key. Otherwise,
        it will be stored in ``adata.obsm[use_rep + "_harmony"]``.
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
    # Check if the data is in an AnnData object
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    # Use only the specified dimensions
    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]

    # Create a pandas dataframe with the batch information
    if isinstance(batch, str):
        batch = adata.obs[batch]

    if groupby is None:
        mat = _harmony(mat, batch, **kwargs)
    else:
        if isinstance(groupby, str): groupby = adata.obs[groupby]
        groups = list(set(groupby))
        for group in groups:
            group_idx = [i for i, x in enumerate(groupby) if x == group]
            mat[group_idx, :] = _harmony(mat[group_idx, :], batch[group_idx], **kwargs)

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_harmony"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat

def _harmony(data_matrix, batch_labels, **kwargs):
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")
    import pandas as pd

    meta = pd.DataFrame({'batch': np.array(batch_labels)})
    harmony_out = harmonypy.run_harmony(data_matrix, meta, 'batch', **kwargs)
    return harmony_out.Z_corr.T