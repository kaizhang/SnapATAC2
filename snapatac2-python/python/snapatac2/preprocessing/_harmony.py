"""
Use harmony to integrate cells from different experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata


def harmony(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    *,
    batch: str | list[str],
    use_rep: str = "X_spectral",
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
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
    if isinstance(use_dims, int):
        use_dims = range(use_dims)
    mat = mat if use_dims is None else mat[:, use_dims]

    # Create a pandas dataframe with the batch information
    if isinstance(batch, str) or isinstance(batch, list):
        batch = pd.DataFrame(adata.obs[batch])

    if groupby is None:
        mat = _harmony(mat, batch, **kwargs)
    else:
        if isinstance(groupby, str):
            groupby = adata.obs[groupby]
        groups = list(set(groupby))
        group_idxs = [
            [i for i, x in enumerate(groupby) if x == group] for group in groups
        ]
        margs = [
            (mat[group_idx, :], batch.iloc[group_idx, :].copy())
            for group_idx in group_idxs
        ]

        with mp.Pool(processes=min(n_jobs, len(groups))) as pool:
            mats = pool.starmap(partial(_harmony, **kwargs), margs)
        for i, group_idx in enumerate(group_idxs):
            mat[group_idx, :] = mats[i]

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
    if data_matrix.shape[0] == 1:
        return data_matrix
    if len(batch_labels.shape) == 1:
        batch_labels = pd.DataFrame(batch_labels)
    # check if batch has >1 unique values
    for b in batch_labels.columns:
        if len(batch_labels[b].unique()) == 1:
            batch_labels = batch_labels.drop(b, axis=1)
    if batch_labels.shape[1] == 0:
        return data_matrix
    harmony_out = harmonypy.run_harmony(
        data_matrix,
        pd.DataFrame(batch_labels.values, columns=batch_labels.columns),
        batch_labels.columns.values,
        **kwargs,
    )
    return harmony_out.Z_corr.T
