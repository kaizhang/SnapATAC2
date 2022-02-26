"""
Use harmony to integrate cells from different experiments.
"""

import numpy as np
from typing import Optional, List, Union

from anndata import AnnData

def harmony(
    adata: AnnData,
    batch: str,
    use_dims: Optional[Union[int, List[int]]] = None,
    use_rep: Optional[str] = None,
    inplace: bool = True,
    **kwargs,
) -> Optional[np.ndarray]:
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
        The annotated data matrix.
    batch
        The name of the column in ``adata.obs`` that differentiates
        among experiments/batches.
    use_dims
        Use these dimensions in `use_rep`.
    use_rep
        The name of the field in ``adata.obsm`` where the lower dimensional
        representation is stored. Defaults to ``'X_spectral'``.
    inplace
        Whether to store the result in the anndata object.
    kwargs
        Any additional arguments will be passed to
        ``harmonypy.run_harmony()``.

    Returns
    -------
    if `inplace=True` it updates adata with the field
    ``adata.obsm[`use_rep`_harmony]``, containing principal components
    adjusted by Harmony such that different experiments are integrated.
    Otherwise, it returns the result as a numpy array.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")

    if use_rep is None: use_rep = "X_spectral"
    mat = adata.obsm[use_rep] if isinstance(adata, AnnData) else adata
    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]

    harmony_out = harmonypy.run_harmony(mat, adata.obs, batch, **kwargs)
    if inplace:
        adata.obsm[use_rep + "_harmony"] = harmony_out.Z_corr.T
    else:
        return harmony_out.Z_corr.T