"""
Use harmony to integrate cells from different experiments.
"""

import numpy as np
from typing import Optional

from anndata import AnnData

def harmony(
    adata: AnnData,
    batch: str,
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
    use_rep
        The name of the field in ``adata.obsm`` where the lower dimensional
        representation is stored. Defaults to ``'X_spectral'``.
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

    if use_rep is None:
        use_rep = "X_spectral"

    harmony_out = harmonypy.run_harmony(adata.obsm[use_rep], adata.obs, batch, **kwargs)
    if inplace:
        adata.obsm[use_rep + "_harmony"] = harmony_out.Z_corr.T
    else:
        return harmony_out.Z_corr.T