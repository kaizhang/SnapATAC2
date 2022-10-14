from __future__ import annotations

import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet

def umap(
    adata: AnnData | AnnDataSet | np.ndarray,
    n_comps: int = 2,
    use_dims: int | list[int] | None = None,
    use_rep: str = "X_spectral",
    key_added: str = 'umap',
    random_state: int = 0,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Parameters
    ----------
    adata
        The annotated data matrix.
    n_comps
        The number of dimensions of the embedding.
    use_dims
        Use these dimensions in `use_rep`.
    use_rep
        Use the indicated representation in `.obsm`.
    key_added
        `adata.obs` key under which to add the cluster labels.
    random_state
        Random seed.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it stores UMAP embedding in
        `adata.obsm["X_`key_added`"]`.
        Otherwise, it returns the result as a numpy array.
    """
    from umap import UMAP

    if isinstance(adata, AnnData) or isinstance(adata, AnnDataSet):
        data = adata.obsm[use_rep]
    else:
        data = adata

    if use_dims is not None:
        if isinstance(use_dims, int):
            data = data[:, :use_dims]
        else:
            data = data[:, use_dims]

    umap = UMAP(
        random_state=random_state, n_components=n_comps
        ).fit_transform(data)
    if inplace:
        adata.obsm["X_" + key_added] = umap
    else:
        return umap