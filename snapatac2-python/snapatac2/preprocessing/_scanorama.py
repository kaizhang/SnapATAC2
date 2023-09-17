from __future__ import annotations

import numpy as np
import itertools

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2._utils import is_anndata 

def scanorama_integrate(
    adata: AnnData | AnnDataSet | np.adarray,
    *,
    batch: str | list[str],
    n_neighbors: int = 20,
    use_rep: str = 'X_spectral',
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    sigma: float = 15,
    approx: bool = True,
    alpha: float = 0.10,
    batch_size: int = 5000,
    inplace: bool = True,
    **kwargs,
):
    """
    Use Scanorama [Hie19]_ to integrate different experiments.

    Scanorama [Hie19]_ is an algorithm for integrating single-cell
    data from multiple experiments stored in an AnnData object. This
    function should be run after performing `tl.spectral` but before computing
    the neighbor graph, as illustrated in the example below.

    This uses the implementation of `scanorama
    <https://github.com/brianhie/scanorama>`__ [Hie19]_.

    Parameters
    ----------
    data
        Matrice or AnnData object. Matrices should be shaped like n_obs x n_vars.
    batch
        Batch labels for cells. If a string, labels will be obtained from `obs`.
    n_neighbors
        Number of mutual nearest neighbors.
    use_rep
        Use the indicated representation in `.obsm`.
    use_dims
        Use these dimensions in `use_rep`.
    groupby
        If specified, split the data into groups and perform batch correction
        on each group separately.
    key_added
        If specified, add the result to ``adata.obsm`` with this key. Otherwise,
        it will be stored in ``adata.obsm[use_rep + "_scanorama"]``.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        if `inplace=True` it updates adata with the field
        ``adata.obsm[`use_rep`_scanorama]``, containing adjusted principal components.
        Otherwise, it returns the result as a numpy array.
    
    See Also
    --------
    :func:`~snapatac2.tl.spectral`: compute spectral embedding of the data matrix.

    Example
    -------
    First, load libraries and example dataset, and preprocess.

    >>> import snapatac2 as snap
    >>> adata = snap.read(snap.datasets.pbmc5k(type='h5ad'), backed=None)
    >>> snap.pp.select_features(adata)
    >>> snap.tl.spectral(adata)

    We now arbitrarily assign a batch metadata variable to each cell
    for the sake of example, but during real usage there would already
    be a column in ``adata.obs`` giving the experiment each cell came
    from.

    >>> adata.obs['batch'] = 2218*['a'] + 2218*['b']

    Finally, run Scanorama. Afterwards, there will be a new table in
    ``adata.obsm`` containing the Scanorama embeddings.

    >>> snap.pp.scanorama_integrate(adata, batch='batch')
    >>> 'X_spectral_scanorama' in adata.obsm
    True
    """
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    # Use only the specified dimensions
    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]

    if isinstance(batch, str):
        batch = adata.obs[batch]

    if groupby is None:
        mat = _scanorama(mat, batch, n_neighbors, sigma, approx, alpha, batch_size, **kwargs)
    else:
        if isinstance(groupby, str): groupby = adata.obs[groupby]
        groups = list(set(groupby))
        for group in groups:
            group_idx = [i for i, x in enumerate(groupby) if x == group]
            mat[group_idx, :] = _scanorama(
                mat[group_idx, :], batch[group_idx], n_neighbors, sigma, approx, alpha, batch_size, **kwargs)

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_scanorama"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat

def _scanorama(data_matrix, batch_labels, knn, sigma, approx, alpha, batch_size, **kwargs):
    try:
        import scanorama
    except ImportError:
        raise ImportError("\nplease install Scanorama:\n\n\tpip install scanorama")
    import pandas as pd

    label_uniq = list(set(batch_labels))

    if len(label_uniq) > 1:
        batch_idx = []
        data_by_batch = []
        for label in label_uniq:
            idx = [i for i, x in enumerate(batch_labels) if x == label]
            batch_idx.append(idx)
            data_by_batch.append(data_matrix[idx,:])
        new_matrix = np.concatenate(scanorama.assemble(
            data_by_batch,
            knn=knn,
            sigma=sigma,
            approx=approx,
            alpha=alpha,
            ds_names=label_uniq,
            batch_size=batch_size,
            verbose=0,
            **kwargs,
        ))
        idx = list(itertools.chain.from_iterable(batch_idx))
        idx = np.argsort(idx)
        data_matrix = new_matrix[idx, :]
    return data_matrix