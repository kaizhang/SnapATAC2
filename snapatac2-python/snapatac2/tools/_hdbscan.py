from typing import Optional, Union, Type
import pandas as pd
import numpy as np
from anndata.experimental import AnnCollection
import anndata as ad

def hdbscan(
    adata: Union[ad.AnnData, AnnCollection],
    random_state: int = 0,
    use_rep: Optional[str] = None,
    key_added: str = 'hdbscan',
) -> None:
    """
    Cluster cells into subgroups using the HDBSCAN algorithm.

    Parameters
    ----------
    adata
        The annotated data matrix.
    random_state
        Change the initialization of the optimization.
    use_rep
        Which data in `adata.obsm` to use for clustering. Default is "X_spectral".
    key_added
        `adata.obs` key under which to add the cluster labels.

    Returns
    -------
    adds fields to `adata`:
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell.
    """
    import hdbscan
    if use_rep is None: use_rep = "X_spectral"
    data = adata.obsm[use_rep]
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(data)
    groups = clusterer.labels_
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=sorted(map(str, np.unique(groups))),
    )