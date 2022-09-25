from __future__ import annotations

import numpy as np
import logging

from snapatac2._snapatac2 import AnnData, AnnDataSet

def transfer_labels(
    adata: AnnData | AnnDataSet,
    use_rep: str | np.ndarray,
    labels: str | list[str],
    n_neighbors: int = 15,
    metric: str = "cosine",
    inplace: bool = True,
):
    """
    Transfer labels.

    Parameters
    ----------
    adata
        AnnData or AnnDataSet object.
    use_rep
    labels
        Cell labels. Labels with `None` values will be predicted.
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    labs = adata.obs[labels].to_numpy(copy=True) if isinstance(labels, str) else np.array(labels, copy=True)
    if (labs != None).all():
        logging.warning("Nothing to do, because every cell has a label.")
        return None
    
    embedding = adata.obsm[use_rep] if isinstance(use_rep, str) else use_rep

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    X = embedding[labs != None, :]
    y = labs[labs != None]
    model.fit(X, y)
    
    labs[np.where(labs == None)] = model.predict(embedding[labs == None, :])
    
    if inplace and isinstance(labels, str):
        adata.obs[labels] = labs
    else:
        return labs