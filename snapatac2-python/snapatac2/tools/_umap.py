import anndata as ad
from typing import Optional

def umap(
    data: ad.AnnData,
    n_comps: int = 2,
    use_rep: Optional[str] = None,
    random_state: int = 0,
) -> None:
    """
    Parameters
    ----------
    data
        AnnData

    Returns
    -------
    None
    """
    from umap import UMAP

    if use_rep is None:
        X = data.obsm["X_spectral"]
    else:
        X = data.obsm[use_rep]
    data.obsm["X_umap"] = UMAP(random_state=random_state, n_components=n_comps).fit_transform(X)