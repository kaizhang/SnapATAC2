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
    n_comps
    use_rep
    random_state

    Returns
    -------
    None
    """
    from umap import UMAP

    if use_rep is None: use_rep = "X_spectral"
    data.obsm["X_umap"] = UMAP(
        random_state=random_state, n_components=n_comps
        ).fit_transform(data.obsm[use_rep])
