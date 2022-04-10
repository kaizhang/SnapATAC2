from typing import Optional
from snapatac2._snapatac2 import AnnData
import numpy as np
from ._utils import render_plot

def umap(
    adata: AnnData,
    color: str,
    size: int = 1,
    show: bool = True,
    interactive: bool = True,
    out_file: Optional[str] = None,
):
    """
    Plot UMAP embedding

    Parameters
    ----------
    adata
        Annotated data matrix.
    color
    interactive
        Whether to make interactive plot
    out_file
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'

    Returns
    -------
    
    """
    import plotly.express as px
    import pandas as pd

    embedding = adata.obsm['X_umap']
    df = pd.DataFrame({
        "UMAP-1": embedding[:, 0],
        "UMAP-2": embedding[:, 1],
        color: adata.obs[color].to_numpy(),
    })
    fig = px.scatter(df, x="UMAP-1", y="UMAP-2", color=color)
    fig.update_layout(
        template="simple_white",
    )
    return render_plot(fig, interactive, show, out_file)