from typing import Optional
from snapatac2._snapatac2 import AnnData
import numpy as np
from ._utils import render_plot

def umap(
    adata: AnnData,
    color: str,
    use_rep: Optional[str] = None,
    marker_size: float = 2,
    width: float = 550,
    height: float = 500,
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

    use_rep = "X_umap" if use_rep is None else use_rep
    embedding = adata.obsm[use_rep] 
    if embedding.shape[1] >= 3:
        df = pd.DataFrame({
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            "UMAP-3": embedding[:, 2],
            color: adata.obs[color],
        })
        fig = px.scatter_3d(df,
            x='UMAP-1', y='UMAP-2', z='UMAP-3',
            color=color, width=width, height=height
        )
    else:
        df = pd.DataFrame({
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            color: adata.obs[color],
        })
        fig = px.scatter(df, x="UMAP-1", y="UMAP-2", color=color, width=width, height=height)
    fig.update_traces(marker_size=marker_size)

    fig.update_layout(
        template="simple_white",
        legend= {'itemsizing': 'constant'},
    )
    return render_plot(fig, interactive, show, out_file)