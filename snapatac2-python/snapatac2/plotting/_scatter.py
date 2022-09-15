from __future__ import annotations

from snapatac2._snapatac2 import AnnData
import numpy as np
from ._utils import render_plot

def umap(
    adata: AnnData,
    color: str | np.ndarray,
    use_rep: str | None = None,
    marker_size: float = 2,
    marker_opacity: float = 0.5,
    width: float = 550,
    height: float = 500,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    """
    Plot UMAP embedding

    Parameters
    ----------
    adata
        Annotated data matrix.
    color
        If the input is a string, it will be used the key to retrieve values from
        `obs`.
    use_rep
    marker_size
    marker_opacity
    width
    height
    interactive
        Whether to make interactive plot
    out_file
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'

    Returns
    -------
    
    """
    import plotly.express as px
    from natsort import index_natsorted
    import pandas as pd

    use_rep = "X_umap" if use_rep is None else use_rep
    embedding = adata.obsm[use_rep] 

    if isinstance(color, str):
        groups = adata.obs[color]
    else:
        groups = color
        color = "color"

    idx = index_natsorted(groups)
    embedding = embedding[idx, :]
    groups = groups[idx]

    if embedding.shape[1] >= 3:
        df = pd.DataFrame({
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            "UMAP-3": embedding[:, 2],
            color: groups,
        })
        fig = px.scatter_3d(df,
            x='UMAP-1', y='UMAP-2', z='UMAP-3',
            color=color, width=width, height=height,
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
    else:
        df = pd.DataFrame({
            "UMAP-1": embedding[:, 0],
            "UMAP-2": embedding[:, 1],
            color: groups,
        })
        fig = px.scatter(
            df, x="UMAP-1", y="UMAP-2", color=color, width=width, height=height,
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
    fig.update_traces(
        marker_size=marker_size,
        marker={"opacity": marker_opacity},
    )

    fig.update_layout(
        template="simple_white",
        legend= {'itemsizing': 'constant'},
    )
    return render_plot(fig, interactive, show, out_file)