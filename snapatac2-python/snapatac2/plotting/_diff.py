from __future__ import annotations

import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2.tools._misc import aggregate_X
from ._utils import render_plot

def regions(
    data: AnnData | AnnDataSet,
    groupby: str | list[str],
    peaks: dict[str, list[str]],
    width: float = 600,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    """
    Parameters
    ----------
    data
        Annotated data matrix.
    groupby
    peaks
    width
    height
    show
        Show the figure
    interactive
        Whether to make interactive plot
    out_file
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'
    """
    import polars as pl
    import plotly.graph_objects as go

    count = pl.DataFrame(aggregate_X(data, groupby = groupby, normalize="RPKM"))
    idx = data.var_ix(np.concatenate(list(peaks.values())).tolist())
    mat = np.log2(1 + count.to_numpy()[idx, :])

    trace = go.Heatmap(
        x = count.columns,
        y = np.concatenate(list(peaks.values()))[::-1],
        z = mat,
        type = 'heatmap',
        colorscale = 'Viridis',
        colorbar={ "title": "log2(1 + RPKM)" },
    )
    data = [trace]
    layout = {
        "yaxis": { "visible": False, "autorange": "reversed" },
        "xaxis": { "title": groupby },
        "width": width,
        "height": height,
    }
    fig = go.Figure(data = data, layout = layout)
    return render_plot(fig, interactive, show, out_file)