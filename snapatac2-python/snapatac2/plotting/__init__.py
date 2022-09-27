from __future__ import annotations

import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2.tools._misc import aggregate_X
from ._base import render_plot, heatmap

__all__ = [
    'tsse', 'scrublet', 'umap', 'network_scores', 'spectral_eigenvalues',
    'regions', 'motif_enrichment',
]

def tsse(
    adata: AnnData,
    show_cells: bool = False,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
    min_fragment: int = 500,
):
    """
    Plot the TSS enrichment vs. number of fragments density figure.

    Parameters
    ----------
    adata
        Annotated data matrix.
    show_cells
        Whether to show individual cells as dots on the plot
    show
        Show the figure
    interactive
        Whether to make interactive plot
    out_file
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'
    min_fragment
        The cells' unique fragments lower than it should be removed

    Returns
    -------
    
    """
    import plotly.graph_objects as go

    selected_cells = adata.obs["n_fragment"] >= min_fragment
    x = adata.obs["n_fragment"][selected_cells]
    y = adata.obs["tsse"][selected_cells]

    log_x = np.log10(x)
    log_x_min, log_x_max = log_x.min(), log_x.max()
    x_range = log_x_max - log_x_min
    bin_x = np.linspace(log_x_min - 0.1 * x_range, log_x_max + 0.2 * x_range, 20)

    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    bin_y = np.linspace(y_min - 0.1 * y_range, y_max + 0.2 * y_range, 15)

    (z, rx, ry) = np.histogram2d(log_x,y, bins=(bin_x, bin_y))
    fig = go.Figure(data =
        go.Contour(
            z = z.T,
            x = 10**rx,
            y = ry,
            colorscale = 'Blues',
        )
    )

    if show_cells:
        fig.add_trace(go.Scatter(
                x = x,
                y = y,
                xaxis = 'x',
                yaxis = 'y',
                mode = 'markers',
                marker = dict(color = 'rgba(0,0,0,0.3)', size = 3)
        ))

    fig.update_xaxes(type="log")
    fig.update_layout(
        template="simple_white",
        xaxis_title="Number of unique fragments",
        yaxis_title="TSS enrichment score",
    )

    return render_plot(fig, interactive, show, out_file)

def scrublet(
    adata: AnnData,
    show: bool = True,
    interactive: bool = True,
    out_file: Optional[str] = None,
):
    """
    Plot doublets

    Parameters
    ----------
    adata
        Annotated data matrix.
    show
        Show the figure
    interactive
        Whether to make interactive plot
    out_file
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'

    Returns
    -------
    
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    doublet_scores = adata.obs["doublet_score"]
    sim_scores = adata.uns["scrublet_sim_doublet_score"]

    thres = adata.uns["scrublet_threshold"] if "scrublet_threshold" in adata.uns else None

    if thres is None:
        title1 = "Observed cells"
        title2 = "Simulated doublets"
    else:
        p1 = (doublet_scores >= thres).sum() / doublet_scores.size
        p2 = (sim_scores >= thres).sum() / sim_scores.size
        title1 = "Observed cells ({:.2%} doublets)".format(p1)
        title2 = "Simulated doublets ({:.2%} doublets)".format(p2)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[title1, title2])

    fig.add_trace(go.Histogram(x=doublet_scores),row=1, col=1)
    if thres is not None:
        fig.add_vline(x=thres, line_width=3, line_dash="dash", line_color="green")
        fig.add_vrect(x0=thres, x1 = doublet_scores.max(), line_width=0, fillcolor="red", opacity=0.2)

    fig.add_trace(go.Histogram(x=sim_scores), row=1, col=2)
    if thres is not None:
        fig.add_vline(x=thres, line_width=3, line_dash="dash", line_color="green")
        fig.add_vrect(x0=thres, x1 = sim_scores.max(), line_width=0, fillcolor="red", opacity=0.2)

    fig.update(layout_showlegend=False)
    return render_plot(fig, interactive, show, out_file)

def spectral_eigenvalues(
    adata: AnnData,
    n_components: int | None = None,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
    **kwargs,
) -> None:
    import plotly.express as px
    import pandas as pd

    data = adata.uns["spectral_eigenvalue"]
    n = data.shape[0] if n_components is None else n_components
    data = data[:n]

    df = pd.DataFrame({"Component": map(str, range(1, n+1)), "Eigenvalue": data})
    fig = px.scatter(df, x="Component", y="Eigenvalue", template="plotly_white", **kwargs)
    fig.add_vline(x=_detect(data))

    return render_plot(fig, interactive, show, out_file)

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

    count = pl.DataFrame(aggregate_X(data, groupby=groupby, normalize="RPKM"))
    idx = data.var_ix(np.concatenate(list(peaks.values())).tolist())
    mat = np.log2(1 + count.to_numpy()[idx, :])

    trace = go.Heatmap(
        x=count.columns,
        y=np.concatenate(list(peaks.values()))[::-1],
        z=mat,
        type='heatmap',
        colorscale='Viridis',
        colorbar={ "title": "log2(1 + RPKM)" },
    )
    data = [trace]
    layout = {
        "yaxis": { "visible": False, "autorange": "reversed" },
        "xaxis": { "title": groupby },
        "width": width,
        "height": height,
    }
    fig = go.Figure(data=data, layout=layout)
    return render_plot(fig, interactive, show, out_file)

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

def network_scores(
    network: 'retworkx.PyDiGraph',
    score_name: str,
    marker_size: float = 2,
    width: float = 800,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    """
    score_name
        Name of the edge attribute
    """
    import plotly.express as px
    import pandas as pd
    import bisect

    def human_format(num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    break_points = [100, 500, 2000, 20000, 50000, 100000, 500000]
    intervals = []
    for i in range(len(break_points)):
        if i == 0:
            intervals.append("0 - " + human_format(break_points[i]))
        else:
            intervals.append(human_format(break_points[i - 1]) + " - " + human_format(break_points[i]))
    intervals.append("> 500k")
    values = [[] for _ in range(len(intervals))]
    for e in network.edges():
        i = bisect.bisect(break_points, e.distance)
        values[i].append(getattr(e, score_name))

    intervals, values = zip(*filter(lambda x: len(x[1]) > 0, zip(intervals, values)))
    values = [np.nanmean(v) for v in values]

    df = pd.DataFrame({
        "Distance to TSS (bp)": intervals,
        "Average score": values,
    })
    fig = px.bar(
        df, x="Distance to TSS (bp)", y="Average score", title = score_name, 
        width=width, height=height,
    )
    return render_plot(fig, interactive, show, out_file)

def motif_enrichment(
    enrichment: list(str, 'pl.DataFrame'),
    min_log_fc: float = 1,
    min_fdr: float = 0.01,
    width: float = 600,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    import pandas as pd
    
    fc = np.vstack([df['log2(fold change)'] for df in enrichment.values()])
    filter1 = np.apply_along_axis(lambda x: np.any(np.abs(x) >= min_log_fc), 0, fc)
    
    fdr = np.vstack([df['adjusted p-value'] for df in enrichment.values()])
    filter2 = np.apply_along_axis(lambda x: np.any(x < min_fdr), 0, fdr)

    passed = np.logical_and(filter1, filter2)
    
    sign = np.sign(fc[:, passed])
    pvals = np.vstack([df['p-value'].to_numpy()[passed] for df in enrichment.values()])
    minval = np.min(pvals[np.nonzero(pvals)])
    pvals = np.clip(pvals, minval, None)
    pvals = sign * np.log(-np.log10(pvals))

    df = pd.DataFrame(
        pvals.T,
        columns=list(enrichment.keys()),
        index=next(iter(enrichment.values()))['motif name'].to_numpy()[passed],
    )
      
    fig = heatmap(
        df.to_numpy(),
        row_names=df.index,
        column_names=df.columns,
        colorscale='RdBu_r',
        width=width,
        height=height,
    )
    return render_plot(fig, interactive, show, out_file)

def _detect(x, saturation=0.01):
    accum_gap = 0
    for i in range(1, len(x)):
        gap = x[i-1] - x[i]
        accum_gap = accum_gap + gap
        if gap < saturation * accum_gap:
            return i
    return None