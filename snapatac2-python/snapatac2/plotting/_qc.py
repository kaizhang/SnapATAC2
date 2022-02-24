from typing import Optional
from anndata import AnnData
import numpy as np
from ._utils import render_plot

def tsse(
    adata: AnnData,
    show_cells: bool = False,
    show: bool = True,
    interactive: bool = True,
    out_file: Optional[str] = None,
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
    sim_scores = adata.uns["scrublet"]["sim_doublet_score"]

    thres = adata.uns["scrublet"]["threshold"] if "threshold" in adata.uns["scrublet"] else None

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