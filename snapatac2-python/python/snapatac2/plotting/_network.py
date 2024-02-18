from __future__ import annotations

import numpy as np
import rustworkx as rx

from ._base import render_plot

def network_edge_stat(
    network: rx.PyDiGraph,
    **kwargs,
):
    """
    Parameters
    ----------
    network
        Network.
    kwargs        
        Additional arguments passed to :func:`~snapatac2.pl.render_plot` to
        control the final plot output. Please see :func:`~snapatac2.pl.render_plot`
        for details.
    """
    from collections import defaultdict
    import plotly.graph_objects as go

    scores = defaultdict(lambda: defaultdict(lambda: []))

    for fr, to, data in network.edge_index_map().values():
        type = "{} -> {}".format(network[fr].type, network[to].type)
        if data.cor_score is not None:
            scores["correlation"][type].append(data.cor_score)
        if data.regr_score is not None:
            scores["regression"][type].append(data.regr_score)
    
    fig = go.Figure()

    for key, vals in scores["correlation"].items():
        fig.add_trace(go.Violin(
            y=vals,
            name=key,
            box_visible=True,
            meanline_visible=True
        ))

    return render_plot(fig, **kwargs)

def network_scores(
    network: rx.PyDiGraph,
    score_name: str,
    width: float = 800,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    """
    score_name
        Name of the edge attribute
    width
        The width of the plot
    height
        The height of the plot
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
        sc = getattr(e, score_name)
        if sc is not None:
            values[i].append(sc)

    intervals, values = zip(*filter(lambda x: len(x[1]) > 0, zip(intervals, values)))
    values = [np.nanmean(v) for v in values]

    df = pd.DataFrame({
        "Distance to TSS (bp)": intervals,
        "Average score": values,
    })
    fig = px.bar(
        df, x="Distance to TSS (bp)", y="Average score", title = score_name, 
    )
    return render_plot(fig, width, height, interactive, show, out_file)