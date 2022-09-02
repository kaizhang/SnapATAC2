from __future__ import annotations

import retworkx
import numpy as np
from ._utils import render_plot

def network_scores(
    network: retworkx.PyDiGraph,
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