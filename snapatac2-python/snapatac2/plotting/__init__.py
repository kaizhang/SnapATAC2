from ._qc import tsse, scrublet
from ._scatter import umap

from typing import Optional
from snapatac2._snapatac2 import AnnData
from ._utils import render_plot

def detect(x, saturation=0.01):
    accum_gap = 0
    for i in range(1, len(x)):
        gap = x[i-1] - x[i]
        accum_gap = accum_gap + gap
        if gap < saturation * accum_gap:
            return i
    return None

def spectral_eigenvalues(
    adata: AnnData,
    n_components: Optional[int] = None,
    show: bool = True,
    interactive: bool = True,
    out_file: Optional[str] = None,
    **kwargs,
) -> None:
    import plotly.express as px
    import pandas as pd

    data = adata.uns["spectral_eigenvalue"]
    n = data.shape[0] if n_components is None else n_components
    data = data[:n]

    df = pd.DataFrame({"Component": map(str, range(1, n+1)), "Eigenvalue": data})
    fig = px.scatter(df, x = "Component", y = "Eigenvalue", template="plotly_white", **kwargs)
    fig.add_vline(x=detect(data))

    return render_plot(fig, interactive, show, out_file)