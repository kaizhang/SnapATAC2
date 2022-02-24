from ._qc import tsse, scrublet

from typing import Optional
from anndata import AnnData
from ._utils import render_plot

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
    df = pd.DataFrame({"Component": map(str, range(n)), "Eigenvalue": data[:n]})
    fig = px.scatter(df, x = "Component", y = "Eigenvalue", template="plotly_white", **kwargs)

    return render_plot(fig, interactive, show, out_file)