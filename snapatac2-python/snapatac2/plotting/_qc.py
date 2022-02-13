import seaborn as sns
import matplotlib.pyplot as plt
import math
from anndata import AnnData
from ._util import save_img
import numpy as np

def tsse(
    adata: AnnData,
    save: bool = True,
    show: bool = True,
    outfile: str = None,
    dpi: int = 150,
    bw_adjust: float = 1.0,
    thresh: float = 0.1,
    xlim: int = 500,
) -> None:
    """
    Plot the TSS enrichment vs. log10(unique fragments) density figure.

    Parameters
    ----------
    adata
        Annotated data matrix.
    save
        Save the figure
    show
        Show the figure
    outfile
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'
    dpi
        Value of dpi for saving the figure, >= 150 is recommend
    bw_adjust
        Bandwidth, smoothing parameter, number in [0, 1]
    thresh
        Lowest iso-proportion level at which to draw a contour line, number in [0, 1]
    xlim
        The cells' unique fragments lower than it should be removed

    Returns
    -------
    
    """
    # remove the cells with less than xlim unique fragments 
    adata = adata[adata.obs["n_fragment"] >= xlim, :]
    tsse_data = adata.obs['tsse']
    n_fragment_data = adata.obs['n_fragment']
    log_nfragment_data = np.array([math.log10(item) for item in n_fragment_data])   
    # set seaborn style
    sns.set_style("white")
    x = log_nfragment_data
    y = tsse_data
    # Add thresh parameter
    sns.kdeplot(x, y, cmap="Blues", cbar=True,shade=True, bw_adjust=bw_adjust,thresh=thresh)
    plt.xlabel("log10(unique fragments)")
    plt.ylabel("TSS enrichment score")
    if show:
        plt.show()
    if save:
        save_img(outfile,dpi)