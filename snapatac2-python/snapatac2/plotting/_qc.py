import seaborn as sns
import matplotlib.pyplot as plt
import math
from anndata import AnnData
from ._util import save_img
import numpy as np

def tsse(
    adata: AnnData,
    save: bool = True,
    outpath: str = None,
    bw_adjust: float = 1.0,
    thresh: float = 0.1,
) -> None:
    """
    Plot the TSS enrichment vs. log10(unique fragments) density figure.

    Parameters
    ----------
    adata
        Annotated data matrix.
    save
        Save the figure
    outpath
        Path for saving the output image
    bw_adjust
        Bandwidth, smoothing parameter, number in [0, 1]
    thresh
        Lowest iso-proportion level at which to draw a contour line, number in [0, 1]

    Returns
    -------
    
    """
    # remove the cells with less than 500 unique fragments 
    adata = adata[adata.obs["n_fragment"] >= 500, :]
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
    plt.show()
    if save:
        save_path = outpath +'/tsse.png'
        save_img(save_path)