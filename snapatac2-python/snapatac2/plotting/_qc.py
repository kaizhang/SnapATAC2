import math
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    plt.xlabel("log10(unique fragments)")
    plt.ylabel("TSS enrichment score")

    
def tsse(
    adata: AnnData,
) -> None:
    """
    Plot the TSS enrichment vs. log10(unique fragments) density figure.

    Parameters
    ----------
    adata
        Annotated data matrix.
    
    Returns
    -------
    
    """
    n_fragment_data = adata.obs['n_fragment']
    log_nfragment_data = [math.log10(item) for item in n_fragment_data]
    tsse_data = adata.obs['tsse']
    x = log_nfragment_data
    y = tsse_data
    fig = plt.figure()
    using_mpl_scatter_density(fig, x, y)
    plt.show()
    