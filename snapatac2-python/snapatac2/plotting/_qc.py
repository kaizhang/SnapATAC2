import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData

def tsse(
    adata: AnnData,
    show: bool = True,
    outfile: Optional[str] = None,
    dpi: int = 300,
    bw_adjust: float = 1.0,
    thresh: float = 0.1,
    min_fragment: int = 500,
) -> None:
    """
    Plot the TSS enrichment vs. number of fragments density figure.

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
    min_fragment
        The cells' unique fragments lower than it should be removed

    Returns
    -------
    
    """
    import seaborn as sns
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

    selected_cells = adata.obs["n_fragment"] >= min_fragment
    sns.kdeplot(
        x = adata.obs['n_fragment'][selected_cells],
        y = adata.obs['tsse'][selected_cells],
        cmap = "Blues",
        cbar = True,
        shade = True,
        bw_adjust = bw_adjust,
        thresh = thresh,
        log_scale = (10, False)
    )
    plt.xlabel("Number of fragments")
    plt.ylabel("TSS enrichment score")
    if show:
        plt.show()
    if outfile:
        plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
        plt.close()