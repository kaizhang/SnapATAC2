import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from anndata import AnnData

def umap(
    adata: AnnData,
    outfile: Optional[str] = None,
    show: bool = True,
    color_key: list = ["leiden","tsse"],
    cmap: str = 'viridis',
    palette_style: str = "husl",
    figsize: tuple = (22,4),
    xy_fontsize: int = 15,
    t_fontsize: int = 20,
    dpi: int = 300,
    s: int = 15,
) -> None:
    """
    Plot the embedding result of UMAP.
    Parameters
    ----------
    adata
        Annotated data matrix.
    outfile
        Path of the output file for saving the output image, end with '.svg' or '.pdf' or '.png'
    show
        Show the figure
    color_key
        Key list for the cluster label, can choose from adata.obs: 'tsse', 'n_fragment', 'frac_dup', 'frac_mito', 'leiden'
    cmap
        Color map of the values in matplotlip library, such as "viridis", "jet" and so forth
    palette_style
        Name of palette in Seaborn library, such as "husl", "pastel", "Set2" and so forth
    figsize
        Size of the figure
    xy_fontsize
        Fontsize of x axis and y axis for the figure
    t_fontsize
        Fontsize of title for the figure    
    dpi
        Value of dpi for saving the figure, >= 150 is recommend
    s
        Size of data points

    Returns
    -------
    
    """

    x=adata.obsm["X_umap"][:,0]
    y=adata.obsm["X_umap"][:,1]   
    if s != None:
        s = s
    else:
        s = 120000 / len(x)      
    sub_num = len(color_key)   
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.figure(1)
    for i in range(sub_num):
        color_xy = list(adata.obs[color_key[i]].values)
        if type(color_xy[0]) == str:
            color_list = list(set(color_xy))
            palette = sns.color_palette(palette_style, len(color_list))
            color_label = []
            for item in color_xy:
                idx = color_list.index(item)
                color_label.append(palette[idx])
            ax = fig.add_subplot(1, sub_num, i+1) 
            ax.scatter(x, y, c = color_label, s = s)
            ax.set_xlabel('UMAP 1', fontsize = xy_fontsize)
            ax.set_ylabel('UMAP 2', fontsize = xy_fontsize)
            ax.set_title(color_key[i], fontsize = t_fontsize)          
            x_p = []
            y_p = []
            c_p = []
            for i in range(len(color_list)):
                idx = color_xy.index(color_list[i])
                x_p.append(x[idx])
                y_p.append(y[idx])
                c_p.append(color_label[idx])
            for i in range(len(x_p)):
                scatter = ax.scatter(x_p[i], y_p[i], c = np.array([c_p[i]]), s = s,label=color_list[i])    
            ax.legend(bbox_to_anchor=(1.15, 1),frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])            
        else:
            ax = fig.add_subplot(1, sub_num, i+1) 
            sp = ax.scatter(x, y, c=color_xy, cmap=cmap,s=s)
            ax.set_xlabel('UMAP 1', fontsize = xy_fontsize)
            ax.set_ylabel('UMAP 2', fontsize = xy_fontsize)
            ax.set_title(color_key[i], fontsize = t_fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sp)          
    if show:
        plt.show()
    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
        plt.close()