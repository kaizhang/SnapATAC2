from __future__ import annotations

import numpy as np

def heatmap(
    data_array: np.ndarray,
    row_names: list[str] | None = None,
    column_names: list[str] | None = None,
    cluster_columns: bool = True,
    cluster_rows: bool = True,
    colorscale = "Blues",
    linkage: str = "ward",
    width: int = 800,
    height: int = 600,
):
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import scipy.cluster.hierarchy as sch
    
    link_func = lambda x: sch.linkage(x, linkage)
    fig = go.Figure()
    
    if cluster_columns:
        dendro_upper = ff.create_dendrogram(data_array.T, linkagefun=link_func, orientation='bottom')
        upper_leaves = list(map(int, dendro_upper['layout']['xaxis']['ticktext']))
        data_array = data_array[:, upper_leaves]
        if column_names is not None:
            dendro_upper['layout']['xaxis']['ticktext'] = np.array(column_names)[upper_leaves]
        for i in range(len(dendro_upper['data'])):
            dendro_upper['data'][i]['yaxis'] = 'y2'
            fig.add_trace(dendro_upper['data'][i])
        fig['layout'] = dendro_upper['layout']

    if cluster_rows:
        dendro_side = ff.create_dendrogram(data_array, linkagefun=link_func, orientation='right')
        side_leaves = list(map(int, dendro_side['layout']['yaxis']['ticktext']))
        data_array = data_array[side_leaves, :]
        if row_names is not None:
            dendro_side['layout']['yaxis']['ticktext'] = np.array(row_names)[side_leaves]
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'
            fig.add_trace(dendro_side['data'][i])
        fig['layout']['yaxis'] = dendro_side['layout']['yaxis']

    # Create Heatmap
    heatmap = [go.Heatmap(
        z=data_array,
        colorscale=colorscale,
        colorbar={'orientation': 'h', 'title': 'log(-log(P))'}
    )]
    if cluster_columns:
        heatmap[0]['x'] = dendro_upper['layout']['xaxis']['tickvals']
    if cluster_rows:
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    for data in heatmap:
        fig.add_trace(data)
        
    fig.update_layout({ 'width': width, 'height': height,
        'showlegend':False, 'hovermode': 'closest',
    })
    
    if cluster_rows:
        fig.update_layout(
            xaxis={
                'domain': [.15, 1], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'ticks':"",
            },
            xaxis2={
                'domain': [0, .15], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': False,
                'ticks':"",
            },
        )
    if cluster_columns:
        fig.update_layout(
            yaxis={
                'domain': [0, .85], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': True,
                'side': 'right', 'ticks': ""
            },
            yaxis2={
                'domain':[.825, .975], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': False,
                'ticks':""
            },
        )
    return fig

def render_plot(
    fig: 'plotly.graph_objects.Figure',
    width: int = 600,
    height: int = 400,
    interactive: bool = True,
    show: bool = True,
    out_file: str | None = None,
) -> 'plotly.graph_objects.Figure' | None:
    fig.update_layout({
        "width": width,
        "height": height,
    })

    # save figure to file
    if out_file is not None:
        if out_file.endswith(".html"):
            fig.write_html(out_file, include_plotlyjs="cdn")
        else:
            fig.write_image(out_file)

    # show figure
    if show:
        if interactive:
            fig.show()
        else:
            from IPython.display import Image
            return Image(fig.to_image(format="png"))

    # return plot object
    if not show and not out_file: return fig