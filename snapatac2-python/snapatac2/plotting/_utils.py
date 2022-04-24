def render_plot(fig, interactive: bool, show: bool, out_file: str):
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