def render_plot(fig, interactive: bool, show: bool, out_file: str):
    if out_file is not None: fig.write_image(out_file)
    if show:
        if interactive:
            fig.show()
        else:
            from IPython.display import Image
            return Image(fig.to_image(format="png"))
    if not show and not out_file: return fig