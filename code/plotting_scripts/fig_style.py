### Figure styles
def get_fig_style(aspect_ratio = 12. / 9., width_scale=1.):
    return {
        'figure.figsize': (width_scale * 5.5, width_scale * 5.5 / aspect_ratio),
        'font.size':10
    }