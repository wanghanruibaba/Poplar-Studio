"""Shared mechanical figure defaults; original method colors are preserved."""
def apply_figure_style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family":"sans-serif", "font.size":8,
        "axes.labelsize":8,"axes.titlesize":8,"legend.fontsize":7,
        "xtick.labelsize":6,"ytick.labelsize":6,"xtick.direction":"out",
        "ytick.direction":"out","legend.frameon":False,"savefig.dpi":300,
        "pdf.fonttype":42,"svg.fonttype":"none"})
