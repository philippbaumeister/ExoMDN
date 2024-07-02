import warnings

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy import linalg

labels = {
    "core_rf": "d$_\\mathrm{Core}$",
    "core_mf": "w$_\\mathrm{Core}$",
    "atmosphere_rf": "d$_\\mathrm{Gas}$",
    "atmosphere_mf": "w$_\\mathrm{Gas}$",
    "mantle_rf": "d$_\\mathrm{Mantle}$",
    "mantle_mf": "w$_\\mathrm{Mantle}$",
    "water_rf": "d$_\\mathrm{Water}$",
    "water_mf": "w$_\\mathrm{Water}$",
    "log_d_atmosphere_core": "ln d$_\\mathrm{Gas}$/d$_\\mathrm{Core}$",
    "log_m_atmosphere_core": "ln w$_\\mathrm{Gas}$/w$_\\mathrm{Core}$",
    "log_d_mantle_core": "ln d$_\\mathrm{Mantle}$/d$_\\mathrm{Core}$",
    "log_m_mantle_core": "ln w$_\\mathrm{Mantle}$/w$_\\mathrm{Core}$",
    "log_d_water_core": "ln d$_\\mathrm{Water}$/d$_\\mathrm{Core}$",
    "log_m_water_core": "ln w$_\\mathrm{Water}$/w$_\\mathrm{Core}$"
    }


def cornerplot_logratios(data, columns, data_components=None, height=2.5,
                         hexbin_kws=None,
                         hist_kws=None,
                         scatter_kws=None,
                         mixture_kws=None):
    if mixture_kws is None:
        mixture_kws = {}
    if scatter_kws is None:
        scatter_kws = {}
    if hexbin_kws is None:
        hexbin_kws = {}
    if hist_kws is None:
        hist_kws = {}
    _mixture_kws = dict(s=30, marker="+", color="k", lw=1.2, cmap="viridis")
    _hist_kws = dict(bins=30, kde=False, color="C0", element="step", lw=3, fill=False)
    _hexbin_kws = dict(gridsize=25, linewidths=0, mincnt=10, cmap="Blues")
    _scatter_kws = dict(s=5, color="0.7", marker=".")

    _mixture_kws.update(mixture_kws)
    _hist_kws.update(hist_kws)
    _scatter_kws.update(scatter_kws)
    _hexbin_kws.update(hexbin_kws)

    g = sns.PairGrid(data=data, vars=columns, height=height, diag_sharey=False)
    g.map_lower(plt.hexbin, **_hexbin_kws)
    g.map_diag(sns.histplot, **_hist_kws)
    g.map_upper(plt.scatter, **_scatter_kws)

    if data_components is not None:
        len_mixture = len(data_components)
        if len_mixture > 200:
            warnings.warn(f"Not showing mixture compnents because number of mixture components is too large (n"
                          f"={len_mixture})")
        else:
            g.data = data_components
            g.map_upper(confidence_ellipse, data=data_components, **_mixture_kws)

            norm = mpl.colors.Normalize(vmin=0, vmax=data_components["weight"].max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=_mixture_kws["cmap"])
            cbar_ax = g.fig.add_axes([0.975, 0.65, 0.03, 0.3])
            plt.colorbar(cmap, cax=cbar_ax, label="Mixture weights")

    g = relabel_axes(g)
    return g


def cornerplot(data, columns, height=2, quantiles=True, hist_kws=None, hexbin_kws=None, line_kws=None, **kwargs):
    if line_kws is None:
        line_kws = {}
    if hexbin_kws is None:
        hexbin_kws = {}
    if hist_kws is None:
        hist_kws = {}

    _line_kws = dict(color="C0", alpha=0.8)
    _hexbin_kws = dict(gridsize=25, extent=(0, 1, 0, 1), linewidths=0, mincnt=10, cmap="Blues")
    _hist_kws = dict(bins=np.linspace(0, 1, 30), kde=False, color=mpl.rcParams["text.color"], element="step",
                     fill=False, lw=3)

    _line_kws.update(line_kws)
    _hexbin_kws.update(hexbin_kws)
    _hist_kws.update(hist_kws)

    def one_line(*args, **kwargs):
        plt.plot(np.linspace(0, 1, 10), np.linspace(1, 0, 10), ls="--", **kwargs)

    def plot_quantiles(*args, **kwargs):
        df = args[0]
        color = kwargs.get("c", "C0")
        desc = df.describe(percentiles=[0.05, 0.5, 0.95])
        plt.axvline(x=desc["5%"], ls=":", c=color, lw=2.5)
        plt.axvline(x=desc["50%"], ls="--", c=color, lw=3)
        plt.axvline(x=desc["95%"], ls=":", c=color, lw=2.5)
        high = desc["95%"] - desc["50%"]
        low = desc["50%"] - desc["5%"]

        label = labels.get(df.name, df.name)
        title = label + f"={desc['50%']:.2f}" + "$_{" + f"-{low:.2f}" + "}" + "^{" + f"+{high:.2f}" + "}$"
        plt.title(title, ha="center", va="bottom", fontsize=mpl.rcParams["axes.labelsize"])

    g = sns.PairGrid(data=data, vars=columns, height=height, diag_sharey=False, corner=True, **kwargs)
    g.map_diag(sns.histplot, **_hist_kws)
    if quantiles:
        g.map_diag(plot_quantiles, **_line_kws)
    g.map_lower(plt.hexbin, **_hexbin_kws)
    g.map_offdiag(one_line, **_line_kws)
    g.set(xlim=(0, 1), ylim=(0, 1))
    for ax in g.axes.ravel():
        if ax is None:
            continue
        # ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    g = relabel_axes(g)
    return g


def confidence_ellipse(x, y, **kwargs):
    df = kwargs.pop("data")
    cmap = kwargs.pop("cmap", "viridis")
    max_alpha = kwargs.pop("max_alpha", 1)
    min_alpha = kwargs.pop("min_alpha", 0)

    norm = mpl.colors.Normalize(vmin=0, vmax=df["weight"].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    min_p, max_p = (df["weight"].min(), df["weight"].max())

    ax = plt.gca()
    ax.scatter(x, y, **kwargs, zorder=10)
    for i, row in df.iterrows():
        covar = np.diag(row[[f"var_{x.name}", f"var_{y.name}"]])
        mean = row[[x.name, y.name]]
        prob = row["weight"]

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / (u[0] + 1e-12))
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=cmap.to_rgba(prob), zorder=prob + 1)
        ell.set_clip_box(ax.bbox)
        alpha = min(1, (prob - min_p) / (max_p - min_p) * (max_alpha - min_alpha) + min_alpha)
        ell.set_alpha(alpha)  # ell_kws.get("alpha", 0.5))
        ax.add_artist(ell)


def relabel_axes(figure):
    for ax in figure.axes.ravel():
        if ax is None:
            continue
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel in labels.keys():
            ax.set_xlabel(labels[xlabel])
        if ylabel in labels.keys():
            ax.set_ylabel(labels[ylabel])
    return figure
