"""The Hertzsprung-Russell diagram itself.

`plot_hr_diagram` draws an HR diagram — luminosity in solar units against
surface temperature, spectral type and true stellar colour — in which every
component can be toggled on or off, so the diagram can be built up step by
step for teaching. `save_hr_diagram_suite` writes a set of high-resolution
PNGs that together illustrate every feature without overcrowding any single
plot.

Matplotlib style (Arial, tick directions, ...) is applied through an
`rc_context`, so importing or using this module never alters global
matplotlib state.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import matplotlib
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, NullLocator
from numpy.typing import ArrayLike

from .blackbody import blackbody_rgb
from .calibrations import (MS_TEFF, SUN_TEFF, ms_lifetime_gyr,
                           ms_lum_from_teff, teff_from_ms_mass, wd_band_lum)
from .catalogs import load_brightest_stars, load_gaia_sample, load_nearest_stars
from .overlays import (FAMOUS_STARS, LUMINOSITY_CLASS_BANDS,
                       LUMINOSITY_CLASS_PATCHES, MS_ANNOTATIONS,
                       SPECTRAL_CLASSES, STAR_GROUP_CLOUDS, format_big_number,
                       format_lifetime, format_lum_tick, smooth_log_spine)

# Applied via rc_context inside plot_hr_diagram; mathtext supplies the solar
# symbol, which Arial lacks.
MPL_STYLE: dict[str, object] = {
    "font.family": "Arial",
    "font.size": 15,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.1,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
}


def marker_sizes(teff: ArrayLike, lum: ArrayLike) -> np.ndarray:
    """Marker area (pt^2) for star dots, tracking each star's radius.

    The radius comes from the Stefan-Boltzmann law,
    R/Rsun = sqrt(L/Lsun) (Tsun/T)^2, so symbol size represents how large the
    star actually is rather than which sample it belongs to. Tune the size
    mapping here; it applies to every star drawn, including the Sun.
    """
    radius = np.sqrt(lum) * (SUN_TEFF / np.asarray(teff, dtype=float)) ** 2
    return np.clip(np.sqrt(1000 * radius) + 1.5, 1.5, 1000.0)


def _auto_point_label(index: int) -> str:
    """Default label for the index-th user point: A, B, ... Z, AA, AB, ..."""
    label = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def plot_hr_diagram(
        temp_lim: tuple[float, float] = (2500.0, 45000.0),
        lum_lim: tuple[float, float] = (1e-5, 1e6),
        show_temp_labels: bool = True,
        show_spectral_labels: bool = True,
        show_color_strip: bool = True,
        show_main_sequence: bool = True,
        show_ms_annotations: bool = True,
        show_radius_lines: bool = True,
        show_sun: bool = True,
        show_nearest: bool = True,
        show_brightest: bool = True,
        show_gaia_sample: bool = True,
        show_famous_stars: bool = False,
        show_luminosity_classes: bool = False,
        show_star_groups: bool = False,
        points: Sequence[tuple[float, float]
                         | tuple[float, float, str | None]] | None = None,
        dark: bool = True,
        figsize: tuple[float, float] = (10.0, 10.5),
        dpi: int = 200,
        savepath: str | None = None,
        show: bool = True) -> None:
    """Draw a Hertzsprung-Russell diagram; every feature can be toggled.

    Parameters
    ----------
    temp_lim
        Surface-temperature axis limits in K, (coolest, hottest); the axis is
        drawn with temperature increasing to the left.
    lum_lim
        Luminosity axis limits in solar luminosities.
    show_temp_labels
        Numeric temperature tick labels (K).
    show_spectral_labels
        O B A F G K M spectral classes above the colour strip (or along the
        top of the graph when the strip is off).
    show_color_strip
        True-colour blackbody bar under the temperature axis.
    show_main_sequence
        The main-sequence line from the dwarf calibration.
    show_ms_annotations
        Masses and lifetimes marked along the main sequence.
    show_radius_lines
        Dashed lines of constant radius (0.001 ... 1000 Rsun).
    show_sun
        The Sun, drawn on top of everything at its radius-based size.
    show_nearest
        The ~1000 nearest stars (Gaia DR3).
    show_brightest
        The ~100 brightest stars in the night sky (Hipparcos).
    show_gaia_sample
        The ~84 000-star Gaia DR3 sample within 200 pc.
    show_famous_stars
        Labelled well-known stars (Betelgeuse, Sirius, Proxima, ...).
    show_luminosity_classes
        Morgan-Keenan luminosity-class regions, drawn over the stars.
    show_star_groups
        Soft gradient clouds for notable groups of stars (main sequence,
        giants, supergiants, red dwarfs, white dwarfs).
    points
        User-specified points, drawn in bright red on top of every other
        component. Each entry is ``(teff, lum)`` — labelled automatically
        A, B, C, ... by position in the sequence — or ``(teff, lum, label)``
        with a custom label string, or ``(teff, lum, None)`` for no label.
    dark
        White-on-black (True) or black-on-white (False).
    figsize
        Figure size in inches.
    dpi
        Resolution of the saved PNG (the on-screen figure uses 110 dpi).
    savepath
        If given, save the figure there (e.g. ``"hr_diagram.png"``).
    show
        Display the figure (True) or close it after saving (False).

    The star catalogues are downloaded on first use and cached in
    ``hr_diagram_data/`` under the current working directory; see
    :mod:`hr_diagram.catalogs`.
    """
    with matplotlib.rc_context(MPL_STYLE):
        _draw(temp_lim, lum_lim, show_temp_labels, show_spectral_labels,
              show_color_strip, show_main_sequence, show_ms_annotations,
              show_radius_lines, show_sun, show_nearest, show_brightest,
              show_gaia_sample, show_famous_stars, show_luminosity_classes,
              show_star_groups, points, dark, figsize, dpi, savepath, show)


def _draw(temp_lim, lum_lim, show_temp_labels, show_spectral_labels,
          show_color_strip, show_main_sequence, show_ms_annotations,
          show_radius_lines, show_sun, show_nearest, show_brightest,
          show_gaia_sample, show_famous_stars, show_luminosity_classes,
          show_star_groups, points, dark, figsize, dpi, savepath, show) -> None:
    bg = "#000000" if dark else "#ffffff"
    ink = "#f2f2f2" if dark else "#1a1a1a"
    faint = "#9a9a9a" if dark else "#6e6e6e"
    ms_ink = "#a9c8ff" if dark else "#33639e"   # mass/lifetime labels: soft blue
    radius_col = "#9ac8a8" if dark else "#4f7f60"   # radius lines: subtle green
    grid_col = "#3a3a3a" if dark else "#d9d9d9"
    # halo behind text so labels stay readable on top of dense star fields
    halo = [patheffects.withStroke(linewidth=3, foreground=bg)]

    fig = plt.figure(figsize=figsize, dpi=110, facecolor=bg)
    ax = fig.add_axes([0.115, 0.16, 0.845, 0.76])
    ax.set_facecolor(bg)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(temp_lim[1], temp_lim[0])     # temperature increases to the left
    ax.set_ylim(lum_lim)

    for spine in ax.spines.values():
        spine.set_color(faint)
    ax.tick_params(colors=faint, labelcolor=ink, which="both")

    # --- temperature (bottom) axis -------------------------------------------
    tick_pool = [2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000,
                 30000, 40000, 60000, 100000]
    xticks = [t for t in tick_pool if temp_lim[0] <= t <= temp_lim[1]]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([format_big_number(t) for t in xticks], fontsize=14.5)

    # --- luminosity (left) axis ----------------------------------------------
    e_lo = math.ceil(math.log10(lum_lim[0]) - 1e-9)
    e_hi = math.floor(math.log10(lum_lim[1]) + 1e-9)
    yticks = [10.0 ** e for e in range(e_lo, e_hi + 1)]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([format_lum_tick(e) for e in range(e_lo, e_hi + 1)],
                       fontsize=14.5)
    ax.set_ylabel("Luminosity  ($\\mathbf{L_\\odot}$)", fontsize=20,
                  fontweight="bold", color=ink)
    ax.grid(axis="y", color=grid_col, linewidth=0.6, alpha=0.55, zorder=0)

    # --- true-colour strip under the temperature axis ------------------------
    strip = None
    if show_color_strip:
        strip = fig.add_axes([0.115, 0.10, 0.845, 0.022])
        strip.set_facecolor(bg)
        strip.set_xscale("log")
        strip.set_xlim(temp_lim[1], temp_lim[0])
        strip.set_ylim(0, 1)
        edges = np.geomspace(temp_lim[0], temp_lim[1], 513)
        mids = np.sqrt(edges[:-1] * edges[1:])
        cmap = ListedColormap(blackbody_rgb(mids))
        strip.pcolormesh(edges, np.array([0.0, 1.0]),
                         np.arange(512)[None, :], cmap=cmap,
                         vmin=0, vmax=511, rasterized=True)
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_color(faint)
        strip.xaxis.set_major_locator(FixedLocator(xticks))
        strip.xaxis.set_minor_locator(NullLocator())
        if show_temp_labels:
            strip.set_xticklabels([format_big_number(t) for t in xticks],
                                  fontsize=14.5)
            strip.tick_params(axis="x", colors=faint, labelcolor=ink, length=4)
        else:
            strip.set_xticklabels([])
            strip.tick_params(axis="x", colors=faint, length=4)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        label_ax = strip           # the temperature numbers sit below the strip
    else:
        label_ax = ax
        if not show_temp_labels:
            ax.tick_params(axis="x", labelbottom=False)

    if show_temp_labels:
        label_ax.set_xlabel("Surface temperature  (K)", fontsize=20,
                            fontweight="bold", color=ink, labelpad=10)

    # --- spectral classes: above the colour strip, or the graph without one --
    if show_spectral_labels:
        host = strip if strip is not None else ax
        sec = host.secondary_xaxis("top")
        boundaries = sorted({b for _, lo, hi in SPECTRAL_CLASSES for b in (lo, hi)
                             if temp_lim[0] < b < temp_lim[1]})
        sec.xaxis.set_major_locator(FixedLocator(boundaries))
        sec.set_xticklabels([""] * len(boundaries))
        centers, letters = [], []
        for letter, lo, hi in SPECTRAL_CLASSES:
            lo_c, hi_c = max(lo, temp_lim[0]), min(hi, temp_lim[1])
            if hi_c / lo_c > 1.06:
                centers.append(math.sqrt(lo_c * hi_c))
                letters.append(letter)
        sec.xaxis.set_minor_locator(FixedLocator(centers))
        sec.set_xticklabels(letters, minor=True, fontsize=17, color=ink)
        sec.tick_params(which="major", colors=faint, length=5)
        sec.tick_params(which="minor", length=0, labelcolor=ink, pad=3)
        sec.spines["top"].set_color(faint)

    # colours used for star markers; slightly darkened on white so pale
    # yellow-white stars stay visible
    def marker_colors(teff):
        rgb = blackbody_rgb(teff)
        return rgb if dark else rgb * 0.80

    # --- MK luminosity-class regions (off by default) --------------------------
    # Drawn ABOVE the star samples (zorder 6.5): translucent fills with dashed
    # outlines, so the regions stay visible on top of the dense Gaia cloud.
    if show_luminosity_classes:
        band_col = (0.55, 0.65, 0.95)
        fill_kw = dict(color=band_col, alpha=0.16 if dark else 0.13,
                       linewidth=0, zorder=6.5)
        edge_kw = dict(color=band_col, alpha=0.9, lw=1.2, ls=(0, (5, 4)),
                       zorder=6.5)
        t_full = [temp_lim[0], temp_lim[1]]
        for label, l_lo, l_hi in LUMINOSITY_CLASS_BANDS:
            ax.fill_between(t_full, l_lo, l_hi, **fill_kw)
            for l_edge in (l_lo, l_hi):
                ax.plot(t_full, [l_edge, l_edge], **edge_kw)
            x_lab = temp_lim[1] / 1.25
            ax.text(x_lab, math.sqrt(l_lo * l_hi), label, fontsize=13,
                    color=ink, alpha=0.9, va="center", zorder=8,
                    path_effects=halo)
        for label, poly, anchor in LUMINOSITY_CLASS_PATCHES:
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs[:-1], ys[:-1], **fill_kw)
            ax.plot(xs, ys, **edge_kw)
            ax.text(anchor[0], anchor[1], label, fontsize=13, color=ink,
                    alpha=0.9, ha="center", va="center", zorder=8,
                    path_effects=halo)
        # class V follows the main sequence; the band is a factor 2.2 wide,
        # roughly the scatter of the observed dwarf sequence
        t_dense = np.geomspace(max(temp_lim[0], MS_TEFF.min()),
                               min(temp_lim[1], MS_TEFF.max()), 200)
        l_dense = ms_lum_from_teff(t_dense)
        ax.fill_between(t_dense, l_dense / 2.2, l_dense * 2.2, **fill_kw)
        ax.plot(t_dense, l_dense / 2.2, **edge_kw)
        ax.plot(t_dense, l_dense * 2.2, **edge_kw)
        ax.text(11500, 280, "V — main sequence", fontsize=13, color=ink,
                alpha=0.9, ha="center", va="center", rotation=-32, zorder=8,
                path_effects=halo)
        t_wd = np.geomspace(4500, min(40000, temp_lim[1]), 100)
        wd_lo, wd_hi = wd_band_lum(t_wd, spread=6.0)
        ax.fill_between(t_wd, wd_lo, wd_hi, **fill_kw)
        ax.plot(t_wd, wd_lo, **edge_kw)
        ax.plot(t_wd, wd_hi, **edge_kw)
        ax.text(11500, 2.5e-3, "white dwarfs", fontsize=13, color=ink,
                alpha=0.9, ha="center", va="center", rotation=-24, zorder=8,
                path_effects=halo)

    # --- soft star-group clouds (off by default) -------------------------------
    # Each region is a Gaussian gradient field on a log-log grid: opacity
    # falls off continuously from the spine, so the clouds bleed into their
    # surroundings and into one another with no edges at all.
    if show_star_groups:
        core_a = 0.30 if dark else 0.22
        n_cell = 360
        gt_edges = np.geomspace(temp_lim[0], temp_lim[1], n_cell + 1)
        gl_edges = np.geomspace(lum_lim[0], lum_lim[1], n_cell + 1)
        gt = np.log10(np.sqrt(gt_edges[:-1] * gt_edges[1:]))   # centres, dex
        gl = np.log10(np.sqrt(gl_edges[:-1] * gl_edges[1:]))
        GT, GL = np.meshgrid(gt, gl)

        def cloud_mesh(d, col):
            rgba = np.empty(d.shape + (4,))
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = col
            rgba[..., 3] = core_a * np.exp(-0.5 * d ** 2)
            ax.pcolormesh(gt_edges, gl_edges, rgba, rasterized=True, zorder=1)

        def cloud_label(x, y, label, col, rotation=0):
            lab_col = tuple(0.35 + 0.65 * c for c in col) if dark \
                else tuple(0.55 * c for c in col)
            ax.text(x, y, label, fontsize=13.5, color=lab_col, ha="center",
                    va="center", rotation=rotation, rotation_mode="anchor",
                    zorder=8, path_effects=halo)

        # branch clouds: distance from a spine running along luminosity,
        # width w_dex in log Teff, rounded caps in log L
        for label, spine_pts, w_dex, anchor, rot, col in STAR_GROUP_CLOUDS:
            lt_s, ll_s = smooth_log_spine(spine_pts)
            order = np.argsort(ll_s)
            tc = np.interp(GL, ll_s[order], lt_s[order])
            d_t = (GT - tc) / w_dex
            d_l = (np.clip(ll_s.min() - GL, 0, None)
                   + np.clip(GL - ll_s.max(), 0, None)) / 0.30
            cloud_mesh(np.hypot(d_t, d_l), col)
            cloud_label(anchor[0], anchor[1], label, col, rot)

        # band clouds: distance from a central curve L(T), half-width h_dex
        # in log L, rounded caps where an end stops inside the plotted range
        def band_cloud(t_lo, t_hi, l_of_t, h_dex, col, cap_dex=0.15):
            lc = np.log10(l_of_t(10.0 ** gt))
            d_l = (GL - lc[None, :]) / h_dex
            lo, hi = math.log10(t_lo), math.log10(t_hi)
            d_t = (np.clip(lo - GT, 0, None)
                   + np.clip(GT - hi, 0, None)) / cap_dex
            cloud_mesh(np.hypot(d_l, d_t), col)

        # main sequence and red dwarfs follow the dwarf calibration and
        # overlap around 4000 K, reading as one continuous sequence
        ms_col = (1.00, 0.85, 0.45)
        band_cloud(3900, MS_TEFF.max(), ms_lum_from_teff, 0.33, ms_col)
        cloud_label(8300, 15, "main sequence", ms_col, rotation=-32)
        rd_col = (1.00, 0.50, 0.30)
        band_cloud(MS_TEFF.min(), 4100, ms_lum_from_teff, 0.42, rd_col)
        cloud_label(3150, 0.014, "red dwarfs", rd_col)

        # white dwarfs: band of roughly constant radius
        wd_col = (0.75, 0.85, 1.00)
        band_cloud(4500, min(40000.0, temp_lim[1]),
                   lambda t: wd_band_lum(t)[0], 0.55, wd_col)
        cloud_label(11500, 2.5e-3, "white dwarfs", wd_col, rotation=-24)

    # --- lines of constant radius --------------------------------------------
    if show_radius_lines:
        for r_sun in 10.0 ** np.arange(-3, 4):
            # L/Lsun = (R/Rsun)^2 (T/Tsun)^4  (Stefan-Boltzmann law)
            t_line = np.geomspace(temp_lim[0], temp_lim[1], 300)
            l_line = r_sun ** 2 * (t_line / SUN_TEFF) ** 4
            inside = (l_line >= lum_lim[0]) & (l_line <= lum_lim[1])
            if not inside.any():
                continue
            ax.plot(t_line[inside], l_line[inside], ls=(0, (6, 6)), lw=0.9,
                    color=radius_col, alpha=0.65, zorder=2)
            # label near the hot (upper-left) end of the visible segment
            t_at_top = SUN_TEFF * (lum_lim[1] / r_sun ** 2) ** 0.25
            if t_at_top < temp_lim[1]:          # line exits through the top
                t_lab = t_at_top / 1.45
            else:                               # line exits through the left edge
                t_lab = temp_lim[1] / 1.15
            t_visible = t_line[inside]
            if not (t_visible.min() * 1.05 < t_lab < t_visible.max() * 0.98):
                continue
            l_lab = r_sun ** 2 * (t_lab / SUN_TEFF) ** 4
            p1 = ax.transData.transform(
                (t_lab * 1.02, r_sun ** 2 * (t_lab * 1.02 / SUN_TEFF) ** 4))
            p2 = ax.transData.transform(
                (t_lab / 1.02, r_sun ** 2 * (t_lab / 1.02 / SUN_TEFF) ** 4))
            rot = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            # the 10 Rsun line runs along the upper main sequence, so its
            # label goes on the underside of the line to stay clear of it
            below = r_sun == 10
            ax.annotate(f"{r_sun:g} $R_\\odot$", (t_lab, l_lab),
                        xytext=(0, -7 if below else 7),
                        textcoords="offset points",
                        rotation=rot, rotation_mode="anchor",
                        ha="center", va="top" if below else "bottom",
                        fontsize=12.5, color=radius_col, zorder=7,
                        path_effects=halo)

    # --- star samples ---------------------------------------------------------
    if show_gaia_sample:
        data = load_gaia_sample()
        if data is not None:
            teff, lum = data
            ax.scatter(teff, lum, s=marker_sizes(teff, lum),
                       c=marker_colors(teff),
                       alpha=0.35, linewidths=0, zorder=3, rasterized=True)
    if show_nearest:
        data = load_nearest_stars()
        if data is not None:
            teff, lum = data
            ax.scatter(teff, lum, s=marker_sizes(teff, lum),
                       c=marker_colors(teff), alpha=0.9,
                       linewidths=0.4, edgecolors=bg, zorder=4)
    if show_brightest:
        data = load_brightest_stars()
        if data is not None:
            teff, lum = data
            ax.scatter(teff, lum, s=marker_sizes(teff, lum),
                       c=marker_colors(teff),
                       alpha=0.95, linewidths=0.6,
                       edgecolors=faint, zorder=6)

    # --- main sequence --------------------------------------------------------
    if show_main_sequence:
        t_dense = np.geomspace(max(temp_lim[0], MS_TEFF.min()),
                               min(temp_lim[1], MS_TEFF.max()), 300)
        ax.plot(t_dense, ms_lum_from_teff(t_dense), color=ink, lw=2.0,
                alpha=0.75, solid_capstyle="round", zorder=5)

    if show_main_sequence and show_ms_annotations:
        for mass, dx, dy, align in MS_ANNOTATIONS:
            teff = float(teff_from_ms_mass(mass))
            lum = float(ms_lum_from_teff(teff))
            if not (temp_lim[0] <= teff <= temp_lim[1] and
                    lum_lim[0] <= lum <= lum_lim[1]):
                continue
            ax.plot([teff], [lum], marker="o", ms=4.5, color=ink,
                    zorder=8, alpha=0.9)
            life = format_lifetime(float(ms_lifetime_gyr(mass, lum)))
            ax.annotate(f"{mass:g} $\\mathregular{{M_\\odot}}$\n{life}",
                        (teff, lum), xytext=(dx, dy),
                        textcoords="offset points",
                        ha=align, va="top", fontsize=12.5, color=ms_ink,
                        linespacing=1.25, zorder=8, path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=faint,
                                        alpha=0.6, lw=0.8,
                                        shrinkA=2, shrinkB=3))

    # --- famous stars (off by default) ----------------------------------------
    if show_famous_stars:
        for name, teff, lum, dx, dy, align in FAMOUS_STARS:
            if not (temp_lim[0] <= teff <= temp_lim[1] and
                    lum_lim[0] <= lum <= lum_lim[1]):
                continue
            # same radius-based sizing as the star samples, so the symbol
            # still represents the star's true relative size
            ax.scatter([teff], [lum], s=marker_sizes(teff, lum),
                       c=marker_colors(np.array([teff])),
                       linewidths=0.6, edgecolors=ink, zorder=9)
            ax.annotate(name, (teff, lum), xytext=(dx, dy),
                        textcoords="offset points", ha=align, va="center",
                        fontsize=12, color=ink, zorder=9, path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=faint,
                                        alpha=0.6, lw=0.8,
                                        shrinkA=2, shrinkB=3))

    # --- the Sun --------------------------------------------------------------
    if show_sun:
        sun_col = blackbody_rgb(SUN_TEFF)[0]
        if not dark:
            sun_col = sun_col * 0.85
        # drawn above every other layer, at its true radius-based size, with
        # a heavy black outline and a bold leader line from the label; on a
        # dark background the leader goes black when the Gaia sample is shown,
        # since the Sun then sits on a bright field of stars
        sun_line = "#000000" if (dark and show_gaia_sample) else ink
        ax.scatter([SUN_TEFF], [1.0], s=float(marker_sizes(SUN_TEFF, 1.0)),
                   c=[sun_col], linewidths=1.6, edgecolors="#000000",
                   zorder=10)
        ax.annotate("Sun", (SUN_TEFF, 1.0), xytext=(24, 20),
                    textcoords="offset points", fontsize=15, color=ink,
                    ha="left", va="bottom", zorder=10, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color=sun_line,
                                    alpha=0.95, lw=1.8,
                                    shrinkA=1, shrinkB=4))

    # --- user-specified points ------------------------------------------------
    # Bright red is reserved for these: no other component uses it, so the
    # points always contrast. Drawn at zorder 11, above even the Sun (10).
    if points:
        point_col = "#ff2222"
        for i, point in enumerate(points):
            teff, lum = float(point[0]), float(point[1])
            label = _auto_point_label(i) if len(point) < 3 else point[2]
            # s is a marker *area* (pt^2), so a 50% larger dot needs 2.25x
            ax.scatter([teff], [lum], s=158, c=point_col, linewidths=1.2,
                       edgecolors=bg, zorder=11)
            if label:
                ax.annotate(str(label), (teff, lum), xytext=(9, 7),
                            textcoords="offset points", ha="left",
                            va="bottom", fontsize=28, fontweight="bold",
                            color=point_col, zorder=11,
                            path_effects=[patheffects.withStroke(
                                linewidth=3.5, foreground=bg)])

    if savepath:
        fig.savefig(savepath, dpi=dpi, facecolor=bg)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_hr_diagram_suite(out_dir: str = "hr_diagram_figures",
                          **shared: object) -> list[str]:
    """Save a set of diagrams that together illustrate every feature.

    One PNG per feature group, so no single plot is overcrowded. Extra
    keyword arguments (e.g. ``dark=False``, ``temp_lim=...``) are passed
    through to :func:`plot_hr_diagram`. Returns the list of paths written.
    """
    everything_off = dict(show_main_sequence=False, show_ms_annotations=False,
                          show_radius_lines=False, show_nearest=False,
                          show_brightest=False, show_gaia_sample=False,
                          show_famous_stars=False,
                          show_luminosity_classes=False, show_star_groups=False)
    variants: list[tuple[str, dict[str, object]]] = [
        ("01_axes_and_sun", {}),
        ("02_main_sequence", dict(show_main_sequence=True,
                                  show_ms_annotations=True)),
        ("03_radius_lines", dict(show_main_sequence=True,
                                 show_radius_lines=True)),
        ("04_nearest_stars", dict(show_nearest=True)),
        ("05_brightest_stars", dict(show_brightest=True)),
        ("06_nearest_and_brightest", dict(show_nearest=True,
                                          show_brightest=True)),
        ("07_gaia_sample", dict(show_gaia_sample=True)),
        ("08_full_diagram", dict(show_main_sequence=True,
                                 show_ms_annotations=True,
                                 show_radius_lines=True, show_nearest=True,
                                 show_brightest=True, show_gaia_sample=True)),
        ("09_luminosity_classes", dict(show_gaia_sample=True,
                                       show_luminosity_classes=True)),
        ("10_star_groups", dict(show_gaia_sample=True, show_star_groups=True)),
        ("11_famous_stars", dict(show_gaia_sample=True,
                                 show_famous_stars=True)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for name, features in variants:
        kwargs: dict[str, object] = dict(everything_off)
        kwargs.update(features)
        kwargs.update(shared)
        path = os.path.join(out_dir, f"hr_diagram_{name}.png")
        plot_hr_diagram(savepath=path, show=False, **kwargs)  # type: ignore[arg-type]
        paths.append(path)
        print("saved", path)
    return paths
