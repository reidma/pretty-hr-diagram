"""Annotation data for the HR diagram: labelled stars, regions and helpers."""

from __future__ import annotations

import numpy as np

# Spectral-class Teff boundaries for dwarfs (Pecaut & Mamajek 2013, ApJS 208, 9)
SPECTRAL_CLASSES: list[tuple[str, float, float]] = [
    ("O", 31500.0, 100000.0), ("B", 9700.0, 31500.0),
    ("A", 7220.0, 9700.0), ("F", 5930.0, 7220.0),
    ("G", 5280.0, 5930.0), ("K", 3850.0, 5280.0),
    ("M", 2300.0, 3850.0)]

# Masses (Msun) at which the main sequence is annotated with mass + lifetime,
# with the text offset (points) and alignment for each label
MS_ANNOTATIONS: list[tuple[float, float, float, str]] = [
    (40.0, 46, -4, "left"),          # top-left corner: label rides above the line
    (10.0, -30, -13, "right"),
    (5.0, -30, -13, "right"),
    (2.0, -30, -13, "right"),
    (1.0, -30, -13, "right"),
    (0.5, -32, -15, "right"),
    (0.2, -36, -18, "right"),
]

# Famous stars: (name, Teff (K), L (Lsun), label dx, dy (points), alignment).
# Teff and L are rounded from published determinations (e.g. Przybilla et al.
# 2006 for Rigel; Schiller & Przybilla 2008 for Deneb; Levesque et al. 2005
# and Ohnaka et al. 2013 for the red supergiants; Liebert et al. 2005 for
# Sirius B; Ribas et al. 2017 for Proxima). Supergiant luminosities are
# uncertain at the tens-of-percent level; positions are illustrative. The
# label offsets are hand-tuned so no two labels collide on the default axes.
FAMOUS_STARS: list[tuple[str, float, float, float, float, str]] = [
    ("Alnilam", 27500.0, 5.0e5, 20, -2, "left"),
    ("Rigel", 12100.0, 1.2e5, 16, -8, "left"),
    ("Deneb", 8500.0, 2.0e5, 18, 6, "left"),
    ("Betelgeuse", 3600.0, 1.3e5, -14, 14, "right"),
    ("Antares", 3660.0, 7.5e4, 12, -14, "left"),
    ("Spica", 25300.0, 2.0e4, 16, 4, "left"),
    ("Mira", 3000.0, 8.5e3, -16, 0, "right"),
    ("Achernar", 15000.0, 3.2e3, 16, 2, "left"),
    ("Polaris", 6015.0, 1.2e3, -18, 6, "right"),
    ("Aldebaran", 3900.0, 4.4e2, 14, 10, "left"),
    ("Regulus", 12460.0, 3.4e2, 16, -8, "left"),
    ("Arcturus", 4290.0, 1.7e2, -18, 8, "right"),
    ("Capella", 4940.0, 7.9e1, -20, 4, "right"),
    ("Vega", 9600.0, 4.0e1, 14, 12, "left"),
    ("Pollux", 4670.0, 3.3e1, 16, -10, "left"),
    ("Sirius", 9940.0, 2.5e1, 14, -12, "left"),
    ("Altair", 7550.0, 1.06e1, 14, 10, "left"),
    ("Procyon", 6530.0, 6.9e0, -18, -6, "right"),
    ("Sirius B", 25200.0, 5.6e-2, 14, 6, "left"),
    ("40 Eridani B", 16500.0, 1.3e-2, 14, 6, "left"),
    ("Barnard's Star", 3220.0, 3.5e-3, -20, 4, "right"),
    ("Proxima Centauri", 3040.0, 1.7e-3, -4, -30, "right"),
    ("Wolf 359", 2800.0, 1.1e-3, -4, -46, "right"),
]

# Schematic Morgan-Keenan luminosity-class regions (Morgan & Keenan 1973,
# ARA&A 11, 29). Boundaries are illustrative, not precise.
LUMINOSITY_CLASS_BANDS: list[tuple[str, float, float]] = [
    # (label, L_low, L_high) full-width bands
    ("Ia — luminous supergiants", 1.5e5, 8.0e5),
    ("Ib — supergiants", 1.0e4, 1.0e5),
    ("II — bright giants", 1.5e3, 1.0e4),
]
# Class IV slants from just above the solar-type turnoff (~6400 K) up to the
# base of the giant branch, staying above the dwarf sequence at every Teff.
LUMINOSITY_CLASS_PATCHES: list[
        tuple[str, list[tuple[float, float]], tuple[float, float]]] = [
    # (label, (Teff, L) vertices, label anchor)
    ("III — giants", [(5600, 15), (3000, 15), (3000, 1200), (5600, 1200)],
     (4100, 130)),
    ("IV — subgiants", [(6400, 3.0), (4750, 4.0), (4750, 16), (6400, 10)],
     (5500, 7.5)),
]

# Schematic "clouds" for notable groups of stars: (label, spine of (Teff, L)
# points the cloud follows, Gaussian half-width of the core in dex of Teff,
# label anchor, label rotation, tint colour). Real populations grade into one
# another, so each cloud is rendered as a smooth Gaussian gradient that fades
# from an opaque core into the surroundings with no edge at all; the spines
# are extended so that neighbouring populations (main sequence - subgiants -
# giants - supergiants) visibly touch. Positions are conventional HR-diagram
# landmarks (e.g. Carroll & Ostlie, An Introduction to Modern Astrophysics,
# 2nd ed., 2017, chs. 8 and 13). The main-sequence, red-dwarf and white-dwarf
# clouds are bands computed in plot_hr_diagram directly, since they follow
# the dwarf calibration and the constant-radius relation.
STAR_GROUP_CLOUDS: list[tuple[str, list[tuple[float, float]], float,
                              tuple[float, float], float,
                              tuple[float, float, float]]] = [
    ("red giants",
     [(5400, 6), (4900, 40), (4400, 200), (4000, 900), (3750, 5e3)],
     0.075, (4000, 190), 0, (1.00, 0.45, 0.35)),
    ("red supergiants",
     [(4300, 1.2e4), (3900, 6e4), (3550, 4.5e5)],
     0.085, (3520, 1.1e5), 0, (1.00, 0.40, 0.30)),
    ("blue supergiants",
     [(9700, 9e3), (12000, 4.5e4), (17000, 2e5), (26000, 6.5e5)],
     0.16, (17000, 1.5e5), 0, (0.45, 0.65, 1.00)),
]


def smooth_log_spine(points: list[tuple[float, float]],
                     iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a polyline of (Teff, L) points by corner-cutting (Chaikin) in
    log-log space; returns (log10 Teff, log10 L) arrays."""
    pts = np.log10(np.array(points, dtype=float))
    for _ in range(iterations):
        mids = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mids.append(0.75 * a + 0.25 * b)
            mids.append(0.25 * a + 0.75 * b)
        mids.append(pts[-1])
        pts = np.array(mids)
    return pts[:, 0], pts[:, 1]


def format_big_number(value: float) -> str:
    """Plain tick label: 3000, 5000, 10 000 ... (never scientific notation)."""
    v = int(round(value))
    s = f"{v:,}".replace(",", " ")
    return s if v >= 10000 else str(v)


def format_lum_tick(exponent: int) -> str:
    """Luminosity decade labels: 0.01 ... 1 ... 100, then powers of ten."""
    if -2 <= exponent <= 2:
        return str(10.0 ** exponent if exponent < 0 else 10 ** exponent)
    return f"$\\mathregular{{10^{{{exponent}}}}}$"


def format_lifetime(t_gyr: float) -> str:
    """Rounded lifetime label, e.g. '~800 Myr' or '~10 Gyr'."""
    if t_gyr < 1.0:
        myr = t_gyr * 1000.0
        val = float(f"{myr:.1g}")
        return f"~{val:g} Myr"
    val = float(f"{t_gyr:.1g}")
    return f"~{val:g} Gyr"
