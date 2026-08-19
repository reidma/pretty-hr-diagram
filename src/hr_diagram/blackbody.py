"""True visual colours of stars, approximated as blackbodies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def blackbody_rgb(teff: ArrayLike) -> np.ndarray:
    """sRGB colour(s) of a blackbody at temperature `teff` (K).

    Normalised so the brightest channel is 1 (hue only, not brightness).
    Stars are close to blackbodies, so this approximates their true visual
    colours (Ballesteros 2012, EPL 97, 34008).

    Pipeline: Planck spectrum -> CIE 1931 XYZ using the multi-lobe Gaussian
    fits to the colour-matching functions of Wyman, Sloan & Shirley 2013,
    J. Computer Graphics Techniques 2(2), 1-11 -> linear sRGB
    (IEC 61966-2-1:1999 primaries) -> gamma-encoded sRGB.

    Returns an (n, 3) array of RGB values in [0, 1], one row per input
    temperature (scalars are treated as length-1 arrays).
    """
    teff = np.atleast_1d(np.asarray(teff, dtype=float))
    lam = np.arange(380.0, 751.0, 1.0) * 1e-9          # m
    lam_nm = lam * 1e9

    def lobe(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    xbar = (1.056 * lobe(lam_nm, 599.8, 37.9, 31.0)
            + 0.362 * lobe(lam_nm, 442.0, 16.0, 26.7)
            - 0.065 * lobe(lam_nm, 501.1, 20.4, 26.2))
    ybar = (0.821 * lobe(lam_nm, 568.8, 46.9, 40.5)
            + 0.286 * lobe(lam_nm, 530.9, 16.3, 31.1))
    zbar = (1.217 * lobe(lam_nm, 437.0, 11.8, 36.0)
            + 0.681 * lobe(lam_nm, 459.0, 26.0, 13.8))

    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    # Planck spectral radiance for each (T, lambda); constant factors cancel
    # in the normalisation below
    with np.errstate(over="ignore"):
        planck = 1.0 / (lam[None, :] ** 5
                        * np.expm1(h * c / (lam[None, :] * kb * teff[:, None])))

    xyz = np.stack([planck @ xbar, planck @ ybar, planck @ zbar], axis=1)
    xyz /= xyz[:, 1:2]

    m = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])          # IEC 61966-2-1:1999
    rgb = np.clip(xyz @ m.T, 0.0, None)
    rgb /= rgb.max(axis=1, keepdims=True)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                   1.055 * rgb ** (1.0 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)
