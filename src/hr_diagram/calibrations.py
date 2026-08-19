"""Stellar calibrations: spectral-type tables and photometric conversions.

Dwarf (luminosity class V) properties are rounded from Pecaut & Mamajek 2013,
ApJS 208, 9 (and the regularly updated version of their Table 5 maintained by
E. Mamajek). All quantities are approximate and chosen for clarity of
visualisation, not for quantitative analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Nominal solar effective temperature (K),
# IAU 2015 Resolution B3 (Prsa et al. 2016, AJ 152, 41)
SUN_TEFF: float = 5772.0
# Bolometric magnitude zero point M_bol(Sun) = 4.74,
# IAU 2015 Resolution B2 (Mamajek et al. 2015, arXiv:1510.06262)
MBOL_SUN: float = 4.74

# Mean dwarf (luminosity class V) properties per spectral type: effective
# temperature (K), Gaia BP-RP colour (mag), mass (Msun) and luminosity (Lsun).
#            SpT     Teff   BP-RP   mass    L
MS_TABLE: list[tuple[str, float, float, float, float]] = [
    ("O5V", 42000.0, -0.42, 40.00, 5.0e5),
    ("B0V", 31500.0, -0.35, 16.00, 4.0e4),
    ("B2V", 20800.0, -0.29,  8.00, 3.0e3),
    ("B5V", 15700.0, -0.21,  5.70, 8.0e2),
    ("B8V", 12300.0, -0.14,  3.80, 2.0e2),
    ("A0V",  9700.0, -0.03,  2.20, 5.0e1),
    ("A5V",  8080.0,  0.20,  1.85, 1.4e1),
    ("F0V",  7220.0,  0.38,  1.60, 7.0e0),
    ("F5V",  6510.0,  0.56,  1.33, 3.0e0),
    ("G0V",  5930.0,  0.76,  1.06, 1.35e0),
    ("G2V",  5772.0,  0.82,  1.00, 1.0e0),
    ("G8V",  5490.0,  0.96,  0.95, 6.0e-1),
    ("K0V",  5280.0,  1.00,  0.90, 4.2e-1),
    ("K3V",  4830.0,  1.20,  0.78, 2.6e-1),
    ("K5V",  4440.0,  1.42,  0.70, 1.6e-1),
    ("M0V",  3850.0,  1.84,  0.57, 7.2e-2),
    ("M2V",  3550.0,  2.20,  0.44, 2.9e-2),
    ("M4V",  3200.0,  2.65,  0.23, 7.0e-3),
    ("M5V",  3060.0,  3.30,  0.16, 3.5e-3),
    ("M6V",  2850.0,  3.70,  0.10, 1.6e-3),
    ("M8V",  2570.0,  4.40,  0.08, 5.0e-4),
    ("M9V",  2380.0,  4.85,  0.079, 2.5e-4),
]

MS_TEFF = np.array([r[1] for r in MS_TABLE])
MS_BPRP = np.array([r[2] for r in MS_TABLE])
MS_MASS = np.array([r[3] for r in MS_TABLE])
MS_LUM = np.array([r[4] for r in MS_TABLE])

# Bolometric correction in the Gaia G band, BC_G(Teff) = Mbol - M_G, anchored
# on the IAU zero point plus Gaia DR3 photometry (Gaia Collaboration, Vallenari
# et al. 2023, A&A 674, A1) and published bolometric luminosities of benchmark
# stars from Pecaut & Mamajek 2013, ApJS 208, 9. Approximate; gravity effects
# are ignored (BC treated as a function of Teff only).
BC_G_TEFF = np.array([2380.0, 2570.0, 2980.0, 3220.0, 3850.0, 4360.0, 5280.0,
                      5772.0, 6510.0, 8080.0, 9600.0, 15700.0, 31500.0, 42000.0])
BC_G_VALS = np.array([-2.60, -2.30, -1.70, -1.00, -0.55, -0.25, 0.02,
                      0.08, 0.06, 0.00, -0.20, -1.00, -2.75, -3.60])

# Bolometric correction in V, BC_V(Teff), rounded from Pecaut & Mamajek 2013,
# ApJS 208, 9 (their dwarf table; used here for all luminosity classes).
BC_V_TEFF = np.array([2570.0, 2850.0, 3060.0, 3550.0, 3850.0, 4440.0, 5280.0,
                      5772.0, 6510.0, 7220.0, 8080.0, 9700.0, 15700.0, 31500.0, 42000.0])
BC_V_VALS = np.array([-4.10, -3.30, -2.60, -1.80, -1.40, -0.72, -0.19,
                      -0.07, -0.03, 0.00, -0.04, -0.21, -1.30, -3.00, -3.90])


def teff_from_bp_rp(bp_rp: ArrayLike) -> np.ndarray:
    """Effective temperature (K) from Gaia BP-RP colour (dwarf calibration)."""
    bp_rp = np.asarray(bp_rp, dtype=float)
    order = np.argsort(MS_BPRP)
    return np.interp(bp_rp, MS_BPRP[order], MS_TEFF[order])


def bc_g_from_teff(teff: ArrayLike) -> np.ndarray:
    """Bolometric correction (mag) in the Gaia G band at temperature `teff`."""
    return np.interp(np.log10(np.asarray(teff, dtype=float)),
                     np.log10(BC_G_TEFF), BC_G_VALS)


def bc_v_from_teff(teff: ArrayLike) -> np.ndarray:
    """Bolometric correction (mag) in Johnson V at temperature `teff`."""
    return np.interp(np.log10(np.asarray(teff, dtype=float)),
                     np.log10(BC_V_TEFF), BC_V_VALS)


def teff_from_b_v(b_v: ArrayLike) -> np.ndarray:
    """Effective temperature (K) from Johnson B-V colour.

    Ballesteros 2012, EPL 97, 34008 (blackbody fit; good to a few percent
    for main-sequence stars, cruder for supergiants).
    """
    b_v = np.asarray(b_v, dtype=float)
    return 4600.0 * (1.0 / (0.92 * b_v + 1.70) + 1.0 / (0.92 * b_v + 0.62))


def ms_lum_from_teff(teff: ArrayLike) -> np.ndarray:
    """Main-sequence luminosity (Lsun) at a given Teff, from MS_TABLE."""
    lt = np.log10(np.asarray(teff, dtype=float))
    return 10.0 ** np.interp(lt, np.log10(MS_TEFF[::-1]), np.log10(MS_LUM[::-1]))


def teff_from_ms_mass(mass: ArrayLike) -> np.ndarray:
    """Main-sequence Teff (K) at a given mass (Msun), interpolated from MS_TABLE."""
    lm = np.log10(np.asarray(mass, dtype=float))
    return 10.0 ** np.interp(lm, np.log10(MS_MASS[::-1]), np.log10(MS_TEFF[::-1]))


def ms_lifetime_gyr(mass: ArrayLike, lum: ArrayLike) -> np.ndarray:
    """Approximate main-sequence lifetime in Gyr.

    t ~ 10 Gyr * (M/Msun)/(L/Lsun) (nuclear timescale; Carroll & Ostlie,
    An Introduction to Modern Astrophysics, 2nd ed., Cambridge University
    Press, 2017). For late-M dwarfs this underestimates: true lifetimes reach
    trillions of years (Laughlin, Bodenheimer & Adams 1997, ApJ 482, 420).
    """
    return 10.0 * np.asarray(mass, dtype=float) / np.asarray(lum, dtype=float)


def wd_band_lum(teff: ArrayLike, radius: float = 0.013,
                spread: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """White-dwarf strip: constant radius ~0.013 Rsun, factor `spread` wide.

    Returns the (lower, upper) luminosity bounds (Lsun) of the strip at each
    temperature; with spread=1 both equal the strip's central luminosity.
    """
    lum = radius ** 2 * (np.asarray(teff, dtype=float) / SUN_TEFF) ** 4
    return lum / spread, lum * spread
