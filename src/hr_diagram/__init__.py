"""Teaching-oriented Hertzsprung-Russell diagram generator.

Draws HR diagrams — luminosity against surface temperature, spectral type
and true stellar colour — from real Gaia DR3 and Hipparcos data, with every
component individually toggleable so the diagram can be built up step by
step for teaching.

Quick start::

    from hr_diagram import plot_hr_diagram, save_hr_diagram_suite

    plot_hr_diagram()                          # the full diagram
    plot_hr_diagram(show_famous_stars=True)    # ... with labelled stars
    save_hr_diagram_suite("figures")           # one PNG per feature group

Star catalogues are downloaded on first use and cached in
``hr_diagram_data/`` under the current working directory. All quantities are
approximate and chosen for clarity of visualisation, not for quantitative
analysis.
"""

from .blackbody import blackbody_rgb
from .calibrations import (MBOL_SUN, SUN_TEFF, bc_g_from_teff, bc_v_from_teff,
                           ms_lifetime_gyr, ms_lum_from_teff, teff_from_b_v,
                           teff_from_bp_rp, teff_from_ms_mass, wd_band_lum)
from .catalogs import (load_brightest_stars, load_gaia_sample,
                       load_nearest_stars)
from .plot import marker_sizes, plot_hr_diagram, save_hr_diagram_suite

__version__ = "1.0.0"

__all__ = [
    "MBOL_SUN",
    "SUN_TEFF",
    "bc_g_from_teff",
    "bc_v_from_teff",
    "blackbody_rgb",
    "load_brightest_stars",
    "load_gaia_sample",
    "load_nearest_stars",
    "marker_sizes",
    "ms_lifetime_gyr",
    "ms_lum_from_teff",
    "plot_hr_diagram",
    "save_hr_diagram_suite",
    "teff_from_b_v",
    "teff_from_bp_rp",
    "teff_from_ms_mass",
    "wd_band_lum",
]
