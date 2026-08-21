"""Star catalogues, bundled with the package as CSV.

Samples:

- the ~1000 nearest stars, and an ~84 000-star all-sky sample within 200 pc,
  both from Gaia DR3 (Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1);
- the ~100 brightest stars in the night sky, from the Hipparcos catalogue
  (Perryman et al. 1997, A&A 323, L49), which covers the very bright stars
  Gaia saturates on.

By default the load functions read the copies bundled in
``hr_diagram/data/`` (retrieved 2026-08-06; see ``data/SOURCES.md`` for
full citations), so no network access is needed. Pass a ``data_dir`` to
re-download the catalogues from the TAP services into that directory and
use those copies instead; a file already present there is reused rather
than re-downloaded, so delete it (or the directory) to force a refresh.
"""

from __future__ import annotations

import csv
import os
import urllib.parse
import urllib.request
from importlib import resources

import numpy as np

from .calibrations import MBOL_SUN, bc_g_from_teff, bc_v_from_teff, \
    teff_from_b_v, teff_from_bp_rp

# Conventional cache directory for re-downloaded catalogues, relative to
# the current working directory (e.g. load_gaia_sample(data_dir=DATA_DIR)).
DATA_DIR: str = "hr_diagram_data"

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

# ~1000 nearest stars: Gaia DR3, the 1000 largest reliable parallaxes
# (roughly within 15 pc). Gaia omits a handful of the very brightest nearby
# stars (e.g. Sirius and the alpha Cen pair), which saturate its detectors.
QUERY_NEAREST = """
SELECT TOP 1000 parallax, phot_g_mean_mag, bp_rp
FROM gaiadr3.gaia_source
WHERE parallax > 40 AND parallax_over_error > 10
AND phot_g_mean_mag IS NOT NULL AND bp_rp IS NOT NULL
ORDER BY parallax DESC
"""

# ~100 brightest stars in the night sky, from the Hipparcos catalogue, which
# is complete at these magnitudes and includes the bright stars Gaia
# saturates on.
QUERY_BRIGHTEST = """
SELECT TOP 130 "HIP", "Vmag", "Plx", "B-V"
FROM "I/239/hip_main"
WHERE "Vmag" IS NOT NULL
ORDER BY "Vmag"
"""

# Large all-sky sample within ~200 pc from Gaia DR3, with photometric and
# astrometric quality cuts in the style of Gaia Collaboration, Babusiaux
# et al. 2018, A&A 616, A10 (the Gaia DR2 HR-diagram paper) and RUWE < 1.4
# (Lindegren et al. 2021, A&A 649, A2). random_index gives an unbiased random
# subsample (~84 000 stars). Extinction is neglected: most of the sample sits
# inside the Local Bubble.
QUERY_GAIA_SAMPLE = """
SELECT phot_g_mean_mag, parallax, bp_rp
FROM gaiadr3.gaia_source
WHERE random_index < 100000000
AND parallax > 5 AND parallax_over_error > 20 AND ruwe < 1.4
AND phot_bp_mean_flux_over_error > 10 AND phot_rp_mean_flux_over_error > 10
AND bp_rp IS NOT NULL
"""


def fetch_catalogue_csv(tap_url: str, adql: str, cache_name: str,
                        timeout: float = 300,
                        data_dir: str | None = None,
                        ) -> dict[str, np.ndarray] | None:
    """Load a catalogue, bundled by default or downloaded on request.

    With ``data_dir=None`` (the default), reads the copy bundled with the
    package — no network access. With a ``data_dir``, runs the ADQL TAP
    query and caches the result there, reusing any file already present.

    Returns the rows as a dict of numpy arrays keyed by column name (empty
    fields become NaN), or None if the download failed and no cache exists.
    """
    if data_dir is None:
        text = (resources.files("hr_diagram") / "data"
                / cache_name).read_text()
    else:
        path = os.path.join(data_dir, cache_name)
        if not os.path.exists(path):
            try:
                os.makedirs(data_dir, exist_ok=True)
                payload = urllib.parse.urlencode({
                    "REQUEST": "doQuery", "LANG": "ADQL",
                    "FORMAT": "csv", "QUERY": adql.strip(),
                }).encode()
                req = urllib.request.Request(
                    tap_url, data=payload,
                    headers={"User-Agent": "pretty-hr-diagram"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    text = resp.read().decode()
                with open(path, "w") as f:
                    f.write(text)
            except Exception as err:
                print(f"Could not download {cache_name}: {err}")
                return None
        with open(path) as f:
            text = f.read()
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return None
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        out[key] = np.array([float(r[key]) if r[key] not in ("", None) else np.nan
                             for r in rows])
    return out


def load_nearest_stars(data_dir: str | None = None,
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """(teff, lum) arrays for the ~1000 nearest stars (Gaia DR3).

    Bundled data by default; pass a ``data_dir`` to download instead.
    """
    tab = fetch_catalogue_csv(GAIA_TAP, QUERY_NEAREST, "gaia_nearest.csv",
                              data_dir=data_dir)
    if tab is None:
        return None
    m_g = tab["phot_g_mean_mag"] + 5.0 * np.log10(tab["parallax"]) - 10.0
    teff = teff_from_bp_rp(tab["bp_rp"])
    mbol = m_g + bc_g_from_teff(teff)
    lum = 10.0 ** (0.4 * (MBOL_SUN - mbol))
    return teff, lum


def load_brightest_stars(n_stars: int = 100, data_dir: str | None = None,
                         ) -> tuple[np.ndarray, np.ndarray] | None:
    """(teff, lum) arrays for the ~100 brightest stars (Hipparcos).

    Bundled data by default; pass a ``data_dir`` to download instead.
    """
    tab = fetch_catalogue_csv(VIZIER_TAP, QUERY_BRIGHTEST,
                              "hipparcos_brightest.csv", data_dir=data_dir)
    if tab is None:
        return None
    good = np.isfinite(tab["Plx"]) & (tab["Plx"] > 0.4) & np.isfinite(tab["B-V"])
    v = tab["Vmag"][good][:n_stars]
    plx = tab["Plx"][good][:n_stars]
    b_v = tab["B-V"][good][:n_stars]
    m_v = v + 5.0 * np.log10(plx) - 10.0
    teff = teff_from_b_v(b_v)
    mbol = m_v + bc_v_from_teff(teff)
    lum = 10.0 ** (0.4 * (MBOL_SUN - mbol))
    return teff, lum


def load_gaia_sample(data_dir: str | None = None,
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """(teff, lum) arrays for the ~84 000-star Gaia DR3 sample within 200 pc.

    Bundled data by default; pass a ``data_dir`` to download instead.
    """
    tab = fetch_catalogue_csv(GAIA_TAP, QUERY_GAIA_SAMPLE, "gaia_sample.csv",
                              data_dir=data_dir)
    if tab is None:
        return None
    m_g = tab["phot_g_mean_mag"] + 5.0 * np.log10(tab["parallax"]) - 10.0
    teff = teff_from_bp_rp(tab["bp_rp"])
    mbol = m_g + bc_g_from_teff(teff)
    lum = 10.0 ** (0.4 * (MBOL_SUN - mbol))
    return teff, lum
