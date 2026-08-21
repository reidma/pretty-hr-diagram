# Bundled star catalogues

These CSVs were retrieved on 2026-08-06 with the ADQL queries in
`hr_diagram/catalogs.py` and are bundled so the package works without
network access. Pass a `data_dir` to the load functions (or to
`plot_hr_diagram`) to re-download fresh copies from the sources instead.

## gaia_nearest.csv

The ~1000 nearest stars: the 1000 largest reliable parallaxes
(`parallax > 40 mas`, `parallax_over_error > 10`) from Gaia DR3, via the
ESA Gaia TAP service (https://gea.esac.esa.int/tap-server/tap).

Citation: Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1
(Gaia DR3). https://doi.org/10.1051/0004-6361/202243940

## gaia_sample.csv

An ~84 000-star unbiased all-sky sample within ~200 pc from Gaia DR3
(`parallax > 5 mas`), with astrometric and photometric quality cuts in
the style of Gaia Collaboration, Babusiaux et al. 2018, A&A 616, A10 and
`RUWE < 1.4` (Lindegren et al. 2021, A&A 649, A2), subsampled by
`random_index`. Same TAP service and citation as above.

## hipparcos_brightest.csv

The ~130 brightest stars in the night sky from the Hipparcos main
catalogue (I/239/hip_main), via the VizieR TAP service
(https://tapvizier.cds.unistra.fr/TAPVizieR/tap).

Citation: Perryman et al. 1997, A&A 323, L49 (the Hipparcos Catalogue).
Catalogue hosted by CDS/VizieR (Ochsenbein et al. 2000, A&AS 143, 23).

## Terms of use

Gaia data are provided by ESA under the Gaia Data License; use of Gaia
data requires citation of the mission (Gaia Collaboration, Prusti et al.
2016, A&A 595, A1) and the data release above. Hipparcos/VizieR data are
provided by CDS under its usage rules, which likewise require citation.
