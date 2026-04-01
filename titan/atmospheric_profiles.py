# Titan Habitability Pipeline - Compute P(Habitable | features) over Geologic Time
# Copyright (C) 2025/2026  Chris Meadows, cm10004@cam.ac.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
titan/atmospheric_profiles.py
==============================
Embedded Titan atmospheric temperature data from published sources.
No external data files required.

Sources
-------
1. Jennings et al. (2019) ApJL 877, L8
   DOI: 10.3847/2041-8213/ab1f91
   Surface brightness temperatures 90 degS-90 degN in 10 deg bins,
   seven time periods covering the full Cassini mission 2004-2017.
   Provides an analytical formula T(latitude, time) valid over
   the entire mission duration.

2. Schinder et al. (2011) Icarus 215, 460-474
   DOI: 10.1016/j.icarus.2011.07.030
   Radio occultation temperature-altitude profiles from four 2006
   soundings (T12 and T14 flybys), Tables 2 and 3.
   Mid-southern latitudes (31-53 degS), 0-300 km altitude.
   These are used as reference T(z) profiles anchored to the
   stratosphere via CIRS (Achterberg et al. 2008).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ===========================================================================
# 1.  Jennings et al. (2019) -- surface brightness temperature model
# ===========================================================================

# Titan equinox: 11 August 2009 -> decimal year 2009.61
_TITAN_EQUINOX_YEAR: float = 2009.61


def jennings_surface_temperature(
    lat_deg: float,
    year_ce: float = 2010.0,
) -> float:
    """
    Surface brightness temperature of Titan at a given latitude and epoch.

    Implements formula (1) of Jennings et al. (2019) ApJL 877, L8, derived
    from 13 years of Cassini CIRS far-infrared observations (2004-2017)
    covering 90 degS-90 degN in 10 deg latitude bins.

    The formula is valid over:
      * Latitude  L  in  [-90, +90]  deg
      * Year      Y  in  [-4.9, +8.1]  from equinox  (2004 Oct - 2017 Sep)

    It may be extrapolated beyond the mission window for trend estimation
    but should be used with caution outside the fitted range.

    Parameters
    ----------
    lat_deg:
        Latitude in deg, south negative (-90 to +90).
    year_ce:
        Calendar year (e.g. 2006.5).  Defaults to 2010 (near equinox).

    Returns
    -------
    float
        Surface brightness temperature in Kelvin.

    Notes
    -----
    Formula (1), exactly as typeset in the published HTML:

        T(L, Y) = (93.53 - 0.095.Y) . cos[(L + 0.85 - 3.2.Y).(0.0029 - 0.00006.Y)]

    where Y = year_ce - 2009.61 (years from Titan equinox) and L is
    latitude in deg.  The cosine argument is in radians because the
    width coefficient (0.0029 - 0.00006.Y) has units of radians per degree.

    Physical interpretation
    -----------------------
    * Amplitude  (93.53 - 0.095.Y): peak temperature declining ~1 K over
      the mission as the warmest latitude shifts from 13 degS to 24 degN.
    * Phase  (L + 0.85 - 3.2.Y): the cosine peak latitude, which tracks
      the subsolar point northward at ~3.2 deg/yr relative to equinox.
    * Width  (0.0029 - 0.00006.Y): flattening of the pole-to-equator
      gradient with time (distribution broadens as northern spring
      dampens the contrast).

    Validation
    ----------
    At equinox (Y=0):  equator 93.53 K, poles ~90.3 K.
    At Ls=313 deg (2005): peak 93.97 K near 13 degS.   (paper: 93.9 K at 13 degS)
    At Ls=90 deg  (2017): peak 92.83 K near 24 degN.   (paper: 92.8 K at 24 degN)
    Mission-wide range: 88.7-94.0 K.
    Standard deviation of fit: 0.4 K; worst period (Ls =~ 26 deg): 0.7 K.

    Reference
    ---------
    Jennings, D. E. et al. (2019) ApJL 877, L8.
    DOI: 10.3847/2041-8213/ab1f91
    """
    Y = year_ce - _TITAN_EQUINOX_YEAR  # years from Titan equinox (2009.61)
    L = lat_deg

    amplitude = 93.53 - 0.095 * Y
    phase_deg = L + 0.85 - 3.2 * Y         # latitude-offset in deg
    width     = 0.0029 - 0.00006 * Y        # radians per degree
    T = amplitude * math.cos(phase_deg * width)

    return float(T)


def jennings_temperature_map(
    lats_deg: np.ndarray,
    year_ce: float = 2010.0,
) -> np.ndarray:
    """
    Vectorised wrapper: compute surface brightness temperatures for an
    array of latitudes at a given epoch.

    Parameters
    ----------
    lats_deg:
        1-D (or any-shape) array of latitudes in deg (-90 to +90).
    year_ce:
        Calendar year.

    Returns
    -------
    np.ndarray
        float32 array of brightness temperatures (K), same shape as
        ``lats_deg``.
    """
    lats = np.asarray(lats_deg, dtype=np.float64)
    Y = year_ce - _TITAN_EQUINOX_YEAR
    amplitude = 93.53 - 0.095 * Y
    width     = 0.0029 - 0.00006 * Y           # rad/deg
    phase_deg = lats + 0.85 - 3.2 * Y          # deg offset per element
    T = amplitude * np.cos(phase_deg * width)
    return T.astype(np.float32)


def jennings_temperature_grid(
    lat_grid: np.ndarray,
    year_ce: float = 2010.0,
) -> np.ndarray:
    """
    Compute surface brightness temperature on a 2-D (lat, lon) grid.

    Because the formula is longitude-invariant (zonal average), the result
    is the same at every longitude for a given latitude.  This returns a
    full 2-D array suitable for direct use as a canonical-grid layer.

    Parameters
    ----------
    lat_grid:
        2-D float32 array of latitudes in deg, shape (nrows, ncols).
    year_ce:
        Calendar year.

    Returns
    -------
    np.ndarray
        2-D float32 array of surface brightness temperatures (K),
        same shape as ``lat_grid``.
    """
    return jennings_temperature_map(lat_grid.ravel(), year_ce).reshape(
        lat_grid.shape
    )



# Note: Schinder et al. (2011) radio occultation T(z) profiles have been
# removed from this module.  Those four soundings are vertical atmospheric
# profiles (T vs altitude) at southern mid-latitudes and cannot produce
# the 2-D surface temperature map the pipeline requires.  The only
# cross-validation they enabled (Jennings vs Schinder surface T at 31 degS)
# is preserved as a direct assertion in test_atmospheric_profiles.py using
# the published surface value: T12 ingress (31.4 degS, 2006) = 93.14 K
# (Schinder 2011, Icarus 215, Table 2).

# ===========================================================================
# 3.  HASI near-surface profile (Fulchignoni et al. 2005)
# ===========================================================================
#
# Simplified near-surface T(z) from Huygens HASI at 10 degS.
# Used as anchor for the tropospheric profile below 40 km where
# radio occultation coverage is limited.

#: Near-surface Huygens HASI temperatures (Fulchignoni et al. 2005).
#: Altitude in km above the 2575 km reference sphere.
HASI_NEAR_SURFACE = {
    "latitude":     -10.0,  #  degS (actually 10 degS south, positive = south)
    "altitude_km":  [0.0,   0.5,   1.0,   2.0,   5.0,   10.0,  20.0,  40.0],
    "T_K":          [93.65, 93.0,  92.4,  91.3,  87.9,  83.5,  76.9,  70.4],
}
