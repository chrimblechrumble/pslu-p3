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


# ===========================================================================
# 2.  Schinder et al. (2011) -- radio occultation T(z) profiles
# ===========================================================================
#
# Tables 2 and 3 from Schinder et al. (2011) Icarus 215, 460-474.
# Units: altitude (km), temperature (K), pressure (mbar), refractivity (N).
# Four soundings from 2006 (T12 and T14 flybys):
#   T12 ingress:  31.4 degS
#   T12 egress:   52.8 degS
#   T14 ingress:  32.7 degS
#   T14 egress:   34.3 degS
#
# Altitude levels common to all four soundings (selected from Tables 2 & 3).
# Values linearly interpolated from the published tables at these rounded
# altitude levels.

#: Four Schinder 2011 radio occultation profiles.
#: Dict keys are (flyby, sounding) strings.
#: Each value is a dict with keys:
#:   "latitude"    : float (deg south, positive = south)
#:   "altitude_km" : list[float]
#:   "T_K"         : list[float]
#:   "P_mbar"      : list[float]
SCHINDER_PROFILES: dict = {
    # -- Table 2: T12 flyby ----------------------------------------------
    "T12_ingress": {
        "latitude": 31.4,   #  degS
        "local_time": "dawn",
        "altitude_km": [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 180.0, 200.0,
            250.0, 300.0,
        ],
        "T_K": [
            93.14, 92.17, 90.73, 89.64, 88.51, 87.51, 86.69, 85.98,
            84.93, 84.20, 83.36, 82.08, 80.79, 79.48, 78.43, 76.86,
            73.77, 72.18, 70.36, 70.57, 75.41, 109.24, 124.54, 133.99,
            140.77, 150.64, 160.77, 168.79, 172.30, 181.04, 185.00,
        ],
        "P_mbar": [
            1467.93, 1409.95, 1335.30, 1274.22, 1205.48, 1142.91,
            1076.97, 1026.16, 966.87, 922.76, 864.69, 778.35, 692.66,
            617.85, 559.84, 488.71, 341.91, 265.37, 142.22, 75.95,
            41.44, 25.45, 17.65, 12.73, 9.36, 5.29, 2.40, 1.16, 0.73,
            0.25, 0.09,
        ],
    },
    "T12_egress": {
        "latitude": 52.8,   #  degS
        "local_time": "dusk",
        "altitude_km": [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 180.0, 200.0,
            250.0, 300.0,
        ],
        "T_K": [
            92.55, 91.64, 90.54, 89.29, 88.21, 87.39, 86.44, 85.64,
            84.81, 84.05, 83.32, 81.83, 80.56, 79.31, 77.78, 76.49,
            73.38, 71.71, 69.49, 69.79, 77.11, 112.57, 123.74, 132.20,
            138.20, 149.05, 160.31, 166.67, 172.86, 178.70, 185.00,
        ],
        "P_mbar": [
            1468.49, 1396.97, 1328.64, 1254.60, 1186.26, 1131.28,
            1063.61, 1009.60, 954.17, 902.63, 857.82, 767.61, 684.04,
            609.23, 544.09, 481.40, 333.48, 261.98, 138.58, 73.66,
            40.07, 25.16, 17.51, 12.59, 9.23, 5.16, 2.34, 1.12, 0.71,
            0.24, 0.09,
        ],
    },
    # -- Table 3: T14 flyby ----------------------------------------------
    "T14_ingress": {
        "latitude": 32.7,   #  degS
        "local_time": "dawn",
        "altitude_km": [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 180.0, 200.0,
            250.0, 300.0,
        ],
        "T_K": [
            93.35, 92.05, 90.72, 89.39, 88.33, 87.37, 86.64, 85.65,
            84.67, 84.02, 83.29, 82.00, 80.69, 79.32, 78.15, 76.86,
            73.72, 72.15, 70.30, 70.50, 75.82, 109.08, 124.60, 134.31,
            141.12, 150.63, 161.27, 168.80, 172.77, 179.92, 185.00,
        ],
        "P_mbar": [
            1473.71, 1403.24, 1334.72, 1260.78, 1198.04, 1136.26,
            1080.33, 1019.43, 961.96, 919.35, 866.19, 772.24, 691.70,
            610.63, 550.30, 487.35, 339.60, 264.36, 141.25, 75.48,
            41.20, 25.49, 17.67, 12.72, 9.36, 5.29, 2.41, 1.16, 0.73,
            0.25, 0.09,
        ],
    },
    "T14_egress": {
        "latitude": 34.3,   #  degS
        "local_time": "dusk",
        "altitude_km": [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0,
            60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 180.0, 200.0,
            250.0, 300.0,
        ],
        "T_K": [
            93.61, 92.16, 90.74, 89.51, 88.52, 87.59, 86.71, 85.73,
            85.20, 84.07, 83.48, 82.21, 80.79, 79.47, 78.24, 76.83,
            73.68, 72.24, 70.35, 70.69, 76.33, 110.15, 124.75, 134.96,
            141.51, 151.83, 163.84, 171.23, 176.01, 182.10, 185.00,
        ],
        "P_mbar": [
            1495.56, 1410.35, 1335.01, 1265.29, 1206.41, 1146.69,
            1084.08, 1015.02, 983.34, 914.41, 868.97, 779.48, 695.77,
            618.98, 551.40, 487.77, 340.50, 265.89, 142.09, 75.73,
            41.56, 25.63, 17.82, 12.87, 9.49, 5.37, 2.47, 1.21, 0.77,
            0.26, 0.10,
        ],
    },
}


def schinder_temperature_at_altitude(
    alt_km: float,
    sounding: str = "T14_ingress",
) -> float:
    """
    Interpolate the Schinder et al. (2011) T(z) profile at a given altitude.

    Parameters
    ----------
    alt_km:
        Altitude above the 2575-km reference sphere, in km.
    sounding:
        One of "T12_ingress", "T12_egress", "T14_ingress", "T14_egress".

    Returns
    -------
    float
        Temperature in K.  Returns NaN if alt_km is out of range.
    """
    profile = SCHINDER_PROFILES[sounding]
    alts = profile["altitude_km"]
    temps = profile["T_K"]
    if alt_km < alts[0] or alt_km > alts[-1]:
        return float("nan")
    return float(np.interp(alt_km, alts, temps))


def schinder_mean_profile(
    alt_km_array: Optional[np.ndarray] = None,
) -> tuple:
    """
    Return the mean T(z) profile averaged over all four Schinder soundings.

    Useful as a latitude-averaged reference troposphere / stratosphere
    temperature profile for atmospheric correction in the pipeline.

    Parameters
    ----------
    alt_km_array:
        Optional 1-D altitude grid in km.  If None, uses the common
        altitude levels from the four soundings.

    Returns
    -------
    alt_km : np.ndarray
        Altitude levels in km.
    T_mean_K : np.ndarray
        Mean temperature in K at each altitude level.
    """
    # Common altitude grid (all four profiles share these levels)
    common_alts = np.array(SCHINDER_PROFILES["T14_ingress"]["altitude_km"])

    if alt_km_array is None:
        alts = common_alts
    else:
        alts = np.asarray(alt_km_array, dtype=np.float64)

    T_all = np.stack([
        np.interp(alts, profile["altitude_km"], profile["T_K"])
        for profile in SCHINDER_PROFILES.values()
    ], axis=0)

    return alts, T_all.mean(axis=0).astype(np.float32)


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
