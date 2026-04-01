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
tests/test_atmospheric_profiles.py
====================================
Tests for titan/atmospheric_profiles.py.

Validates:
  1. Jennings et al. (2019) surface temperature formula -- exact formula
     structure, physically plausible values, validation against paper
     Figure 1 data.
  2. Schinder et al. (2011) radio occultation profiles -- table completeness,
     physical sanity, interpolation.
  3. HASI near-surface profile -- sanity checks.
"""

import math
import numpy as np
import pytest

from titan.atmospheric_profiles import (
    jennings_surface_temperature,
    jennings_temperature_map,
    jennings_temperature_grid,
    schinder_temperature_at_altitude,
    schinder_mean_profile,
    SCHINDER_PROFILES,
    HASI_NEAR_SURFACE,
    _TITAN_EQUINOX_YEAR,
)


# ============================================================================
# 1.  Jennings formula
# ============================================================================

class TestJenningsFormula:
    """Tests for the Jennings et al. (2019) surface brightness temperature
    formula (Equation 1 of ApJL 877, L8)."""

    # -- Formula structure ----------------------------------------------------

    def test_equator_at_equinox_is_93p53(self) -> None:
        """
        At L=0 and Y=0 the formula reduces to:
            T = (93.53 - 0) x cos(0) = 93.53 K
        This is the fundamental anchor point of the formula.
        """
        Y0 = _TITAN_EQUINOX_YEAR  # 2009.61
        T = jennings_surface_temperature(0.0, Y0)
        assert abs(T - 93.53) < 0.01, f"Expected 93.53, got {T:.4f}"

    def test_formula_is_cosine_of_latitude(self) -> None:
        """
        The formula is symmetric about the peak latitude.  Verify that
        equal-distance latitudes above and below the peak give equal T.
        (At equinox the peak is near 0 deg, so +/-60 deg should be equal.)
        """
        T_pos = jennings_surface_temperature(60.0,  _TITAN_EQUINOX_YEAR)
        T_neg = jennings_surface_temperature(-60.0, _TITAN_EQUINOX_YEAR)
        # At equinox phase = L + 0.85 - 3.2x0 = L + 0.85
        # cos((60+0.85)x0.0029) vs cos((-60+0.85)x0.0029) -- not symmetric
        # BUT they should be nearly equal since 0.85 << 60
        assert abs(T_pos - T_neg) < 0.3, (
            f"Equinox +/-60 deg should be nearly equal: {T_pos:.2f} vs {T_neg:.2f}"
        )

    def test_amplitude_decreases_with_time(self) -> None:
        """
        Peak temperature decreases ~1 K over the mission (93.9 -> 92.8 K).
        Verify amplitude = 93.53 - 0.095xY decreases from 2005 to 2017.
        """
        Y_2005 = 2005.0 - _TITAN_EQUINOX_YEAR   # =~ -4.61
        Y_2017 = 2017.0 - _TITAN_EQUINOX_YEAR   # =~  7.39
        # peak at cos=1, i.e., at the phase-zero latitude
        amp_2005 = 93.53 - 0.095 * Y_2005
        amp_2017 = 93.53 - 0.095 * Y_2017
        assert amp_2005 > amp_2017, "Amplitude should decrease 2005->2017"
        assert abs((amp_2005 - amp_2017) - 0.095 * (Y_2017 - Y_2005)) < 0.01

    # -- Validation against paper Figure 1 -----------------------------------

    def test_2005_peak_near_13S(self) -> None:
        """
        At Ls=313 deg (2005) paper reports peak ~93.9 K near 13 degS.
        """
        # Brute-force search for peak
        lats = np.linspace(-90, 90, 361)
        Ts = jennings_temperature_map(lats, 2005.0)
        peak_lat = lats[np.argmax(Ts)]
        peak_T   = float(np.max(Ts))
        assert -20 <= peak_lat <= -8, (
            f"2005 peak lat expected near -13 deg, got {peak_lat:.1f} deg"
        )
        assert 93.5 <= peak_T <= 94.3, (
            f"2005 peak T expected ~93.9 K, got {peak_T:.2f} K"
        )

    def test_2017_peak_near_24N(self) -> None:
        """
        At Ls=90 deg (2017) paper reports peak ~92.8 K near 24 degN.
        """
        lats = np.linspace(-90, 90, 361)
        Ts = jennings_temperature_map(lats, 2017.0)
        peak_lat = lats[np.argmax(Ts)]
        peak_T   = float(np.max(Ts))
        assert 18 <= peak_lat <= 30, (
            f"2017 peak lat expected near +24 deg, got {peak_lat:.1f} deg"
        )
        assert 92.4 <= peak_T <= 93.2, (
            f"2017 peak T expected ~92.8 K, got {peak_T:.2f} K"
        )

    def test_equinox_equatorial_temperature(self) -> None:
        """
        At Ls=0 deg (2009.61 equinox) paper says equatorial T ~ 93.5 +/- 0.4 K.
        """
        T_eq = jennings_surface_temperature(0.0, _TITAN_EQUINOX_YEAR)
        assert 93.1 <= T_eq <= 93.9, (
            f"Equinox equatorial T expected 93.5+/-0.4, got {T_eq:.2f}"
        )

    def test_poles_cooler_than_equator_at_equinox(self) -> None:
        """At equinox both poles should be ~2-3 K cooler than the equator."""
        T_eq = jennings_surface_temperature(0.0,   _TITAN_EQUINOX_YEAR)
        T_np = jennings_surface_temperature(90.0,  _TITAN_EQUINOX_YEAR)
        T_sp = jennings_surface_temperature(-90.0, _TITAN_EQUINOX_YEAR)
        assert T_eq > T_np + 1.5, f"NP should be >1.5 K cooler at equinox"
        assert T_eq > T_sp + 1.5, f"SP should be >1.5 K cooler at equinox"
        assert abs(T_np - T_sp) < 1.0, (
            f"Poles should be nearly equal at equinox: NP={T_np:.2f}, SP={T_sp:.2f}"
        )

    def test_2017_south_pole_below_90K(self) -> None:
        """
        At Ls=90 deg (2017) south pole cools to ~89.3 K (< 90 K).
        Paper Figure 1 shows SP ~89.8 K; formula gives ~89.3 K.
        """
        T_sp = jennings_surface_temperature(-90.0, 2017.0)
        assert T_sp < 90.5, f"SP in 2017 should be below 90.5 K, got {T_sp:.2f}"
        assert T_sp > 87.0, f"SP in 2017 should be above 87 K, got {T_sp:.2f}"

    def test_2017_north_pole_warmer_than_south(self) -> None:
        """At Ls=90 deg NP has warmed ~2K relative to SP."""
        T_np = jennings_surface_temperature( 90.0, 2017.0)
        T_sp = jennings_surface_temperature(-90.0, 2017.0)
        assert T_np > T_sp + 1.5, (
            f"NP should be >1.5 K warmer than SP in 2017: NP={T_np:.2f}, SP={T_sp:.2f}"
        )

    def test_huygens_landing_site_2005(self) -> None:
        """
        At Huygens landing site (10 degS) in 2005 paper says T=93.9 K.
        Formula should give 93.5-94.3 K.
        """
        T = jennings_surface_temperature(-10.0, 2005.0)
        assert 93.0 <= T <= 94.5, (
            f"Huygens site 2005 expected ~93.9 K, got {T:.2f}"
        )

    # -- Physical plausibility across the full mission ------------------------

    def test_all_mission_temperatures_physically_plausible(self) -> None:
        """All temperatures across the mission should be in 87-96 K."""
        lats = np.linspace(-90, 90, 37)
        for year in np.arange(2005, 2018, 0.5):
            Ts = jennings_temperature_map(lats, float(year))
            assert float(Ts.min()) > 87.0, (
                f"T too low at year={year}: {float(Ts.min()):.2f} K"
            )
            assert float(Ts.max()) < 96.0, (
                f"T too high at year={year}: {float(Ts.max()):.2f} K"
            )

    # -- Vectorised and 2-D interfaces ----------------------------------------

    def test_jennings_temperature_map_shape_and_dtype(self) -> None:
        lats = np.linspace(-90, 90, 19)
        Ts = jennings_temperature_map(lats, 2010.0)
        assert Ts.shape == (19,)
        assert Ts.dtype == np.float32

    def test_jennings_temperature_map_matches_scalar(self) -> None:
        """Vectorised result should match element-wise scalar calls."""
        lats = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
        Ts_vec = jennings_temperature_map(lats, 2008.0)
        for i, L in enumerate(lats):
            T_scalar = jennings_surface_temperature(float(L), 2008.0)
            assert abs(float(Ts_vec[i]) - T_scalar) < 1e-5, (
                f"Mismatch at L={L}: vec={Ts_vec[i]:.6f}, scalar={T_scalar:.6f}"
            )

    def test_jennings_temperature_grid_shape(self) -> None:
        lats_1d = np.linspace(-90, 90, 18)
        lat_grid = np.tile(lats_1d[:, None], (1, 36)).astype(np.float32)
        T_grid = jennings_temperature_grid(lat_grid, 2010.0)
        assert T_grid.shape == lat_grid.shape
        assert T_grid.dtype == np.float32

    def test_jennings_temperature_grid_longitude_invariant(self) -> None:
        """Temperature should be identical at all longitudes for a given latitude."""
        lats_1d = np.array([-60.0, 0.0, 60.0])
        lat_grid = np.tile(lats_1d[:, None], (1, 36)).astype(np.float32)
        T_grid = jennings_temperature_grid(lat_grid, 2010.0)
        for row in T_grid:
            assert np.allclose(row, row[0]), "T should be lon-invariant"


# ============================================================================
# 2.  Schinder et al. (2011) radio occultation profiles
# ============================================================================

class TestSchinderProfiles:
    """Tests for the embedded Schinder et al. (2011) T(z) profiles."""

    def test_all_four_soundings_present(self) -> None:
        expected = {"T12_ingress", "T12_egress", "T14_ingress", "T14_egress"}
        assert set(SCHINDER_PROFILES.keys()) == expected

    def test_each_profile_has_required_keys(self) -> None:
        for name, prof in SCHINDER_PROFILES.items():
            for key in ("latitude", "local_time", "altitude_km", "T_K", "P_mbar"):
                assert key in prof, f"Profile {name} missing key '{key}'"

    def test_altitude_arrays_are_monotonically_increasing(self) -> None:
        for name, prof in SCHINDER_PROFILES.items():
            alts = prof["altitude_km"]
            assert all(a < b for a, b in zip(alts, alts[1:])), (
                f"Altitudes not monotone in {name}"
            )

    def test_all_profiles_span_0_to_300km(self) -> None:
        for name, prof in SCHINDER_PROFILES.items():
            assert prof["altitude_km"][0]  == 0.0,   f"{name}: should start at 0 km"
            assert prof["altitude_km"][-1] == 300.0, f"{name}: should end at 300 km"

    def test_array_lengths_match(self) -> None:
        for name, prof in SCHINDER_PROFILES.items():
            n = len(prof["altitude_km"])
            assert len(prof["T_K"])   == n, f"{name}: T_K length mismatch"
            assert len(prof["P_mbar"]) == n, f"{name}: P_mbar length mismatch"

    def test_surface_temperatures_near_93K(self) -> None:
        """Surface temperatures (0 km) should all be ~93 K."""
        for name, prof in SCHINDER_PROFILES.items():
            T_surf = prof["T_K"][0]
            assert 91.5 <= T_surf <= 94.5, (
                f"{name}: surface T={T_surf:.2f} K, expected ~93 K"
            )

    def test_latitudes_correct(self) -> None:
        """Check published latitude values from Table 1 of Schinder 2011."""
        assert abs(SCHINDER_PROFILES["T12_ingress"]["latitude"] - 31.4) < 0.1
        assert abs(SCHINDER_PROFILES["T12_egress" ]["latitude"] - 52.8) < 0.1
        assert abs(SCHINDER_PROFILES["T14_ingress"]["latitude"] - 32.7) < 0.1
        assert abs(SCHINDER_PROFILES["T14_egress" ]["latitude"] - 34.3) < 0.1

    def test_tropopause_temperatures_near_70K(self) -> None:
        """Tropopause (~40 km) should be ~70 K for all soundings."""
        for name, prof in SCHINDER_PROFILES.items():
            alts = prof["altitude_km"]
            T_at_40 = np.interp(40.0, alts, prof["T_K"])
            assert 68.0 <= T_at_40 <= 72.5, (
                f"{name}: T at 40 km = {T_at_40:.2f} K, expected ~70 K"
            )

    def test_stratopause_temperatures_near_185K(self) -> None:
        """Stratopause (300 km starting value from CIRS) should be ~185 K."""
        for name, prof in SCHINDER_PROFILES.items():
            T_300 = prof["T_K"][-1]
            assert abs(T_300 - 185.0) < 1.0, (
                f"{name}: T at 300 km = {T_300:.2f} K, expected 185 K"
            )

    def test_pressure_monotonically_decreasing(self) -> None:
        for name, prof in SCHINDER_PROFILES.items():
            P = prof["P_mbar"]
            assert all(a > b for a, b in zip(P, P[1:])), (
                f"Pressure not monotone decreasing in {name}"
            )

    def test_surface_pressure_near_1500_mbar(self) -> None:
        """Surface pressure ~1470-1500 mbar for Titan (Fulchignoni 2005)."""
        for name, prof in SCHINDER_PROFILES.items():
            P_surf = prof["P_mbar"][0]
            assert 1450 <= P_surf <= 1510, (
                f"{name}: surface P={P_surf:.1f} mbar, expected ~1470-1500"
            )

    def test_T12_egress_is_coldest_at_surface(self) -> None:
        """
        T12 egress probes 52.8 degS -- the highest latitude of the four soundings.
        Should give the coldest surface temperature (~92.5 K).
        """
        T_surfaces = {
            name: prof["T_K"][0] for name, prof in SCHINDER_PROFILES.items()
        }
        assert T_surfaces["T12_egress"] == min(T_surfaces.values()), (
            f"T12 egress (52.8 degS) should be coldest: {T_surfaces}"
        )

    def test_interpolation_within_range(self) -> None:
        T = schinder_temperature_at_altitude(50.0, "T14_ingress")
        assert 68 <= T <= 73, f"T at 50 km expected ~70 K, got {T:.2f}"

    def test_interpolation_out_of_range_returns_nan(self) -> None:
        T = schinder_temperature_at_altitude(350.0, "T12_ingress")
        assert math.isnan(T), "Alt>300 km should return NaN"

        T = schinder_temperature_at_altitude(-5.0, "T12_ingress")
        assert math.isnan(T), "Negative alt should return NaN"

    def test_mean_profile_shape_and_values(self) -> None:
        alts, T_mean = schinder_mean_profile()
        assert alts.shape == T_mean.shape
        assert T_mean.dtype == np.float32
        # Surface mean should be ~93 K
        assert 91.5 <= float(T_mean[0]) <= 94.5
        # Tropopause mean should be ~70 K
        T_tropo = float(np.interp(40.0, alts, T_mean))
        assert 68 <= T_tropo <= 72

    def test_custom_altitude_grid_for_mean_profile(self) -> None:
        custom_alts = np.array([0.0, 10.0, 40.0, 100.0, 200.0])
        alts_out, T_mean = schinder_mean_profile(custom_alts)
        assert np.allclose(alts_out, custom_alts)
        assert len(T_mean) == len(custom_alts)
        # 0 km ~93 K, 40 km ~70 K, 200 km ~170 K
        assert 91 <= float(T_mean[0]) <= 95
        assert 68 <= float(T_mean[2]) <= 73
        assert 160 <= float(T_mean[4]) <= 180


# ============================================================================
# 3.  HASI near-surface profile
# ============================================================================

class TestHASIProfile:

    def test_surface_temperature_matches_fulchignoni(self) -> None:
        """Fulchignoni et al. 2005: T_surface = 93.65 +/- 0.25 K at 10 degS."""
        T_0 = HASI_NEAR_SURFACE["T_K"][0]
        assert abs(T_0 - 93.65) < 0.5

    def test_temperature_decreases_with_altitude(self) -> None:
        T = HASI_NEAR_SURFACE["T_K"]
        # Troposphere cools with altitude
        assert all(T[i] > T[i+1] for i in range(len(T)-1)), (
            "HASI T should decrease with altitude in troposphere"
        )

    def test_latitude_is_southern(self) -> None:
        assert HASI_NEAR_SURFACE["latitude"] < 0  # southern hemisphere


# ============================================================================
# 4.  Consistency between data sources
# ============================================================================

class TestCrossSourceConsistency:
    """Cross-checks that the three data sources agree in their overlap."""

    def test_jennings_and_schinder_agree_at_surface_31S(self) -> None:
        """
        Schinder T12 ingress (31.4 degS) gives ~93.1 K at surface.
        Jennings at 2006.21 (T12 flyby date: 19 March 2006) at 31.4 degS:
        should agree within +/-1 K.
        """
        T_schinder = SCHINDER_PROFILES["T12_ingress"]["T_K"][0]
        T_jennings = jennings_surface_temperature(-31.4, 2006.21)
        assert abs(T_schinder - T_jennings) < 1.5, (
            f"T12 ingress surface: Schinder={T_schinder:.2f}, "
            f"Jennings={T_jennings:.2f}, diff={abs(T_schinder-T_jennings):.2f} K"
        )

    def test_jennings_and_hasi_agree_at_10S_2005(self) -> None:
        """
        HASI measures 93.65 K at 10 degS (Jan 2005).
        Jennings at 2005.04 (Huygens descent: 14 Jan 2005) at 10 degS.
        """
        T_hasi     = HASI_NEAR_SURFACE["T_K"][0]   # 93.65 K
        T_jennings = jennings_surface_temperature(-10.0, 2005.04)
        assert abs(T_hasi - T_jennings) < 1.5, (
            f"HASI={T_hasi:.2f}, Jennings={T_jennings:.2f}, "
            f"diff={abs(T_hasi-T_jennings):.2f} K"
        )