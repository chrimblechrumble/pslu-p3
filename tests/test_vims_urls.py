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
tests/test_vims_urls.py
========================
Unit tests for the VIMS Nantes portal URL helpers.

All URLs verified by direct inspection of https://vims.univ-nantes.fr.
Tests do NOT make network requests -- they verify URL construction only.

Key facts confirmed from the portal:
  - cube ID format: {sclk}_{counter}, e.g. 1477222875_1
  - raw PDS:    https://vims.univ-nantes.fr/cube/v{id}.qub  (302 -> PDS JPL)
  - calibrated: https://vims.univ-nantes.fr/cube/C{id}_ir.cub
  - navigation: https://vims.univ-nantes.fr/cube/N{id}_ir.cub
  - previews:   https://vims.univ-nantes.fr/data/previews/{band}/{flyby}/{id}.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titan.io.vims_reader import (
    PORTAL_BASE,
    PREVIEW_BAND_COMBOS,
    cube_url_calibrated,
    cube_url_label,
    cube_url_navigation,
    cube_url_preview,
    cube_url_raw,
)

# The first cube ID from flyby TA -- verified against the portal
SAMPLE_ID = "1477222875_1"
SAMPLE_FLYBY_CODE = "00ATI"


class TestCubeURLs:
    """Verify URL construction against known portal URLs."""

    def test_raw_url_format(self) -> None:
        url = cube_url_raw(SAMPLE_ID)
        assert url == f"{PORTAL_BASE}/cube/v{SAMPLE_ID}.qub"

    def test_raw_url_has_v_prefix(self) -> None:
        """The raw URL must have a 'v' prefix before the cube ID."""
        url = cube_url_raw(SAMPLE_ID)
        assert "/cube/v" in url, "Raw URL must have 'v' prefix"

    def test_label_url_format(self) -> None:
        url = cube_url_label(SAMPLE_ID)
        assert url == f"{PORTAL_BASE}/cube/v{SAMPLE_ID}.lbl"
        assert url.endswith(".lbl")

    def test_calibrated_url_format(self) -> None:
        url = cube_url_calibrated(SAMPLE_ID)
        assert url == f"{PORTAL_BASE}/cube/C{SAMPLE_ID}_ir.cub"

    def test_calibrated_url_has_C_prefix(self) -> None:
        """Calibrated cubes use 'C' prefix (uppercase)."""
        url = cube_url_calibrated(SAMPLE_ID)
        assert "/cube/C" in url

    def test_calibrated_url_ends_ir_cub(self) -> None:
        url = cube_url_calibrated(SAMPLE_ID)
        assert url.endswith("_ir.cub")

    def test_navigation_url_format(self) -> None:
        url = cube_url_navigation(SAMPLE_ID)
        assert url == f"{PORTAL_BASE}/cube/N{SAMPLE_ID}_ir.cub"

    def test_navigation_url_has_N_prefix(self) -> None:
        """Navigation cubes use 'N' prefix (uppercase)."""
        url = cube_url_navigation(SAMPLE_ID)
        assert "/cube/N" in url

    def test_navigation_url_ends_ir_cub(self) -> None:
        url = cube_url_navigation(SAMPLE_ID)
        assert url.endswith("_ir.cub")

    def test_three_urls_are_distinct(self) -> None:
        """Raw, calibrated, and navigation URLs must all differ."""
        raw  = cube_url_raw(SAMPLE_ID)
        cal  = cube_url_calibrated(SAMPLE_ID)
        nav  = cube_url_navigation(SAMPLE_ID)
        assert raw != cal
        assert raw != nav
        assert cal != nav

    def test_all_urls_start_with_portal_base(self) -> None:
        for url in [cube_url_raw(SAMPLE_ID), cube_url_label(SAMPLE_ID),
                    cube_url_calibrated(SAMPLE_ID), cube_url_navigation(SAMPLE_ID)]:
            assert url.startswith(PORTAL_BASE)

    def test_cube_id_preserved_in_url(self) -> None:
        """The cube ID must appear verbatim in every URL."""
        for url in [cube_url_raw(SAMPLE_ID), cube_url_calibrated(SAMPLE_ID),
                    cube_url_navigation(SAMPLE_ID)]:
            assert SAMPLE_ID in url, f"Cube ID not found in: {url}"


class TestPreviewURLs:
    def test_preview_url_format(self) -> None:
        url = cube_url_preview(SAMPLE_ID, SAMPLE_FLYBY_CODE, "surface_rgb")
        band_path = PREVIEW_BAND_COMBOS["surface_rgb"]
        expected = (
            f"{PORTAL_BASE}/data/previews/"
            f"{band_path}/{SAMPLE_FLYBY_CODE}/{SAMPLE_ID}.jpg"
        )
        assert url == expected

    def test_tholin_ratio_band_combo(self) -> None:
        """The tholin proxy band ratio must use the R_159_126 combo."""
        assert PREVIEW_BAND_COMBOS["tholin_ratio"] == "R_159_126"

    def test_surface_rgb_combo(self) -> None:
        assert PREVIEW_BAND_COMBOS["surface_rgb"] == "RGB_203_158_279"

    def test_all_preview_urls_end_jpg(self) -> None:
        for combo in PREVIEW_BAND_COMBOS:
            url = cube_url_preview(SAMPLE_ID, SAMPLE_FLYBY_CODE, combo)
            assert url.endswith(".jpg"), f"Preview URL should end in .jpg: {url}"

    def test_flyby_code_in_preview_url(self) -> None:
        url = cube_url_preview(SAMPLE_ID, SAMPLE_FLYBY_CODE, "surface_rgb")
        assert SAMPLE_FLYBY_CODE in url

    def test_cube_id_in_preview_url(self) -> None:
        url = cube_url_preview(SAMPLE_ID, SAMPLE_FLYBY_CODE, "deep_surface_5um")
        assert SAMPLE_ID in url

    def test_preview_band_combos_not_empty(self) -> None:
        assert len(PREVIEW_BAND_COMBOS) >= 10


class TestPortalBase:
    def test_portal_base_is_https(self) -> None:
        assert PORTAL_BASE.startswith("https://")

    def test_portal_base_is_nantes(self) -> None:
        assert "nantes" in PORTAL_BASE