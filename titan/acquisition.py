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
titan/acquisition.py
=====================
Stage 1 -- Data Acquisition.

Manages downloading, verifying, and cataloguing all raw datasets.

Many Titan datasets require manual steps (web form submission, email
request, institutional login) that cannot be automated.  This module
handles both cases:

  Automated downloads
    - Cornell eCommons GTDR tiles (direct URL, no login)
    - USGS Astropedia GeoTIFFs (if direct download URLs are known)

  Manual downloads  (requires human action)
    - USGS Astropedia via Map-a-Planet (form-based reprojection tool)
    - VIMS parquet/mosaics from Nantes portal
    - JPL geomorphology shapefiles (contact Rosaly Lopes or Mendeley)
    - PDS Atmospheres Node (CIRS, INMS)
    - PDS Plasma Interactions (MAG)

For manual datasets, the module:
  1. Detects whether the file is already present
  2. Prints step-by-step download instructions
  3. Verifies integrity (SHA-256) once the file appears
  4. Generates a status report

SHA-256 verification
---------------------
Where hashes are known they are embedded in DatasetSpec.sha256.
Once verified, a <filename>.verified stamp file is written so
re-verification is skipped on subsequent runs.

References
----------
GTDR:    https://ecommons.cornell.edu/handle/1813/57031
Astropedia: https://astrogeology.usgs.gov/search?target=titan
VIMS:    https://vims.univ-nantes.fr/
PDS Atm: https://pds-atmospheres.nmsu.edu/
PDS PPI: https://pds-ppi.igpp.ucla.edu/
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Datasets that can be fetched automatically (no login / form required).
# Cornell eCommons: direct download (GTDR/GTDE tiles, .IMG.gz).
# CaltechDATA: direct S3-backed downloads (VIMS+ISS mosaic).
_AUTO_DOWNLOAD_STEMS = {
    "gtde_east",        # interpolated global DEM -- preferred
    "gtde_west",
    "gtdr_east",        # sparse GTDR standard tracks
    "gtdr_west",
    "vims_mosaic",
    "vims_mosaic_hdr",
}


# ---------------------------------------------------------------------------
# SHA-256 utility
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 hash of a file.

    Parameters
    ----------
    path:
        File to hash.
    chunk_size:
        Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(
    path: Path,
    expected_sha256: str,
    stamp_dir: Optional[Path] = None,
) -> bool:
    """
    Verify a file's SHA-256 hash and write a stamp file if it passes.

    Parameters
    ----------
    path:
        File to verify.
    expected_sha256:
        Expected hex digest.  Empty string = skip verification.
    stamp_dir:
        Directory where ``.verified`` stamp files are stored.
        Defaults to the file's parent directory.

    Returns
    -------
    bool
        True if verified (or no hash provided), False if mismatch.
    """
    if not expected_sha256:
        return True  # no hash to check

    stamp_dir = Path(stamp_dir or path.parent)
    stamp     = stamp_dir / f"{path.name}.verified"

    # Skip re-verification if stamp exists and is newer than the file
    if stamp.exists() and stamp.stat().st_mtime >= path.stat().st_mtime:
        return True

    logger.info("Verifying %s ...", path.name)
    actual = sha256_file(path)
    if actual.lower() == expected_sha256.lower():
        stamp.write_text(f"{actual}\n{time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        logger.info("[OK]  %s  verified.", path.name)
        return True
    else:
        logger.error(
            "SHA-256 MISMATCH for %s\n  expected: %s\n  actual:   %s",
            path.name, expected_sha256.lower(), actual,
        )
        return False


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """
    Download a file with a tqdm progress bar.

    A browser-like User-Agent header is sent because some servers
    (CaltechDATA, Cornell eCommons) return 403 or redirect-loops
    without one.

    Parameters
    ----------
    url:
        Remote URL.
    dest:
        Local destination path.
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        raise ImportError("pip install requests tqdm")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=dest.name, leave=True,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))

    shutil.move(str(tmp), str(dest))
    logger.info("Downloaded %s -> %s", url, dest)


# ---------------------------------------------------------------------------
# Acquisition manager
# ---------------------------------------------------------------------------

class DataAcquisitionManager:
    """
    Manages downloading and verifying all pipeline datasets.

    Parameters
    ----------
    config:
        Pipeline configuration object.
    """

    def __init__(self, config: "PipelineConfig") -> None:
        from configs.pipeline_config import PipelineConfig
        self.config = config
        self.config.make_dirs()

    # -- Public interface ----------------------------------------------------

    def acquire_all(self, dry_run: bool = False) -> "AcquisitionReport":
        """
        Attempt to acquire every dataset in the catalogue.

        Automated datasets are downloaded if absent.
        Manual datasets print instructions and are checked for presence.

        Parameters
        ----------
        dry_run:
            If True, only report status without downloading.

        Returns
        -------
        AcquisitionReport
        """
        report = AcquisitionReport()

        for name, spec in self.config.datasets.items():
            dest = self.config.data_dir / spec.local_filename

            # -- Synthesised datasets (no download needed) ------------------
            # Generated at runtime by the preprocessing pipeline
            # (e.g. cirs_temperature from the Jennings 2019 formula).
            if spec.file_format == "synthesised":
                logger.info(
                    "'%s' is synthesised at runtime -- no download required.", name
                )
                report.present.append(name)
                continue

            # -- Directory datasets (shapefiles) ----------------------------
            if spec.file_format == "shapefile_dir":
                status = self._check_shapefile_dir(name, spec, report)
                continue

            # -- Special case: VIMS parquet (multiple acceptable filenames) -
            if name == "vims_footprint":
                found = self.config.get_vims_parquet()
                if found is not None:
                    logger.info(
                        "VIMS parquet found: %s (%s rows)%s",
                        found,
                        "~5.4M" if found.stat().st_size > 10_000_000 else "1,000 (sample)",
                        " [via --vims-parquet]" if self.config.vims_parquet_path else "",
                    )
                    report.present.append(name)
                else:
                    report.manual_required.append(name)
                    if not dry_run:
                        self._print_manual_instructions(name, spec)
                continue

            # -- Already present --------------------------------------------
            if dest.exists():
                ok = verify_file(dest, spec.sha256)
                if ok:
                    report.present.append(name)
                else:
                    report.corrupt.append(name)
                continue

            # -- Auto-download ----------------------------------------------
            if name in _AUTO_DOWNLOAD_STEMS and spec.url and not dry_run:
                try:
                    logger.info("Auto-downloading %s ...", name)
                    _download_with_progress(spec.url, dest)
                    verify_file(dest, spec.sha256)
                    report.downloaded.append(name)
                except Exception as exc:
                    logger.warning("Auto-download failed for %s: %s", name, exc)
                    report.failed.append((name, str(exc)))
                    self._print_manual_instructions(name, spec)

            # -- Manual download required -----------------------------------
            else:
                report.manual_required.append(name)
                if not dry_run:
                    self._print_manual_instructions(name, spec)

        return report

    def acquire_one(self, name: str) -> bool:
        """
        Acquire a single dataset by name.

        Parameters
        ----------
        name:
            Dataset key from the catalogue.

        Returns
        -------
        bool
            True if the dataset is now present and verified.
        """
        spec = self.config.datasets.get(name)
        if spec is None:
            logger.error("Dataset '%s' not in catalogue.", name)
            return False

        dest = self.config.data_dir / spec.local_filename
        if dest.exists():
            return verify_file(dest, spec.sha256)

        if name in _AUTO_DOWNLOAD_STEMS and spec.url:
            try:
                _download_with_progress(spec.url, dest)
                return verify_file(dest, spec.sha256)
            except Exception as exc:
                logger.error("Download failed for %s: %s", name, exc)
                return False

        self._print_manual_instructions(name, spec)
        return False

    def status(self) -> "AcquisitionReport":
        """
        Report current status of all datasets without downloading.

        Returns
        -------
        AcquisitionReport
        """
        return self.acquire_all(dry_run=True)

    # -- Helpers ------------------------------------------------------------

    def _check_shapefile_dir(
        self,
        name: str,
        spec: "DatasetSpec",
        report: "AcquisitionReport",
    ) -> None:
        """
        Check whether the geomorphology shapefile directory is populated.
        """
        shp_dir = self.config.get_shapefile_dir()
        expected_layers = [
            "Craters", "Dunes", "Plains_3",
            "Basins", "Mountains", "Labyrinth",
        ]
        present_layers = [
            s for s in expected_layers
            if (shp_dir / f"{s}.shp").exists()
        ]
        if len(present_layers) == len(expected_layers):
            report.present.append(name)
        elif len(present_layers) > 0:
            report.partial.append((name, present_layers))
        else:
            report.manual_required.append(name)
            self._print_manual_instructions(name, spec)

    @staticmethod
    def _print_manual_instructions(name: str, spec: "DatasetSpec") -> None:
        """Print clearly formatted manual download instructions."""
        sep = "-" * 65
        print(f"\n{sep}")
        print(f"  MANUAL DOWNLOAD REQUIRED: {name}")
        print(sep)
        print(f"  Description : {spec.description[:80]}..."
              if len(spec.description) > 80 else f"  Description : {spec.description}")
        print(f"  Target file : {spec.local_filename}")
        print(f"  URL         : {spec.url}")
        if spec.manual_instructions:
            print(f"\n  Instructions:")
            for line in spec.manual_instructions.split("\n"):
                print(f"    {line.strip()}")
        if spec.sha256:
            print(f"\n  Expected SHA-256: {spec.sha256}")
        print(sep)

    def create_gravity_k2_json(self) -> Path:
        """
        Create the gravity_k2.json scalar file if it doesn't exist.

        This is a trivial file containing the published Love number from
        Iess et al. (2012).  No download required; we create it ourselves.

        Returns
        -------
        Path
            Path to the created file.
        """
        dest = self.config.data_dir / "gravity_k2.json"
        if dest.exists():
            return dest
        payload = {
            "k2": 0.589,
            "k2_uncertainty": 0.150,
            "source": "Iess et al. (2012). Science, 337, 457-459.",
            "doi": "10.1126/science.1219631",
            "note": (
                "Tidal Love number k2 from Cassini gravity measurements. "
                "k2 = 0.589 +/- 0.150 implies a global subsurface liquid layer "
                "(water-ammonia ocean) beneath Titan's ice shell."
            ),
        }
        dest.write_text(json.dumps(payload, indent=2))
        logger.info("Created gravity_k2.json")
        return dest


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

class AcquisitionReport:
    """
    Summary of dataset acquisition status.

    Attributes
    ----------
    present:
        Datasets that are present and verified.
    downloaded:
        Datasets that were downloaded in this run.
    manual_required:
        Datasets that need manual download steps.
    partial:
        Shapefile sets where only some layers are present.
    failed:
        (name, error) tuples for download failures.
    corrupt:
        Datasets that are present but fail SHA-256 verification.
    """

    def __init__(self) -> None:
        self.present:         List[str]                 = []
        self.downloaded:      List[str]                 = []
        self.manual_required: List[str]                 = []
        self.partial:         List[Tuple[str, List[str]]] = []
        self.failed:          List[Tuple[str, str]]     = []
        self.corrupt:         List[str]                 = []

    @property
    def ready_count(self) -> int:
        return len(self.present) + len(self.downloaded)

    @property
    def total_count(self) -> int:
        return (len(self.present) + len(self.downloaded)
                + len(self.manual_required) + len(self.failed)
                + len(self.corrupt) + len(self.partial))

    def print_summary(self) -> None:
        """Print a coloured status table to stdout."""
        sep = "=" * 65
        print(f"\n{sep}")
        print("  TITAN PIPELINE -- DATA ACQUISITION STATUS")
        print(sep)
        print(f"  Ready      : {self.ready_count} / {self.total_count} datasets")
        if self.present:
            print(f"\n  [OK]  Present + verified:")
            for n in self.present:
                print(f"       {n}")
        if self.downloaded:
            print(f"\n  v  Downloaded this run:")
            for n in self.downloaded:
                print(f"       {n}")
        if self.partial:
            print(f"\n  ~  Partial (some layers only):")
            for n, layers in self.partial:
                print(f"       {n}: {', '.join(layers)}")
        if self.manual_required:
            print(f"\n  !  Manual download required:")
            for n in self.manual_required:
                print(f"       {n}")
        if self.corrupt:
            print(f"\n  [FAIL]  Corrupt (SHA-256 mismatch):")
            for n in self.corrupt:
                print(f"       {n}")
        if self.failed:
            print(f"\n  [FAIL]  Download failed:")
            for n, err in self.failed:
                print(f"       {n}: {err}")
        print(sep)

        # Pipeline readiness advice
        critical = {
            "sar_mosaic", "gtdr_east", "gtdr_west",
            "geomorphology_shapefiles"
        }
        missing_critical = critical & set(self.manual_required) & set(
            n for n, _ in self.failed
        )
        if missing_critical:
            print(
                f"\n  [WARNING]   Critical datasets missing: {missing_critical}\n"
                "      The pipeline will run with reduced accuracy.\n"
                "      Features derived from missing data will be NaN.\n"
            )
        else:
            print(
                "\n  Pipeline has sufficient data to proceed.\n"
                "  Run: python run_pipeline.py\n"
            )
        print(sep)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable status summary."""
        return {
            "present":          self.present,
            "downloaded":       self.downloaded,
            "manual_required":  self.manual_required,
            "partial":          [(n, ls) for n, ls in self.partial],
            "failed":           [(n, e) for n, e in self.failed],
            "corrupt":          self.corrupt,
            "ready":            self.ready_count,
            "total":            self.total_count,
        }

    def save(self, path: Path) -> None:
        """Save status report to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Acquisition report saved -> %s", path)
