"""
configs/site_catalogue.py
===========================
Single source of truth for all named Titan surface sites.

Every script that references site coordinates must import from here.
Coordinates are west-positive longitude (0-360°), matching all Cassini
raster products and the canonical pipeline grid.

Crater coordinates: Hedgepeth et al. (2020) Icarus 344:113664, Table 1.
Lake/sea coordinates: IAU Gazetteer + Cassini SAR/VIMS confirmation.
Mission sites: Lebreton et al. (2005); Lorenz et al. (2021) PSJ.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple


class Site(NamedTuple):
    """A named Titan surface site."""
    short_name: str
    full_name: str
    lon_W: float        # west-positive longitude, 0-360°
    lat: float           # planetocentric latitude, -90 to +90°
    site_type: str       # "lake", "sea", "crater", "land", "lander"


# -------------------------------------------------------------------------
# Master catalogue
# -------------------------------------------------------------------------
# Order: lakes/seas (north polar, south polar), craters, landmarks, mission
# sites.  This order has no computational significance.

SITES: List[Site] = [
    # -- North polar lakes and lacus --------
    # Coordinates are IAU Gazetteer centres (planetographic +West, 0-360),
    # reset June 2026 after the north-polar lake 180-deg registration fix in
    # shapefile_rasteriser._stereo_to_canonical.  Verified to sit on filled
    # lakes in the corrected polar_lakes raster (<=3 km).
    Site("Freeman",    "Freeman Lacus",     211.10, 73.60, "lake"),
    # Muggel replaces invalid "Oib" (not an IAU feature); Muggel is the largest
    # north lacus, nearest covered IAU lake to the old Oib sampling location.
    Site("Muggel",     "Müggel Lacus",      203.50, 84.44, "lake"),
    Site("Koitere",    "Koitere Lacus",      36.14, 79.40, "lake"),
    # Rwegura replaces "Hammar" (IAU Hammar is at 48.6N, outside Birch polar
    # coverage); nearest covered north lacus to the old Hammar location.
    Site("Rwegura",    "Rwegura Lacus",     105.20, 71.50, "lake"),
    Site("Mackay",     "Mackay Lacus",       97.53, 78.32, "lake"),
    # "Paxsi" dropped (IAU Paxsi is an equatorial ringed feature, not a lacus);
    # its slot is taken by Towada Lacus (defined below), the nearest real lake.
    Site("Neagh",      "Neagh Lacus",        32.16, 81.11, "lake"),
    Site("Bolsena",    "Bolsena Lacus",      10.28, 75.75, "lake"),
    Site("Logtak",     "Logtak Lacus",      226.10, 70.80, "lake"),
    # Arala replaces "Cardiel" (no filled lake at IAU Cardiel 206.5W/70.2N);
    # nearest covered north lacus to the old Cardiel location.
    Site("Arala",      "Arala Lacus",       124.90, 78.10, "lake"),
    Site("Uvs",        "Uvs Lacus",         245.70, 69.60, "lake"),
    Site("Waikare",    "Waikare Lacus",     126.00, 81.60, "lake"),
    Site("Ligeia E",   "Ligeia E shore",     82.0,  79.0, "lake"),
    Site("Jingpo",     "Jingpo Lacus",      336.00, 73.00, "lake"),
    # IAU Gazetteer (planetographic +West, 0-360; verified June 2026).
    # Used by the generate_temporal_maps animation candidate-site list.
    Site("Kivu",       "Kivu Lacus",        121.0,  87.0, "lake"),
    Site("Towada",     "Towada Lacus",      244.2,  71.4, "lake"),
    # -- North polar seas (maria) ------------------------------------------
    Site("Kraken S",   "Kraken Mare S",     310.0,  68.0, "sea"),
    Site("Kraken N",   "Kraken Mare N",     348.0,  80.0, "sea"),
    Site("Ligeia",     "Ligeia Mare",        78.0,  79.0, "sea"),
    Site("Punga",      "Punga Mare",        339.0,  85.0, "sea"),
    # -- South polar lakes -------------------------------------------------
    Site("Ontario",    "Ontario Lacus",     179.0, -72.0, "lake"),
    # IAU Gazetteer (planetographic +West, 0-360; verified June 2026).
    # Used by the generate_temporal_maps animation candidate-site list.
    Site("Crveno",     "Crveno Lacus",      184.91, -79.55, "lake"),
    Site("Sionascaig", "Sionascaig Lacus",  278.12, -41.52, "lake"),
    Site("Urmia",      "Urmia Lacus",       276.55, -39.27, "lake"),
    # -- Impact craters (Hedgepeth et al. 2020, Table 1) -------------------
    Site("Menrva",     "Menrva crater",      87.0,  20.0, "crater"),
    Site("Forseti",    "Forseti crater",     10.7,  25.9, "crater"),
    Site("Afekan",     "Afekan crater",     200.3,  26.0, "crater"),
    Site("Hano",       "Hano crater",       345.0,  40.5, "crater"),
    Site("Sinlap",     "Sinlap crater",      16.0,  11.5, "crater"),
    Site("Soi",        "Soi crater",        140.9,  24.3, "crater"),
    Site("Selk",       "Selk crater",       199.0,   7.0, "crater"),
    Site("Ksa",        "Ksa crater",         65.3,  13.7, "crater"),
    Site("Momoy",      "Momoy crater",       44.6,  11.7, "crater"),
    # -- Equatorial terrain regions ----------------------------------------
    Site("Belet",      "Belet dunes",       250.0,   5.0, "land"),
    Site("Shangri-La", "Shangri-La dunes",  155.0,  -5.0, "land"),
    Site("Xanadu",     "Xanadu",            100.0,  -5.0, "land"),
    Site("Fensal",     "Fensal dunes",       20.0,  15.0, "land"),
    Site("Aztlan",     "Aztlan",            100.0,  10.0, "land"),
    # -- Cryovolcanic candidate sites --------------------------------------
    Site("Hotei",      "Hotei Regio",        75.0, -26.0, "land"),
    Site("Sotra",      "Sotra Facula",      144.5,   9.8, "land"),
    # -- Mission landing sites ---------------------------------------------
    Site("Huygens",    "Huygens site",      192.3, -10.6, "lander"),
    Site("Dragonfly",  "Dragonfly (Selk)",  199.0,   7.0, "lander"),
    # -- Mountain range ----------------------------------------------------
    Site("Mithrim",    "Mithrim Montes",    236.0,   0.0, "land"),
]

# -------------------------------------------------------------------------
# Lookup helpers
# -------------------------------------------------------------------------

#: Dict mapping short_name -> Site for O(1) lookup.
SITE_DICT: Dict[str, Site] = {s.short_name: s for s in SITES}


def get_site(name: str) -> Site:
    """Look up a site by short_name.  Raises KeyError if not found."""
    return SITE_DICT[name]


def get_coords(name: str) -> Tuple[float, float]:
    """Return (lon_W, lat) for the given short_name."""
    s = SITE_DICT[name]
    return s.lon_W, s.lat


def sites_by_type(*types: str) -> List[Site]:
    """Return all sites matching any of the given type strings."""
    return [s for s in SITES if s.site_type in types]


# -------------------------------------------------------------------------
# Convenience lists (commonly used subsets)
# -------------------------------------------------------------------------

#: Sites used in thesis ranking tables (generate_rankings.py)
RANKING_SITES: List[str] = [
    "Freeman", "Muggel", "Koitere", "Ontario", "Rwegura", "Mackay",
    "Towada", "Neagh", "Bolsena", "Logtak", "Arala", "Uvs",
    "Waikare", "Ligeia E",
    "Selk", "Huygens",
    "Belet", "Shangri-La", "Xanadu",
]

#: Craters used in impact_melt_proxy (temporal_features.py)
#: Coordinates and diameters from Hedgepeth et al. (2020) Table 1.
#: (lon_W, lat, diameter_km, name)
IMPACT_MELT_CRATERS: List[Tuple[float, float, float, str]] = [
    (199.0,   7.0,  84.0, "Selk"),
    ( 65.3,  13.7,  45.0, "Ksa"),
    ( 16.0,  11.5,  88.0, "Sinlap"),
    (345.0,  40.5, 105.0, "Hano"),
    (200.3,  26.0, 115.0, "Afekan"),
    ( 87.0,  20.0, 400.0, "Menrva"),
    ( 10.7,  25.9, 125.0, "Forseti"),
    ( 44.6,  11.7,  40.0, "Momoy"),
]
