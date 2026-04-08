"""
overlay_sites.py
================
Adds top-10 habitable site markers and mission lander positions to any
habitability map PNG produced by the pipeline.

Fully reproducible: all site coordinates are hard-coded from published
Titan cartography (Cassini RADAR and ISS mosaics); no pipeline outputs
or cached data are required.  Deleting outputs/ and data/processed/ and
rerunning the pipeline will regenerate identical overlays.

Marker symbols
--------------
  ▲  triangle (cyan)   = polar lake or sea shore
  ■  square   (amber)  = land site
  ★  star     (magenta)= mission lander (Huygens 2005; Dragonfly ~2034)

Epoch-aware display
-------------------
Markers are shown only for epochs at which each site appears in the top-10.
The epoch is inferred from the frame number (frame_NNN) or the filename
containing one of: past, lake_formation, present, near_future, future.

Integration with the pipeline
------------------------------
Call from generate_temporal_maps.py immediately after each fig.savefig():

    from overlay_sites import overlay_frame
    overlay_frame(fpath)        # annotates in-place (overwrites)
    # -- OR --
    overlay_frame(fpath, out=fpath.parent / "annotated" / fpath.name)

Call from run_pipeline.py after generate_paper_figures():

    from overlay_sites import overlay_png_files
    overlay_png_files(fig_dir.glob("fig1_posterior*.png"))

Standalone usage
----------------
    python overlay_sites.py --frames-dir outputs/temporal_maps/animation/frames
    python overlay_sites.py --single-frame outputs/present/figures/fig1_posterior.png
    python overlay_sites.py --frames-dir outputs/present/figures --no-labels
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ─── Canonical grid ────────────────────────────────────────────────────────────
NROWS, NCOLS = 1802, 3603   # must match pipeline canonical grid

# ─── Site catalogue ────────────────────────────────────────────────────────────
# Each entry: (short_label, lon_W_deg, lat_deg, site_type, [top10_epochs])
# site_type: 'lake'   -> triangle  ▲
#            'land'   -> square    ■
#            'lander' -> star      ★ (always shown)
#
# top10_epochs: list of canonical epoch strings for which this site is top-10
# Source: Bayesian posterior formula applied to Cassini-derived feature values
# (see thesis §Results, Table top10_past – top10_future)
#
# Coordinates from: Lopes et al. (2019) Nature Astronomy geomorphological map;
# Lorenz et al. (2021) PSJ Dragonfly landing site; Lebreton et al. (2005) Nature Huygens.

SITES: list[tuple] = [
    # ── Polar lakes / seas ───────────────────────────────────────────────────
    ("Kraken S",     310.0,  72.0,  "lake",
     ["PAST","LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Ligeia E",      82.0,  79.0,  "lake",
     ["PAST","LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Kraken N",     348.0,  80.0,  "lake",
     ["LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Punga",        339.0,  85.0,  "lake",
     ["LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Ontario",      179.0, -72.0,  "lake",
     ["LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Ligeia open",   79.0,  79.0,  "lake",
     ["LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    # ── Impact craters (top-10 at PAST epoch) ─────────────────────────────────
    ("Menrva",        87.3,  19.0,  "land",   ["PAST","FUTURE"]),
    ("Selk",         199.0,   7.0,  "land",   ["PAST","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Hano",         349.0, -38.6,  "land",   ["PAST"]),
    ("Sinlap",        16.0,  11.3,  "land",   ["PAST"]),
    ("Ksa",           65.6,  14.0,  "land",   ["PAST"]),
    ("Afekan",       200.5,  -1.4,  "land",   ["PAST"]),
    # ── Cryovolcanic candidates (top-10 at PAST epoch) ────────────────────────
    ("Hotei Regio",   75.0, -26.0,  "land",   ["PAST","PRESENT","NEAR_FUTURE"]),
    ("Sotra Facula", 144.5,   9.8,  "land",   ["PAST"]),
    # ── Equatorial dune seas (top-10 at PRESENT / FUTURE) ─────────────────────
    ("Belet",        250.0,   7.0,  "land",   ["PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Shangri-La",   155.0,  -5.0,  "land",   ["PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Fensal",        20.0,  15.0,  "land",   ["PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Aztlan",       100.0,  10.0,  "land",   ["FUTURE"]),
    # ── Mission landers (always shown) ────────────────────────────────────────
    # Huygens: Lebreton et al. (2005); Dragonfly: Lorenz et al. (2021)
    ("Huygens",      192.3, -10.6,  "lander",
     ["PAST","LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
    ("Dragonfly",    199.0,   7.0,  "lander",
     ["PAST","LAKE_FORMATION","PRESENT","NEAR_FUTURE","FUTURE"]),
]

# ─── Visual constants ──────────────────────────────────────────────────────────
COLOURS = {
    "lake":    (100, 200, 255, 230),   # cyan-blue
    "land":    (255, 210,  60, 230),   # amber
    "lander":  (255,  50, 255, 255),   # magenta
}
OUTLINE_COL = (0, 0, 0, 200)
LABEL_SHADOW = (0, 0, 0, 180)
LABEL_FG     = (255, 255, 255, 220)

# ─── Frame → epoch mapping ─────────────────────────────────────────────────────
_FRAME_EPOCH: dict[int, str] = {}
# Epoch time boundaries derived from generate_temporal_maps.py linspace + pauses
# Approximate frame ranges (0-indexed, 72 total):
for _i in range( 0,  4): _FRAME_EPOCH[_i] = "PAST"
for _i in range( 4, 16): _FRAME_EPOCH[_i] = "PAST"
for _i in range(16, 24): _FRAME_EPOCH[_i] = "LAKE_FORMATION"
for _i in range(24, 30): _FRAME_EPOCH[_i] = "PRESENT"
for _i in range(30, 44): _FRAME_EPOCH[_i] = "NEAR_FUTURE"
for _i in range(44, 55): _FRAME_EPOCH[_i] = "NEAR_FUTURE"
for _i in range(55, 72): _FRAME_EPOCH[_i] = "FUTURE"


def epoch_from_path(path: Path) -> str:
    """Infer epoch string from a PNG filename."""
    name = path.stem.lower()
    # Priority 1: explicit epoch string in filename
    for ep in ("past", "lake_formation", "present", "near_future", "future"):
        if ep in name:
            return ep.upper()
    # Priority 2: frame number
    if "frame_" in name:
        try:
            num = int(name.split("frame_")[1][:3])
            return _FRAME_EPOCH.get(num, "PRESENT")
        except (ValueError, IndexError):
            pass
    # Priority 3: posterior or figure names → present
    return "PRESENT"


# ─── Coordinate helpers ────────────────────────────────────────────────────────
def _lonlat_to_pixel(lon_W: float, lat: float,
                     img_w: int, img_h: int) -> tuple[int, int]:
    """Equirectangular west-positive lon/lat → pixel (x, y)."""
    x = int(round(lon_W / 360.0 * img_w)) % img_w
    y = max(0, min(img_h - 1, int(round((90.0 - lat) / 180.0 * img_h))))
    return x, y


# ─── Drawing helpers ───────────────────────────────────────────────────────────
def _font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _triangle(draw: ImageDraw.Draw, cx, cy, sz, fill, outline):
    h = int(sz * 0.866)
    pts = [(cx, cy - h*2//3), (cx - sz//2, cy + h//3), (cx + sz//2, cy + h//3)]
    draw.polygon(pts, fill=fill, outline=outline)


def _square(draw: ImageDraw.Draw, cx, cy, sz, fill, outline):
    s = sz // 2
    draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=fill, outline=outline)


def _star(draw: ImageDraw.Draw, cx, cy, sz, fill, outline):
    pts = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = sz if i % 2 == 0 else sz // 2
        pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
    draw.polygon(pts, fill=fill, outline=outline)


# ─── Core overlay function ────────────────────────────────────────────────────
def overlay_frame(
    src: Path | str,
    out: Path | str | None = None,
    marker_size: int = 14,
    show_labels: bool = True,
    epoch: str | None = None,
) -> Path:
    """
    Annotate a single habitability map PNG with site markers.

    Parameters
    ----------
    src:
        Input PNG path.
    out:
        Output PNG path.  If None, the source file is overwritten in-place.
    marker_size:
        Pixel radius of markers (default 14).
    show_labels:
        Whether to draw text labels next to markers.
    epoch:
        Override epoch detection.  One of PAST / LAKE_FORMATION /
        PRESENT / NEAR_FUTURE / FUTURE.  If None, epoch is inferred
        from the filename.

    Returns
    -------
    Path
        Path to the annotated file (same as ``out`` or ``src``).
    """
    if not _PIL_OK:
        raise ImportError("Pillow is required: pip install Pillow>=10.2")

    src = Path(src)
    out = Path(out) if out else src   # in-place by default

    img = Image.open(src).convert("RGBA")
    W, H = img.size

    epoch = epoch or epoch_from_path(src)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    font    = _font(max(9, marker_size - 3))

    drawn = []
    for label, lon_W, lat, stype, epochs in SITES:
        if epoch not in epochs:
            continue
        x, y  = _lonlat_to_pixel(lon_W, lat, W, H)
        fill   = COLOURS[stype]
        sz     = marker_size + (4 if stype == "lander" else 0)

        if stype == "lander":
            _star(draw, x, y, sz, fill, OUTLINE_COL)
        elif stype == "lake":
            _triangle(draw, x, y, sz, fill, OUTLINE_COL)
        else:
            _square(draw, x, y, sz, fill, OUTLINE_COL)

        if show_labels:
            tx, ty = x + sz // 2 + 3, y - sz // 2 - 1
            draw.text((tx + 1, ty + 1), label, font=font, fill=LABEL_SHADOW)
            draw.text((tx,     ty    ), label, font=font, fill=LABEL_FG)

        drawn.append(label)

    # ── Legend ─────────────────────────────────────────────────────────────
    lx, ly   = 12, H - 76
    leg_font = _font(max(8, marker_size - 4))
    legend   = [
        ("▲  Lake / sea shore",  "lake"),
        ("■  Land site",          "land"),
        ("★  Mission lander",     "lander"),
    ]
    bg = Image.new("RGBA", (200, 68), (0, 0, 0, 150))
    overlay.paste(bg, (lx - 4, ly - 4), bg)
    for i, (txt, st) in enumerate(legend):
        draw.text((lx, ly + i * 22), txt, font=leg_font,
                  fill=COLOURS[st][:3] + (255,))

    out.parent.mkdir(parents=True, exist_ok=True)
    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(out)
    return out


# ─── Batch helpers ────────────────────────────────────────────────────────────
def overlay_png_files(
    paths: Iterable[Path],
    out_dir: Path | None = None,
    **kwargs,
) -> list[Path]:
    """
    Annotate multiple PNG files.

    If out_dir is given, annotated copies are saved there (preserving filenames).
    If out_dir is None, files are annotated in-place.

    Returns list of output paths.
    """
    outputs = []
    for p in paths:
        p = Path(p)
        dest = (out_dir / p.name) if out_dir else p
        overlay_frame(p, dest, **kwargs)
        outputs.append(dest)
    return outputs


def overlay_directory(
    frames_dir: Path,
    out_dir: Path | None = None,
    **kwargs,
) -> int:
    """
    Annotate all PNG files in a directory.

    Parameters
    ----------
    frames_dir:
        Directory containing frame PNG files.
    out_dir:
        Output directory.  If None, files are annotated in-place.
    **kwargs:
        Forwarded to overlay_frame().

    Returns
    -------
    int
        Number of frames processed.
    """
    pngs = sorted(Path(frames_dir).glob("*.png"))
    if not pngs:
        print(f"overlay_sites: no PNGs found in {frames_dir}")
        return 0
    out = Path(out_dir) if out_dir else None
    results = overlay_png_files(pngs, out, **kwargs)
    print(f"overlay_sites: annotated {len(results)} frames → "
          f"{out or frames_dir}")
    return len(results)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frames-dir",   type=Path,
                   help="Directory of frame PNGs to annotate")
    p.add_argument("--single-frame", type=Path,
                   help="Single PNG to annotate")
    p.add_argument("--out-dir",      type=Path,
                   help="Output directory (default: annotate in-place)")
    p.add_argument("--marker-size",  type=int, default=14)
    p.add_argument("--epoch",        type=str, default=None,
                   help="Override epoch (PAST/LAKE_FORMATION/PRESENT/NEAR_FUTURE/FUTURE)")
    p.add_argument("--no-labels",    action="store_true",
                   help="Suppress text labels")
    args = p.parse_args()

    kw = dict(marker_size=args.marker_size,
              show_labels=not args.no_labels,
              epoch=args.epoch)

    if args.single_frame:
        dest = (args.out_dir / args.single_frame.name
                if args.out_dir else None)
        out = overlay_frame(args.single_frame, dest, **kw)
        print(f"Saved: {out}")
    elif args.frames_dir:
        overlay_directory(args.frames_dir, args.out_dir, **kw)
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
