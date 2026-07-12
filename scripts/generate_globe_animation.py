#!/usr/bin/env python3
"""
generate_globe_animation.py -- Titan habitability maps as a rotating globe.

Wraps the multi-epoch habitability field
(``outputs/temporal_maps/titan_temporal_habitability.nc``) onto a sphere by
orthographic reprojection and renders spinning-globe animations for a
presentation intro slide:

  * a per-epoch spin (mp4 + gif) for each of the five named epochs
    (Past/LHB, Lake formation, Present, Near future, Future), and
  * a 60-second temporal sweep (mp4) that morphs Past -> Future while rotating.

Method
------
For each output pixel inside the disc we invert the orthographic projection
(centred on a chosen viewing latitude ``viewlat``) to a (lat, lon) on the unit
sphere, then sample the equirectangular P(H) texture there.  The texture is the
per-epoch ``P_habitable`` slice from the NetCDF cube, coloured exactly like the
thesis maps (matplotlib ``plasma``, vmin=0.10, vmax=0.75, NaN -> ``#0a0a2a``).
A cosine limb-darkening term gives the sphere its shading, and the globe is
composited onto a solid background -- so there is *no* GIF transparency, which
is what caused earlier flashing / white-interior / border artefacts.

Encoding uses ffmpeg.  MP4 is H.264 (yuv420p).  GIFs use a single shared palette
(``palettegen``/``paletteuse``) to avoid per-frame palette flashing, and hold the
final frame for ``--pause`` seconds (via ``tpad``) before looping forever.

Usage
-----
    .venv/bin/python scripts/generate_globe_animation.py [VIEWLAT] [--pause S] [--fps N]

    VIEWLAT   viewing latitude in degrees (default 30; e.g. -30 for the southern
              hemisphere, 0 for equatorial, 90 for a north-polar view).

Outputs land in ``team/intro_globe/``.  Requires numpy, xarray, matplotlib and
Pillow (available in ``.venv``) plus ``ffmpeg`` on PATH.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import colormaps, colors
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "outputs/temporal_maps/titan_temporal_habitability.nc"
DEST = ROOT / "team/intro_globe"

VMIN, VMAX = 0.10, 0.75          # colour scale, matched to the thesis maps
NAVY = (10, 10, 42)              # #0a0a2a, used for NaN texels
BG = (0, 0, 0)                   # solid frame background (no transparency)
NAMED = [("past", -3.5), ("lake", -1.0), ("present", 0.0),
         ("nearfuture", 0.25), ("future", 5.9)]

_ds = xr.open_dataset(NC)
EPOCHS = _ds["epoch_Gya"].values
CMAP, NORM = colormaps["plasma"], colors.Normalize(VMIN, VMAX, clip=True)


def to_rgb(idx, step=1):
    """Equirectangular RGB texture for epoch index ``idx`` (optionally downsampled)."""
    f = _ds["P_habitable"].isel(epoch_Gya=idx).values[::step, ::step]
    rgb = (CMAP(NORM(f))[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(f)] = NAVY
    return rgb


def make_renderer(size, viewlat, shade=True, feather=1.5, bg=BG):
    """Return frame(rgb, lon0) that orthographically projects a texture onto a globe."""
    p1 = np.radians(viewlat)
    s1, c1 = np.sin(p1), np.cos(p1)
    ys, xs = np.mgrid[0:size, 0:size]
    X = (xs - size / 2) / (size / 2 - 1)
    Y = (size / 2 - ys) / (size / 2 - 1)
    r2 = X * X + Y * Y
    inside = r2 <= 1.0
    Z = np.sqrt(np.clip(1 - r2, 0, 1))                       # cos(angular distance)
    lat = np.degrees(np.arcsin(np.clip(Z * s1 + Y * c1, -1, 1)))    # inverse orthographic
    base_lon = np.degrees(np.arctan2(X, Z * c1 - Y * s1))
    bright = (0.45 + 0.55 * Z)[..., None] if shade else 1.0
    alpha = (np.clip((1.0 - np.sqrt(r2)) * (size / 2) / feather, 0, 1) * inside)[..., None]
    bgarr = np.array(bg, np.float32)

    def frame(rgb, lon0):
        H, W, _ = rgb.shape
        row = np.clip(((90.0 - lat) / 180.0 * (H - 1)).astype(int), 0, H - 1)
        col = np.clip((((base_lon + lon0) % 360.0) / 360.0 * (W - 1)).astype(int), 0, W - 1)
        g = rgb[row, col].astype(np.float32) * bright
        return np.clip(g * alpha + bgarr * (1 - alpha), 0, 255).astype(np.uint8)

    return frame


def idx_of(gya):
    return int(np.argmin(np.abs(EPOCHS - gya)))


def encode_dir(frame_dir, base, fps, pause):
    """PNG frame dir -> mp4, plus a looping gif that holds the last frame ``pause`` s."""
    src = f"{frame_dir}/f_%04d.png"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", src,
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(DEST / f"{base}.mp4")], capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", "palettegen=stats_mode=full",
                    f"{frame_dir}/pal.png"], capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", src, "-i", f"{frame_dir}/pal.png",
                    "-lavfi", f"[0:v]tpad=stop_mode=clone:stop_duration={pause}[v];"
                              f"[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(DEST / f"{base}.gif")], capture_output=True)


def main():
    ap = argparse.ArgumentParser(description="Titan habitability rotating-globe animations.")
    ap.add_argument("viewlat", nargs="?", type=float, default=30.0,
                    help="viewing latitude in degrees (default 30; e.g. -30, 0, 90)")
    ap.add_argument("--pause", type=float, default=5.0,
                    help="seconds to hold the final gif frame before looping (default 5)")
    ap.add_argument("--fps", type=int, default=24, help="fps for per-epoch spins (default 24)")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    tag = f"lat{int(args.viewlat)}"
    render = make_renderer(600, args.viewlat)

    # ---- per-epoch spins: mp4 + looping gif with an end-pause ----
    for name, gya in NAMED:
        rgb = to_rgb(idx_of(gya))
        d = tempfile.mkdtemp(prefix=f"globe_{name}_")
        for k, ang in enumerate(range(0, 360, 5)):          # 72 frames, one full turn
            Image.fromarray(render(rgb, ang), "RGB").save(f"{d}/f_{k:04d}.png")
        encode_dir(d, f"titan_spin_{name}_{tag}", args.fps, args.pause)
        shutil.rmtree(d, ignore_errors=True)
        print(f"  titan_spin_{name}_{tag}.mp4 / .gif")

    # ---- 60 s temporal sweep (mp4): morph Past->Future over 3 rotations ----
    fps, dur, rots, size = 20, 60, 3, 600
    n = fps * dur
    tex = [to_rgb(i, step=3) for i in range(len(EPOCHS))]    # downsampled textures held in RAM
    out = DEST / f"titan_temporal_{tag}_60s.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}",
         "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ne = len(EPOCHS)
    for k in range(n):
        t = k / (n - 1) * (ne - 1)
        i0 = int(t); i1 = min(i0 + 1, ne - 1); w = t - i0
        blended = (tex[i0].astype(np.float32) * (1 - w) + tex[i1].astype(np.float32) * w).astype(np.uint8)
        proc.stdin.write(render(blended, k * (360.0 * rots / n)).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"  titan_temporal_{tag}_60s.mp4 ({dur}s)")


if __name__ == "__main__":
    main()
