#!/usr/bin/env python3
"""
prototype_f4_seasonal.py  --  PROTOTYPE  (branch: half-season-modelling)

Question: over the Cassini half-Titan-year (2004-2017 ~ 0.44 Titan yr, one
equinox crossing), how much does the methane-cycle feature f4 actually vary,
driven by the Jennings et al. (2019) T(phi, Y) surface-temperature model?

f4  (titan/features.py::_methane_cycle)  is a blend of:
    CIRS |dT/dlat| gradient   weight 0.50   <-- the ONLY seasonally-varying term
  + SAR channel density       weight 0.35   <-- static (geomorphic)
  + latitude prior exp(...)   weight 0.15   <-- static (+/-50 deg storm tracks)
so the seasonal swing of the FULL feature is 0.50 x (swing of the normalised
CIRS gradient).  This prototype isolates and quantifies that driver -- no new
data, only the existing Jennings model.

NB this is a magnitude check, not a thesis figure.  The Jennings model is
latitude-only, so f4's seasonal part is zonal (latitude bands).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from titan.atmospheric_profiles import jennings_surface_temperature

EQUINOX = 2009.61  # _TITAN_EQUINOX_YEAR

# Cassini at Titan; Jennings validity Y in [-4.9, +8.1] -> 2004.7 .. 2017.7
years = np.linspace(2004.7, 2017.7, 53)
lats  = np.linspace(-89.0, 89.0, 179)          # 1-deg latitude bins
dlat  = float(lats[1] - lats[0])

# T(lat, year) and meridional gradient |dT/dlat|  (mirrors _methane_cycle)
T    = np.array([[jennings_surface_temperature(la, yr) for la in lats]
                 for yr in years])              # shape (n_year, n_lat)
grad = np.abs(np.gradient(T, dlat, axis=1))     # K/deg

# Normalise by a FIXED reference (mission-wide max) so the seasonal swing is on
# a consistent cross-epoch scale.  (The pipeline normalises per-epoch; for a
# seasonal-magnitude check we need comparability across epochs.)
f4_cirs = grad / grad.max()                     # CIRS-driven f4 component in [0,1]

lat_prior = np.exp(-((np.abs(lats) - 50.0) / 25.0) ** 2)   # static term, context

bands = {
    "N. Polar (>60N)":       lats > 60,
    "N. Mid (40-60N)":       (lats >= 40) & (lats <= 60),
    "Equatorial (|lat|<30)": np.abs(lats) < 30,
    "S. Mid (40-60S)":       (lats <= -40) & (lats >= -60),
    "S. Polar (<60S)":       lats < -60,
}

# ---------------------------------------------------------------- plot
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.6))

epochs = {2005.0: "2005  (S summer, Ls~313)",
          2009.6: "2009.6 (equinox)",
          2017.0: "2017  (N summer, Ls~90)"}
for yr, lbl in epochs.items():
    i = int(np.argmin(np.abs(years - yr)))
    axA.plot(lats, f4_cirs[i], lw=2, label=lbl)
axA.plot(lats, lat_prior, "k--", alpha=0.45, lw=1, label="static lat prior (x0.15)")
axA.axvline(0, color="gray", lw=0.6)
axA.set_xlabel("Latitude (deg, S negative)")
axA.set_ylabel("f4 CIRS-gradient component (normalised)")
axA.set_title("(A) Methane-cycle driver pattern shifts across the equinox")
axA.legend(fontsize=8); axA.grid(alpha=0.3)

for name, mask in bands.items():
    axB.plot(years, f4_cirs[:, mask].mean(axis=1), lw=2, label=name)
axB.axvline(EQUINOX, color="gray", ls=":", lw=1.2)
axB.text(EQUINOX + 0.1, 0.02, "equinox 2009.6", color="gray", fontsize=8,
         transform=axB.get_xaxis_transform())
axB.set_xlabel("Calendar year (Cassini mission)")
axB.set_ylabel("Regional-mean f4 CIRS component (normalised)")
axB.set_title("(B) Seasonal f4 asymmetry over the Cassini half-year")
axB.legend(fontsize=8); axB.grid(alpha=0.3)

fig.suptitle("PROTOTYPE  --  Seasonal variation of methane-cycle feature f4 "
             "from the Jennings T(phi,Y) model", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "outputs/diagnostics/prototype_f4_seasonal.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}\n")

# ---------------------------------------------------------------- magnitude
print(f"{'band':24s}  f4_cirs min -> max   (seasonal swing)")
maxswing = 0.0
for name, mask in bands.items():
    s = f4_cirs[:, mask].mean(axis=1)
    sw = float(s.max() - s.min()); maxswing = max(maxswing, sw)
    print(f"{name:24s}  {s.min():.3f} -> {s.max():.3f}   ({sw:+.3f})")

full_swing = 0.50 * maxswing                       # static terms dilute 50%
dPH = full_swing * 0.15 * 6.0 / 11.0               # w4=0.15, lambda/(kappa+lambda)
print(f"\nLargest regional CIRS swing        : {maxswing:+.3f}")
print(f"-> full-feature f4 swing (x0.50)   : {full_swing:+.3f}")
print(f"-> resulting Delta P(H) (w4=0.15)  : {dPH:+.4f}")
print(f"   peak T-gradient over mission    : {grad.max():.4f} K/deg")
