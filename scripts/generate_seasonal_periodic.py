#!/usr/bin/env python3
"""
generate_seasonal_periodic.py  --  periodic-vs-Jennings comparison figure
(Appendix I, "Seasonal Sensitivity Within the Cassini Window"; Fig. I.2).

Companion to generate_seasonal_trend.py (which uses the Jennings 2019 fit).
The Jennings model fits a cosine in LATITUDE with parameters varying LINEARLY in
year (a+bY); the subsolar march is L_peak = -0.85 + 3.2*Y.  The authors state it
is valid only over Y in [-4.9, +8.1].  That implied march (3.2 deg/yr ~ 94 deg
per Titan year) exceeds the physical 26.7 deg obliquity, so the linear fit
captures the observed half-window trend but must not be extrapolated and cannot
show the solstice turnover.  This script adds an ILLUSTRATIVE PERIODIC model that replaces only the
subsolar march with the true astronomical subsolar latitude:

    L_sub(t) = 26.7 deg * sin(2*pi*(year - 2009.61) / 29.46)     (obliquity, period)
    T_per(L,t) = 93.53 * cos[(L - L_sub(t)) * 0.0029]            (Jennings cosine
                 form; amplitude & contrast held at their equinox values, since
                 the Jennings linear-in-Y amplitude/width were not fit for a
                 periodic march -- hence "illustrative", not a re-fit.)

Everything downstream (|dT/dlat| -> f4 -> Beta P(H)) is identical to the Jennings
prototype, so the only difference between the curves is linear vs sinusoidal
subsolar motion.  KEEPS generate_seasonal_trend.py / seasonal_ph_trend.png intact
for side-by-side discussion.  No new data.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root importable
from titan.atmospheric_profiles import jennings_surface_temperature as T_jennings

EQ, PRESENT_YEAR, PERIOD, OBLIQ = 2009.61, 2011.0, 29.46, 26.7
KAPPA, LAMBDA, W4, W_CIRS = 5.0, 6.0, 0.15, 0.50
PH_PER_DF4 = LAMBDA * W4 / (KAPPA + LAMBDA)

def T_periodic(L, yr):
    Lsub = OBLIQ * np.sin(2 * np.pi * (yr - EQ) / PERIOD)
    return 93.53 * np.cos((L - Lsub) * 0.0029)

lats = np.linspace(-89.0, 89.0, 179); dlat = float(lats[1] - lats[0])

def cirs_field(Tfun, yrs):
    T = np.array([[Tfun(la, yr) for la in lats] for yr in yrs])
    return np.abs(np.gradient(T, dlat, axis=1))

# Normalise BOTH models by the same present-epoch reference (Jennings 2011 max),
# so deviations are on one comparable scale.
D = cirs_field(T_jennings, [PRESENT_YEAR])[0].max()

def ph_curve(Tfun, lat_deg, ph_present, yrs):
    li = int(np.argmin(np.abs(lats - lat_deg)))
    g  = cirs_field(Tfun, yrs)[:, li] / D
    g0 = (cirs_field(Tfun, [PRESENT_YEAR])[0] / D)[li]
    return ph_present + PH_PER_DF4 * W_CIRS * (g - g0)

rows  = list(csv.DictReader(open("RANKINGS.csv")))
top10 = sorted(rows, key=lambda r: float(r["P(H)_present"]), reverse=True)[:10]
post  = np.load("outputs/present/inference/posterior_analytical.npy")
H, _  = post.shape
pix_lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
npolar_med = float(np.nanmedian(post[pix_lat > 60][np.isfinite(post[pix_lat > 60])]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# (a) mission window: top-10 sites, periodic model (shows solstice rollover)
yrs_m = np.linspace(2004.7, 2017.7, 53)
cmap = plt.cm.viridis(np.linspace(0, 0.9, 10))
for k, r in enumerate(top10):
    axA.plot(yrs_m, ph_curve(T_periodic, float(r["lat"]), float(r["P(H)_present"]), yrs_m),
             lw=1.8, color=cmap[k], label=f"#{k+1} {r['short_name']}")
_y0, _y1 = axA.get_ylim(); axA.set_ylim(_y0, _y1 + 0.15 * (_y1 - _y0))  # headroom: keep top labels clear of curves + title
axA.axvline(EQ, color="0.4", ls=":", lw=1.1); axA.axvline(2016.98, color="0.4", ls="-.", lw=1)
axA.text(EQ, 0.98, "equinox", color="0.4", fontsize=8, ha="center", va="top", transform=axA.get_xaxis_transform())
axA.text(2016.98, 0.98, "N-solstice", color="0.4", fontsize=8, ha="center", va="top", transform=axA.get_xaxis_transform())
axA.set_xlim(2004.7, 2017.7); axA.set_xlabel("Calendar year"); axA.set_ylabel(r"$P(H \mid \mathbf{f})$")
axA.set_title("(a) Periodic model, Cassini window: rollover at the solstice", fontsize=10)
axA.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9); axA.grid(alpha=0.3)

# (b) one Titan year: Jennings vs periodic for the N.Polar band -> divergence vs periodicity
yrs_e = np.linspace(2002.2, 2031.7, 120)
axB.plot(yrs_e, ph_curve(T_jennings, 75, npolar_med, yrs_e), color="#c0392b", ls="--", lw=2,
         label="Jennings 2019 (linear fit, outside validity)")
axB.plot(yrs_e, ph_curve(T_periodic, 75, npolar_med, yrs_e), color="#1f4e9c", lw=2,
         label="Periodic (illustrative)")
_y0, _y1 = axB.get_ylim(); axB.set_ylim(_y0, _y1 + 0.15 * (_y1 - _y0))  # headroom: keep top labels clear of curves + title
axB.axvspan(2004.7, 2017.7, color="0.85", alpha=0.6, label="Cassini data window")
for yr, lab in [(2002.24, "S-sol"), (2009.61, "equinox"), (2016.98, "N-sol"),
                (2024.34, "equinox"), (2031.71, "S-sol")]:
    axB.axvline(yr, color="0.6", ls=":", lw=0.8)
    axB.text(yr, 0.98, lab, color="0.5", fontsize=7, ha="center", va="top", transform=axB.get_xaxis_transform())
axB.set_xlim(2002.2, 2031.7); axB.set_xlabel("Calendar year (~one Titan year)")
axB.set_ylabel(r"N. Polar median $P(H \mid \mathbf{f})$")
axB.set_title("(b) One Titan year: Jennings linear fit (outside validity) vs periodic", fontsize=10)
axB.legend(fontsize=8, loc="lower right", framealpha=0.9); axB.grid(alpha=0.3)

fig.tight_layout()
out = "outputs/diagnostics/seasonal_ph_periodic.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

# magnitude / behaviour summary
tw = float(top10[0]["lat"]); twp = float(top10[0]["P(H)_present"])
cj = ph_curve(T_jennings, tw, twp, yrs_m); cp = ph_curve(T_periodic, tw, twp, yrs_m)
print(f"top-1 {top10[0]['short_name']} over mission: Jennings {cj[0]:.3f}->{cj[-1]:.3f}, "
      f"periodic {cp[0]:.3f}->{cp[-1]:.3f} (min {cp.min():.3f} near solstice)")
nb_j = ph_curve(T_jennings, 75, npolar_med, yrs_e); nb_p = ph_curve(T_periodic, 75, npolar_med, yrs_e)
print(f"N.Polar over a Titan year: Jennings range [{nb_j.min():.3f},{nb_j.max():.3f}] (diverges), "
      f"periodic [{nb_p.min():.3f},{nb_p.max():.3f}] (bounded/repeats)")
