#!/usr/bin/env python3
"""
generate_seasonal_trend.py  --  thesis figure for the seasonal appendix
(Appendix I, "Seasonal Sensitivity Within the Cassini Window"; Fig. I.1).

Seasonal P(H) over the Cassini half-Titan-year (2004-2017), with the variation
driven ONLY by the methane-cycle feature f4 (the one seasonally-varying feature);
the other seven features are held at their present values.

Model (faithful to the pipeline):
  * Present epoch anchors at year_ce = 2011.0 (titan/preprocessing.py).
  * Seasonal surface temperature from Jennings et al. (2019) T(L,Y).
  * f4 = 0.50*CIRS|dT/dlat| + 0.35*channel + 0.15*latprior; only the CIRS term
    varies with year.  Beta posterior => dP(H) = lambda*w4/(kappa+lambda)*df4
    = 0.0818*df4, with df4 = 0.50*(CIRSnorm(lat,year) - CIRSnorm(lat,2011)).

Writes outputs/diagnostics/seasonal_ph_trend.png (caption supplied in LaTeX).
No new data: uses only titan.atmospheric_profiles.jennings_surface_temperature.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from titan.atmospheric_profiles import jennings_surface_temperature

EQUINOX, PRESENT_YEAR = 2009.61, 2011.0
KAPPA, LAMBDA, W4, W_CIRS = 5.0, 6.0, 0.15, 0.50
PH_PER_DF4 = LAMBDA * W4 / (KAPPA + LAMBDA)        # 0.0818

years = np.linspace(2004.7, 2017.7, 53)
lats  = np.linspace(-89.0, 89.0, 179)
dlat  = float(lats[1] - lats[0])

T    = np.array([[jennings_surface_temperature(la, yr) for la in lats] for yr in years])
grad = np.abs(np.gradient(T, dlat, axis=1))
D    = grad[int(np.argmin(np.abs(years - PRESENT_YEAR)))].max()    # present-epoch normaliser
cirs = grad / D

def _cirs(lat_deg, yr):
    return cirs[int(np.argmin(np.abs(years - yr))), int(np.argmin(np.abs(lats - lat_deg)))]

def ph_curve(lat_deg, ph_present):
    base = _cirs(lat_deg, PRESENT_YEAR)
    return np.array([ph_present + PH_PER_DF4 * W_CIRS * (_cirs(lat_deg, yr) - base) for yr in years])

# ---- inputs ----
rows  = list(csv.DictReader(open("RANKINGS.csv")))
top10 = sorted(rows, key=lambda r: float(r["P(H)_present"]), reverse=True)[:10]
post  = np.load("outputs/present/inference/posterior_analytical.npy")
H, Wd = post.shape
pix_lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
bands = {"N. Polar ($>$60°N)": (pix_lat > 60, 75),
         "N. Mid (40–60°N)": ((pix_lat >= 40) & (pix_lat <= 60), 50),
         "Equatorial ($|$lat$|<$30°)": (np.abs(pix_lat) < 30, 0),
         "S. Mid (40–60°S)": ((pix_lat <= -40) & (pix_lat >= -60), -50),
         "S. Polar ($<$60°S)": (pix_lat < -60, -75)}

# ---- figure ----
plt.rcParams.update({"font.size": 10})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

def _decorate(ax):
    ax.axvline(PRESENT_YEAR, color="0.4", ls="--", lw=1)
    ax.axvline(EQUINOX, color="0.4", ls=":", lw=1.1)
    ax.set_xlabel("Calendar year (Cassini mission)")
    ax.grid(alpha=0.3)
    ax.set_xlim(2004.7, 2017.7)

cmap = plt.cm.viridis(np.linspace(0, 0.9, 10))
for k, r in enumerate(top10):
    axA.plot(years, ph_curve(float(r["lat"]), float(r["P(H)_present"])), lw=1.8, color=cmap[k],
             label=f"#{k+1} {r['short_name']}")
_decorate(axA)
axA.text(EQUINOX, 1.005, "equinox", color="0.4", fontsize=8, ha="center", transform=axA.get_xaxis_transform())
axA.set_ylabel(r"$P(H \mid \mathbf{f})$")
axA.set_title("(a) Top-10 present sites (all north-polar lakes)", fontsize=10)
axA.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)

band_cols = ["#1f4e9c", "#5b9bd5", "#7f7f7f", "#e08020", "#c0392b"]
for (name, (mask, rep)), col in zip(bands.items(), band_cols):
    sub = post[mask, :]; med = float(np.nanmedian(sub[np.isfinite(sub)]))
    axB.plot(years, ph_curve(rep, med), lw=2, color=col, label=name)
_decorate(axB)
axB.text(EQUINOX, 1.005, "equinox", color="0.4", fontsize=8, ha="center", transform=axB.get_xaxis_transform())
axB.set_ylabel(r"Median $P(H \mid \mathbf{f})$")
axB.set_title("(b) Latitude bands: modest northern decline toward summer", fontsize=10)
axB.legend(fontsize=8, loc="center right", framealpha=0.9)

fig.tight_layout()
out = "outputs/diagnostics/seasonal_ph_trend.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")

# magnitude summary (for the caption / verification)
tw = ph_curve(float(top10[0]["lat"]), float(top10[0]["P(H)_present"]))
print(f"top-1 {top10[0]['short_name']}: {tw[0]:.3f} (2004) -> {tw[-1]:.3f} (2017), swing {tw.max()-tw.min():+.3f}")
for name, (mask, rep) in bands.items():
    sub = post[mask, :]; med = float(np.nanmedian(sub[np.isfinite(sub)])); c = ph_curve(rep, med)
    print(f"  {name:28s}: {c[0]:.3f} -> {c[-1]:.3f}  (swing {c.max()-c.min():+.3f})")
