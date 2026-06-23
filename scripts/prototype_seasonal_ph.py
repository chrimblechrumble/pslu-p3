#!/usr/bin/env python3
"""
prototype_seasonal_ph.py  --  PROTOTYPE  (branch: half-season-modelling)

P(H) vs Cassini-mission-time for the top-10 present sites and for latitude
bands, with the seasonal variation driven ONLY by the methane-cycle feature f4
(the one seasonally-varying feature); the other seven features are held at their
present values.

Model (faithful to the pipeline):
  * Present epoch anchors at year_ce = 2011.0 (titan/preprocessing.py).
  * f4 = 0.50*CIRS|dT/dlat| + 0.35*channel + 0.15*latprior; only the CIRS term
    varies with year, via titan.atmospheric_profiles.jennings_surface_temperature.
  * Beta posterior mean => a change df4 shifts P(H) by lambda*w4*df4/(kappa+lambda)
    = 6*0.15*df4/11 = 0.0818*df4, with df4 = 0.50*(CIRSnorm(year)-CIRSnorm(2011)).
  => P(H)(site,year) = P(H)_present(site) + 0.0818 * 0.50 *
                       (CIRSnorm(lat,year) - CIRSnorm(lat,2011)).

Normalisation: CIRS gradient is normalised by the 2011 (present-epoch) map max,
so the present anchor matches how the pipeline normalised f4. No new data.
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

# T(lat,year) -> |dT/dlat|, normalised by the PRESENT (2011) map max
T    = np.array([[jennings_surface_temperature(la, yr) for la in lats] for yr in years])
grad = np.abs(np.gradient(T, dlat, axis=1))
present_idx = int(np.argmin(np.abs(years - PRESENT_YEAR)))
D = grad[present_idx].max()                        # present-epoch normaliser
cirs = grad / D                                    # CIRS-norm(lat, year)

def cirs_at(lat_deg, yr):
    li = int(np.argmin(np.abs(lats - lat_deg)))
    yi = int(np.argmin(np.abs(years - yr)))
    return cirs[yi, li]

def ph_curve(lat_deg, ph_present):
    base = cirs_at(lat_deg, PRESENT_YEAR)
    return np.array([ph_present + PH_PER_DF4 * W_CIRS * (cirs_at(lat_deg, yr) - base)
                     for yr in years])

# ----- top-10 present sites from RANKINGS.csv -----
rows = list(csv.DictReader(open("RANKINGS.csv")))
top10 = sorted(rows, key=lambda r: float(r["P(H)_present"]), reverse=True)[:10]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 6.0))

cmap = plt.cm.viridis(np.linspace(0, 0.92, 10))
for k, r in enumerate(top10):
    lat = float(r["lat"]); php = float(r["P(H)_present"])
    axA.plot(years, ph_curve(lat, php), lw=2, color=cmap[k],
             label=f"#{k+1} {r['short_name']} ({lat:+.0f}°)")
axA.axvline(PRESENT_YEAR, color="gray", ls="--", lw=1)
axA.text(PRESENT_YEAR+0.1, axA.get_ylim()[0], " present anchor 2011", color="gray", fontsize=8)
axA.axvline(EQUINOX, color="gray", ls=":", lw=1)
axA.set_xlabel("Calendar year (Cassini mission)"); axA.set_ylabel("P(H | f)")
axA.set_title("(A) Top-10 present sites: seasonal P(H) (f4 dynamic, others static)")
axA.legend(fontsize=7, ncol=2, loc="upper left"); axA.grid(alpha=0.3)

# ----- latitude bands: present-posterior median baseline + seasonal df4 -----
post = np.load("outputs/present/inference/posterior_analytical.npy")
H, W = post.shape
pix_lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
bands = {"N. Polar (>60N)": (pix_lat > 60, 75), "N. Mid (40-60N)": ((pix_lat>=40)&(pix_lat<=60), 50),
         "Equatorial (|lat|<30)": (np.abs(pix_lat)<30, 0), "S. Mid (40-60S)": ((pix_lat<=-40)&(pix_lat>=-60), -50),
         "S. Polar (<60S)": (pix_lat < -60, -75)}
for name, (mask, rep_lat) in bands.items():
    sub = post[mask, :]; med = float(np.nanmedian(sub[np.isfinite(sub)]))
    axB.plot(years, ph_curve(rep_lat, med), lw=2, label=f"{name}  (present {med:.2f})")
axB.axvline(PRESENT_YEAR, color="gray", ls="--", lw=1); axB.axvline(EQUINOX, color="gray", ls=":", lw=1)
axB.set_xlabel("Calendar year (Cassini mission)"); axB.set_ylabel("Median P(H | f)")
axB.set_title("(B) Latitude bands: seasonal median P(H)")
axB.legend(fontsize=8); axB.grid(alpha=0.3)

fig.suptitle("PROTOTYPE  --  Seasonal P(H) over the Cassini half-Titan-year "
             "(only f4 varies; anchored to present 2011)", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "outputs/diagnostics/prototype_seasonal_ph.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}\n")

# magnitude report
print(f"{'site':12s} lat   P(H)present  P(H)2004 -> 2017   swing")
for k, r in enumerate(top10):
    lat = float(r["lat"]); php = float(r["P(H)_present"]); c = ph_curve(lat, php)
    print(f"#{k+1:<2d}{r['short_name']:9s} {lat:+5.0f}  {php:.3f}      {c[0]:.3f} -> {c[-1]:.3f}   ({c.max()-c.min():+.3f})")
