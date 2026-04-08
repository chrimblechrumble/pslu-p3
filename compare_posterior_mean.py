import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Compare posterior columns at the seam ────────────────────────────────
# Run for all five anchor epochs
epochs = ["past", "lake_formation", "present", "near_future", "future"]
print(f"{'Epoch':<16} {'col-0 mean':>12} {'col-3602 mean':>14} {'|diff|':>10} {'max |diff|':>12}")
print("-" * 68)
for epoch in epochs:
    p = Path(f"outputs/{epoch}/inference/posterior_mean.npy")
    if not p.exists():
        print(f"  {epoch:<14}: NOT FOUND")
        continue
    arr = np.load(p)
    c0   = arr[:, 0]
    c3602 = arr[:, 3602]
    diff  = np.abs(c0 - c3602)
    print(f"  {epoch:<14}  {c0.mean():>12.5f}  {c3602.mean():>14.5f}  {diff.mean():>10.5f}  {diff.max():>12.5f}")

# ── 2. Visual check: plot the seam difference for "present" as a profile ────
p = Path("outputs/present/inference/posterior_mean.npy")
if p.exists():
    arr = np.load(p)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: column-mean profile across all longitudes near the seam
    col_means = arr.mean(axis=0)
    axes[0].plot(col_means, lw=0.6)
    axes[0].axvline(0,    color='red', lw=1.2, label='col 0 (0°W)')
    axes[0].axvline(3602, color='red', lw=1.2, linestyle='--', label='col 3602 (360°W)')
    axes[0].set_xlabel("Column (pixel)"); axes[0].set_ylabel("Mean P(H)")
    axes[0].set_title("Column-mean posterior — look for a step at col 0/3602")
    axes[0].legend()

    # Right: zoom in on ±30 columns around the seam
    left_cols  = arr[:, :30].mean(axis=0)    # cols 0-29
    right_cols = arr[:, 3573:].mean(axis=0)  # cols 3573-3602
    ax = axes[1]
    ax.plot(range(30),        left_cols,  label="cols 0–29  (0°W side)")
    ax.plot(range(30-1, -1, -1), right_cols, label="cols 3602–3573  (360°W side)",
            linestyle='--')
    ax.set_xlabel("Distance from seam (columns)")
    ax.set_ylabel("Mean P(H)")
    ax.set_title("Zoom: 30 columns either side of the 0°/360°W seam")
    ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/seam_check.png", dpi=120)
    print("\nPlot saved → outputs/seam_check.png")
