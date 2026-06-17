# Bayesian Formula Verification

Confirms that `run_pipeline.py` implements **exactly** the Beta-conjugate posterior
defined in `thesis/methods.tex` — same weights, same priors, same formula.

Last verified: June 2026 (post north-polar lake registration fix).

## Formula correspondence

`run_pipeline.py` analytical block (≈ lines 526–559) vs `methods.tex` Eqs. (261–275):

| methods.tex | run_pipeline.py | Match |
|---|---|---|
| α₀ = μ₀κ,  β₀ = (1−μ₀)κ | `alpha0 = mu0*kappa`, `beta0 = (1-mu0)*kappa` | ✓ |
| α_post = α₀ + λ Σᵢ wᵢ fᵢ | `alpha_p = alpha0 + lam*w_sum` | ✓ |
| β_post = β₀ + λ Σᵢ wᵢ(1−fᵢ) | `beta_p = beta0 + lam*(1-w_sum)` | ✓ ¹ |
| P(H) = α_post / (α_post+β_post) | `alpha_p / ab_sum` | ✓ |
| α_post + β_post = κ + λ = 11 | `ab_sum` ≡ 11 at every pixel | ✓ |
| σ = √[αβ / ((α+β)²(α+β+1))] | `a_std = sqrt(...)` | ✓ |
| 95% equal-tailed CI (2.5/97.5 pct) | `beta.ppf(0.025/0.975, α, β)` | ✓ |
| missing data → impute fᵢ = μᵢ | `np.where(finite, layer, m_dict[name])` | ✓ |

¹ The code's `(1 − w_sum)` equals the thesis's `Σᵢ wᵢ(1−fᵢ)` because **Σᵢ wᵢ = 1.0**
(enforced by `BayesianPriorConfig.validate()`), so the identity is exact, not approximate.

## Parameters (config = thesis)

- κ = `cfg.priors.beta_concentration` = **5.0**  (thesis κ = 5)
- λ = `cfg.priors.likelihood_sharpness` = **6.0**  (thesis λ = 6)
- μ₀ = Σᵢ wᵢμᵢ = **0.3306 ≈ 0.331**;  α₀ = 1.653, β₀ = 3.347
- All 8 weights and prior means come from `configs/temporal_config.get_prior_set()`
  and match `methods.tex` Table 2.1 element-for-element.

## Worked example (methods.tex)

Towada Lacus (244.2°W, 71.4°N), site-averaged Σᵢ wᵢ fᵢ = 0.775:

```
α_post = 1.653 + 6 × 0.775 = 6.303
β_post = 3.347 + 6 × (1 − 0.775) = 4.697
P(H)   = 6.303 / 11 = 0.573        95% CI [0.29, 0.83]
```

The formula, the methods.tex printed numbers, **and the actual pipeline posterior
sampled at Towada** all agree: **P(H) = 0.573** (to 3 dp).

## Single source of truth (fixed June 2026)

`run_pipeline.py` now writes the analytical Beta posterior to **`posterior_mean.npy`**
(formerly the sklearn/GNB classifier output, which is unbounded and diverged from the
model), and persists the analytical **std** and **95% CI** to `posterior_std.npy`,
`hdi_low.npy`, `hdi_high.npy` (formerly all-NaN). The GNB result is retained only in
`feature_importances.json` as a validation diagnostic. Every persisted posterior layer
therefore corresponds to the `methods.tex` formula.

## Reproduce

```bash
python - <<'PY'
import numpy as np
from configs.pipeline_config import BayesianPriorConfig
from configs.temporal_config import TemporalMode, get_prior_set
pc = BayesianPriorConfig(); K, L = pc.beta_concentration, pc.likelihood_sharpness
ps = get_prior_set(TemporalMode.PRESENT)
w = dict(zip(ps.feature_names, ps.weights)); m = dict(zip(ps.feature_names, ps.prior_means))
mu0 = sum(w[k]*m[k] for k in w)
wsum = 0.775                                  # Towada site-averaged weighted feature sum
a = mu0*K + L*wsum; b = (1-mu0)*K + L*(1-wsum)
print(f"kappa={K} lambda={L} mu0={mu0:.4f} sum_w={sum(w.values()):.4f}")
print(f"alpha={a:.3f} beta={b:.3f} P(H)={a/(a+b):.3f}")   # -> 6.303 4.697 0.573
PY
```
