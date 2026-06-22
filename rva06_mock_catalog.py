"""
rva06_mock_catalog.py
======================
§3.8  Mock Catalog verification — isotropic null hypothesis rejection
        N=2,000 random direction shuffles on z-controlled residuals

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
from scipy.linalg import lstsq
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, section, check, SEED, THETA_AXIS_MAX, THETA_ANTI_MIN

np.random.seed(SEED)
N_MOCK = 2000

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)
axis, perp, _ = make_groups(d)

# ══════════════════════════════════════════════════════════════════════
# §3.8  Mock Catalog verification
# ══════════════════════════════════════════════════════════════════════
section(f"§3.8  Mock Catalog Verification  (N={N_MOCK:,} isotropic shuffles)")

# Step 1: compute z-only residuals (fully removes z-distribution effect)
z_v  = d['zHD'].values
v_v  = d['v_pure'].values
X_z  = np.column_stack([np.ones(len(d)), z_v])
b_z  = lstsq(X_z, v_v)[0]
resid = v_v - X_z @ b_z

print(f"  Residuals: v_pure − ŷ(z),  mean={resid.mean():+.1f}, std={resid.std():.1f} km/s")

# Step 2: observed test statistic (residual Δv between groups)
theta_v = d['theta'].values
is_ax   = theta_v < THETA_AXIS_MAX
is_pp   = (theta_v >= THETA_AXIS_MAX) & (theta_v <= THETA_ANTI_MIN)
obs_stat = resid[is_pp].mean() - resid[is_ax].mean()

print(f"  Observed residual Δv (perp − axis): {obs_stat:+.1f} km/s")

# Step 3: shuffle directions N_MOCK times (keep v_pure fixed)
mock_stats = np.empty(N_MOCK)
for i in range(N_MOCK):
    shuffled      = resid.copy()
    np.random.shuffle(shuffled)
    mock_stats[i] = shuffled[is_pp].mean() - shuffled[is_ax].mean()

mock_mean  = mock_stats.mean()
mock_std   = mock_stats.std()
z_score    = (obs_stat - mock_mean) / mock_std
n_exceed   = int(np.sum(mock_stats >= obs_stat))
p_upper    = 3 / N_MOCK   # Rule of 3 upper bound when n_exceed=0

print(f"  Mock distribution: mean={mock_mean:+.1f}, std={mock_std:.1f} km/s")
print(f"  z-score:           {z_score:.1f}σ")
print(f"  Exceedances:       {n_exceed}/{N_MOCK}")
if n_exceed == 0:
    print(f"  p-value upper bound (Rule of 3): p < {p_upper:.4f}")
else:
    print(f"  p-value (empirical): p = {n_exceed/N_MOCK:.4f}")

print(f"\n  Interpretation: the observed directional signal is non-random")
print(f"  at {z_score:.1f}σ under the isotropic null hypothesis.")
print(f"  Note: this confirms non-randomness, NOT physical vs systematic origin.")
print(f"        For the latter, see §5.1 (CF3↔Pantheon+ cross-validation).")

# ── Verification ──────────────────────────────────────────────────────
print()
checks = []
checks.append(check("obs residual Δv",  obs_stat,   157.4,  3.0))
checks.append(check("mock std",          mock_std,    30.4,  2.0))
checks.append(check("z-score",           z_score,      5.2,  0.3))
checks.append(check("exceedances",       n_exceed,     0.0,  0.0, fmt='.0f'))
print(f"  Verification: {sum(checks)}/{len(checks)} passed")
