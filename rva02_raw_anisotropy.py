"""
rva02_raw_anisotropy.py
========================
§3.1  Raw recession velocity anisotropy
        Welch t-test, Mann-Whitney U, Bootstrap 95% CI

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, section, check, SEED

np.random.seed(SEED)
N_BOOT = 10000

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)
axis, perp, _ = make_groups(d)

# ══════════════════════════════════════════════════════════════════════
# §3.1  Raw anisotropy
# ══════════════════════════════════════════════════════════════════════
section("§3.1  Raw recession velocity anisotropy")

print(f"  {'Group':<14} {'N':>5}  {'mean v_pure (km/s)':>20}  {'z̄':>7}")
print(f"  {'-'*52}")
print(f"  {'Axis (θ<45°)':<14} {len(axis):>5}  {axis['v_pure'].mean():>20,.0f}  {axis['zHD'].mean():>7.3f}")
print(f"  {'Perp (45~135°)':<14} {len(perp):>5}  {perp['v_pure'].mean():>20,.0f}  {perp['zHD'].mean():>7.3f}")

dv_raw = perp['v_pure'].mean() - axis['v_pure'].mean()
print(f"\n  Δv_raw (perp − axis) = {dv_raw:+,.1f} km/s")

# Welch t-test (primary)
t_w, p_welch = stats.ttest_ind(perp['v_pure'], axis['v_pure'], equal_var=False)
print(f"\n  Welch t-test:    t={t_w:.2f},  p={p_welch:.2e}")

# Mann-Whitney U (nonparametric robustness)
_, p_mw = mannwhitneyu(perp['v_pure'], axis['v_pure'], alternative='two-sided')
print(f"  Mann-Whitney U:  p={p_mw:.2e}")

# z-distribution check (motivates §3.3, 3.4)
_, p_zdiff = stats.ttest_ind(perp['zHD'], axis['zHD'], equal_var=False)
print(f"  z-distribution p={p_zdiff:.2e}  → groups differ in z → z-control needed")

# Bootstrap 95% CI on Δv_raw (independent-sample bootstrap)
print(f"\n  Bootstrap 95% CI on Δv_raw (N={N_BOOT:,}, independent-sample):")
boot_dv = np.array([
    np.random.choice(perp['v_pure'].values, len(perp), replace=True).mean() -
    np.random.choice(axis['v_pure'].values, len(axis), replace=True).mean()
    for _ in range(N_BOOT)
])
ci_lo, ci_hi = np.percentile(boot_dv, [2.5, 97.5])
print(f"  Bootstrap CI: [{ci_lo:+,.0f}, {ci_hi:+,.0f}] km/s")
print(f"  Note: Paper cites [+11,847, +12,339] km/s (†not independently reproduced).")
print(f"        That CI likely used a different bootstrap variant (e.g. block or paired).")
print(f"        The present independent-sample bootstrap gives a wider, conservative CI.")

# ── Verification ──────────────────────────────────────────────────────
print()
checks = []
checks.append(check("Δv_raw",   dv_raw,   12093, 5,   fmt='+,.1f'))
checks.append(check("Welch p",  p_welch,  4.77e-9, 1e-9, fmt='.2e'))
checks.append(check("MW-U p",   p_mw,     4.22e-8, 1e-8, fmt='.2e'))
print(f"  Verification: {sum(checks)}/{len(checks)} passed")
