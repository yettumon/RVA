"""
rva04_distance_matching.py
===========================
§3.4  Method 3 (primary): Distance matching  |Δz| < 0.01
§3.5  Signal decomposition

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, greedy_match, section, check, DZ_MAX

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)
axis, perp, _ = make_groups(d)

# ══════════════════════════════════════════════════════════════════════
# §3.4  Distance matching  |Δz| < 0.01, 1:1 greedy nearest-neighbour
# ══════════════════════════════════════════════════════════════════════
section(f"§3.4  Method 3 (Primary): Distance Matching  |Δz|<{DZ_MAX}")

pairs = greedy_match(axis, perp, dz_max=DZ_MAX)
N     = len(pairs)
dv_mean = pairs['dv'].mean()
dv_se   = pairs['dv'].std() / np.sqrt(N)
dv_med  = pairs['dv'].median()
dz_mean = pairs['dz'].mean()
dz_max  = pairs['dz'].max()

t_val, p_match  = ttest_1samp(pairs['dv'].values, 0)
_, p_zbal        = ttest_ind(pairs['z_ax'], pairs['z_pp'], equal_var=False)
_, p_wilcox      = wilcoxon(pairs['dv'].values)

print(f"  Matched pairs (axis 100% matched):  N = {N}")
print(f"  |Δz| mean / max:                    {dz_mean:.5f} / {dz_max:.5f}")
print(f"  z-balance t-test:                   p = {p_zbal:.4f}  "
      f"{'✅ balanced' if p_zbal > 0.05 else '❌ imbalanced'}")
print(f"\n  {'Statistic':<30} {'Value'}")
print(f"  {'-'*50}")
print(f"  {'mean Δv (perp − axis)':<30} {dv_mean:+.1f} ± {dv_se:.1f} km/s")
print(f"  {'median Δv':<30} {dv_med:+.1f} km/s")
print(f"  {'t-test (μ=0)':<30} t={t_val:.2f},  p={p_match:.2e}")
print(f"  {'Wilcoxon (nonparametric)':<30} p={p_wilcox:.2e}")

# ── Verification ──────────────────────────────────────────────────────
checks = []
checks.append(check("N pairs",       N,          205,    0,   fmt='.0f'))
checks.append(check("|Δz| mean",     dz_mean,  0.00097, 0.00005))
checks.append(check("|Δz| max",      dz_max,   0.00414, 0.00005))
checks.append(check("z-balance p",   p_zbal,    0.9296,  0.005))
checks.append(check("mean Δv",       dv_mean,   380.6,   1.0))
checks.append(check("SE Δv",         dv_se,      39.5,   1.0))
checks.append(check("median Δv",     dv_med,    438.4,   2.0))
checks.append(check("t",             t_val,       9.62,   0.05))
print(f"\n  Verification: {sum(checks)}/{len(checks)} passed")

# ══════════════════════════════════════════════════════════════════════
# §3.5  Signal decomposition
# ══════════════════════════════════════════════════════════════════════
section("§3.5  Signal decomposition")

dv_raw       = perp['v_pure'].mean() - axis['v_pure'].mean()
dv_direction = dv_mean
dv_z_diff    = dv_raw - dv_direction

print(f"  Raw signal:          Δv = {dv_raw:+,.0f} km/s  (100%)")
print(f"  z-distribution diff:  ~  {dv_z_diff:+,.0f} km/s  ({dv_z_diff/dv_raw*100:.0f}% — majority)")
print(f"  Pure directional:     ~  {dv_direction:+.0f} km/s  ({dv_direction/dv_raw*100:.1f}% ← core result)")
print(f"\n  The pure directional component ({dv_direction:.0f} km/s) is small in absolute")
print(f"  value but extremely significant: p<10⁻¹⁶.")

checks2 = []
checks2.append(check("z-diff component", dv_z_diff, 11712, 50))
print(f"  Verification: {sum(checks2)}/{len(checks2)} passed")
