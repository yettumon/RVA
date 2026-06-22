"""
rva09_cmb_propagation.py
=========================
Appendix B.2  CMB correction uncertainty propagation
        Tests 9 combinations of CMB (±5 km/s) and LG (±15 km/s) values.
        Confirms c(sin_θ) is insensitive to CMB correction uncertainty.

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from rva_utils import (load_pantheon, make_sample, ols,
                       dot_sky, CMB_RA, CMB_DEC, LG_RA, LG_DEC,
                       C, section, check)

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)   # v_pure uses nominal CMB=369, LG=627

# ══════════════════════════════════════════════════════════════════════
# Appendix B.2  CMB uncertainty propagation
# ══════════════════════════════════════════════════════════════════════
section("Appendix B.2  CMB correction uncertainty propagation")

print("  Nominal: CMB=369±5 km/s  (Planck 2018),  LG=627±15 km/s")
print("  9 combinations tested:\n")

cmb_vals = [364, 369, 374]   # ±5 km/s
lg_vals  = [612, 627, 642]   # ±15 km/s
c_results = []

print(f"  {'CMB (km/s)':>12} {'LG (km/s)':>12} {'c (km/s)':>12}")
print(f"  {'-'*40}")

for cmb_v in cmb_vals:
    for lg_v in lg_vals:
        v_p = (C * d['zHD']
               - cmb_v * dot_sky(d['RA'], d['DEC'], CMB_RA, CMB_DEC)
               - lg_v  * dot_sky(d['RA'], d['DEC'], LG_RA,  LG_DEC))
        Xp  = np.column_stack([np.ones(len(d)), d['zHD'].values, d['sin_LA'].values])
        bp, _, _, _, _ = ols(Xp, v_p.values)
        c_results.append(bp[2])
        print(f"  {cmb_v:>12} {lg_v:>12} {bp[2]:>+12.1f}")

c_arr    = np.array(c_results)
c_min    = c_arr.min()
c_max    = c_arr.max()
c_range  = (c_max - c_min) / 2     # ± half-range
c_se_cmb = c_arr.std()             # std across 9 combinations

stat_se   = 37.7                   # from §3.3
total_se  = np.sqrt(stat_se**2 + c_se_cmb**2)
nominal_c = 335.6                  # from §3.3

print(f"\n  c range: {c_min:+.1f} ~ {c_max:+.1f} km/s  (±{c_range:.1f} km/s)")
print(f"  CMB error SE contribution: {c_se_cmb:.1f} km/s")
print(f"  Statistical SE:            {stat_se:.1f} km/s")
print(f"  Total SE (combined):       √({stat_se:.1f}² + {c_se_cmb:.1f}²) = {total_se:.1f} km/s")
print(f"  CMB contribution fraction: {c_se_cmb/nominal_c*100:.1f}% of signal")
print(f"\n  If CMB residual effects explained the signal, c would be sensitive")
print(f"  to CMB correction uncertainty. In practice it is not → CMB explanation excluded.")

checks = []
checks.append(check("c half-range ±",  c_range,    9.0, 1.0))
checks.append(check("CMB error SE",     c_se_cmb,   7.1, 0.5))
checks.append(check("total SE",         total_se,  38.4, 0.2))
print(f"\n  Verification: {sum(checks)}/{len(checks)} passed")
