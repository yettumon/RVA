"""
rva01_sample_correction.py
===========================
§2.1  Data & sample selection
§2.2  CMB + LG peculiar velocity correction
§2.3  Angle calculation & group classification

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, dot_sky, section, check
from rva_utils import C, CMB_RA, CMB_DEC, CMB_V, LG_RA, LG_DEC, LG_V

# ══════════════════════════════════════════════════════════════════════
# §2.1  Data load & sample selection
# ══════════════════════════════════════════════════════════════════════
section("§2.1  Data load & sample selection")

raw = load_pantheon()
print(f"  Total catalogue: N={len(raw)}")

d = make_sample(raw)
print(f"  Analysis sample (IS_CALIBRATOR=0, 0.15<zHD<0.50): N={len(d)}")

# ══════════════════════════════════════════════════════════════════════
# §2.2  Peculiar velocity correction
# ══════════════════════════════════════════════════════════════════════
section("§2.2  Peculiar velocity correction")

print(f"  v_pure = c·zHD − CMB_V·cos(θ_CMB) − LG_V·cos(θ_LG)")
print(f"  CMB:  {CMB_V} km/s  RA={CMB_RA}°  Dec={CMB_DEC}°  (Planck 2018)")
print(f"  LG:   {LG_V} km/s  RA={LG_RA}°   Dec={LG_DEC}°")

# LG contribution breakdown (paper §2.2: ~0.1% of total signal)
v_no_lg  = C * d['zHD'] - CMB_V * dot_sky(d['RA'], d['DEC'], CMB_RA, CMB_DEC)
lg_delta = (d['v_pure'] - v_no_lg)
print(f"  LG correction per-object mean: {lg_delta.mean():+.1f} km/s")
print(f"  (Paper: LG contribution to Δv ~+14 km/s, ~0.1% of total signal)")

# ══════════════════════════════════════════════════════════════════════
# §2.3  Angle calculation & group classification
# ══════════════════════════════════════════════════════════════════════
section("§2.3  Angle calculation & group classification")

axis, perp, anti = make_groups(d)

print(f"  {'Group':<28} {'N':>5}  {'mean θ':>8}  {'z̄':>7}")
print(f"  {'-'*52}")
print(f"  {'Principal axis (θ < 45°)':<28} {len(axis):>5}  {axis['theta'].mean():>7.1f}°  {axis['zHD'].mean():>7.3f}")
print(f"  {'Perpendicular (45°≤θ≤135°)':<28} {len(perp):>5}  {perp['theta'].mean():>7.1f}°  {perp['zHD'].mean():>7.3f}")
print(f"  {'Anti-axis (θ > 135°)':<28} {len(anti):>5}  {anti['theta'].mean():>7.1f}°  {anti['zHD'].mean():>7.3f}")

checks = []
checks.append(check("N total",    len(d),    665, 0, fmt='.0f'))
checks.append(check("N axis",     len(axis), 205, 0, fmt='.0f'))
checks.append(check("N perp",     len(perp), 415, 0, fmt='.0f'))
checks.append(check("N anti",     len(anti),  45, 0, fmt='.0f'))
checks.append(check("mean θ axis", axis['theta'].mean(), 18.3, 0.1))
checks.append(check("mean θ perp", perp['theta'].mean(), 81.0, 0.1))
checks.append(check("z̄ axis",     axis['zHD'].mean(),   0.259, 0.001))
checks.append(check("z̄ perp",     perp['zHD'].mean(),   0.299, 0.001))

print(f"\n  Verification: {sum(checks)}/{len(checks)} passed")
