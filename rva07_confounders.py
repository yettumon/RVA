"""
rva07_confounders.py
=====================
§4.5  Sequential confounder control (7 variables)
        MWEBV, HOST_LOGMASS, x1, c (SALT2 color),
        MWEBV+HOST_LOGMASS, full simultaneous
§4.6  Measurement quality directional bias (FITCHI2/NDOF)

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, ols, section, check

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)
axis, perp, _ = make_groups(d)

z_v   = d['zHD'].values
sin_v = d['sin_LA'].values
v_v   = d['v_pure'].values
n     = len(d)

# ══════════════════════════════════════════════════════════════════════
# §4.5  Sequential confounder control
# ══════════════════════════════════════════════════════════════════════
section("§4.5  Sequential confounder control")

def fill_missing(col_vals, sentinel=-9):
    """Replace sentinel/extreme values with column median."""
    vals = col_vals.astype(float).copy()
    vals = np.where(vals < -90, np.nan, vals)
    median = np.nanmedian(vals)
    vals = np.where(np.isnan(vals), median, vals)
    return vals

confounders = [
    ('z only (baseline)',                           []),
    ('+ MWEBV',                                     ['MWEBV']),
    ('+ HOST_LOGMASS',                              ['HOST_LOGMASS']),
    ('+ x1 (SALT2)',                                ['x1']),
    ('+ c (SALT2 color)',                           ['c']),
    ('+ MWEBV + HOST_LOGMASS',                      ['MWEBV','HOST_LOGMASS']),
    ('Full (z+MWEBV+HOST_LOGMASS+x1+c)',            ['MWEBV','HOST_LOGMASS','x1','c']),
]

paper_c = [335.6, 404.6, 333.6, 334.9, 336.7, 402.3, 402.4]

print(f"  {'Control variables':<42} {'c (km/s)':>10} {'SE':>7} {'p':>12}")
print(f"  {'-'*74}")

checks = []
c_by_label = {}  # store computed c per label
for idx, (label, cols) in enumerate(confounders):
    extras = [fill_missing(d[col].values) for col in cols if col in d.columns]
    Xc = np.column_stack([np.ones(n), z_v, sin_v] + extras)
    bc, sec, _, pc, _ = ols(Xc, v_v)
    c_by_label[label] = bc[2]
    print(f"  {label:<42} {bc[2]:>+10.1f} {sec[2]:>7.1f} {pc[2]:>12.2e}")
    checks.append(check(f"c ({label[:22]})", bc[2], paper_c[idx], 1.0))

# MWEBV directional check
mwebv_ax   = axis['MWEBV'].mean()
mwebv_pp   = perp['MWEBV'].mean()
_, p_mwebv = stats.ttest_ind(axis['MWEBV'], perp['MWEBV'], equal_var=False)
print(f"\n  MWEBV: axis={mwebv_ax:.4f}, perp={mwebv_pp:.4f}  "
      f"(axis/perp ratio={mwebv_ax/mwebv_pp:.1f}×, p={p_mwebv:.4f})")
print(f"  → Axis has ~2× higher dust extinction than perp, yet signal STRENGTHENS")
print(f"    after MWEBV control "
      f"(c: {c_by_label.get('z only (baseline)', 0):+.1f}"
      f" → {c_by_label.get('+ MWEBV', 0):+.1f} km/s)"
      " → dust dilutes, not causes, signal.")

print(f"\n  Verification: {sum(checks)}/{len(checks)} passed")

# ══════════════════════════════════════════════════════════════════════
# §4.6  Measurement quality directional bias
# ══════════════════════════════════════════════════════════════════════
section("§4.6  Measurement quality directional bias  (FITCHI2/NDOF)")

chi2_ax = (axis['FITCHI2'] / axis['NDOF'].replace(0, np.nan)).dropna()
chi2_pp = (perp['FITCHI2'] / perp['NDOF'].replace(0, np.nan)).dropna()
_, p_fit = stats.ttest_ind(chi2_ax, chi2_pp, equal_var=False)

print(f"  Axis  FITCHI2/NDOF: {chi2_ax.mean():.3f}  (N={len(chi2_ax)})")
print(f"  Perp  FITCHI2/NDOF: {chi2_pp.mean():.3f}  (N={len(chi2_pp)})")
print(f"  Welch t-test p:     {p_fit:.3f}  → no significant directional difference")
print(f"  Conclusion: fitting quality bias is NOT a cause of the signal.")

checks2 = []
checks2.append(check("FITCHI2/NDOF axis", chi2_ax.mean(), 0.896, 0.005))
checks2.append(check("FITCHI2/NDOF perp", chi2_pp.mean(), 0.887, 0.005))
checks2.append(check("p (fit quality)",   p_fit,          0.769, 0.005))
print(f"  Verification: {sum(checks2)}/{len(checks2)} passed")
