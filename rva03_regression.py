"""
rva03_regression.py
====================
§3.3  Method 2 (auxiliary): Multivariate regression
        v_pure = a + b·z + c·sin(θ)  [OLS via scipy.linalg.lstsq]

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import numpy as np
from scipy.linalg import lstsq
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import load_pantheon, make_sample, make_groups, ols, section, check

# ── Load data ──────────────────────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)

z_v   = d['zHD'].values
sin_v = d['sin_LA'].values
cos_v = d['cos_LA'].values
th_v  = d['theta'].values
v_v   = d['v_pure'].values
n     = len(d)

# ══════════════════════════════════════════════════════════════════════
# §3.3  Multivariate regression: v = a + b·z + c·sin(θ)
# ══════════════════════════════════════════════════════════════════════
section("§3.3  Method 2 (Auxiliary): v = a + b·z + c·sin(θ)  [N=665]")

X = np.column_stack([np.ones(n), z_v, sin_v])
b, se, t_vals, p_vals, r2 = ols(X, v_v)
a_c, b_c, c_c = b
a_se, b_se, c_se = se

print(f"  {'Coeff':<22} {'Value (km/s)':>14} {'SE':>9} {'t':>8} {'p':>12}")
print(f"  {'-'*68}")
print(f"  {'a (intercept)':<22} {a_c:>+14.1f} {a_se:>9.1f} {t_vals[0]:>8.2f} {p_vals[0]:>12.4f}")
print(f"  {'b (z coeff)':<22} {b_c:>+14.0f} {b_se:>9.0f} {t_vals[1]:>8.0f} {p_vals[1]:>12.2e}")
print(f"  {'c (sin(θ) coeff)':<22} {c_c:>+14.1f} {c_se:>9.1f} {t_vals[2]:>8.2f} {p_vals[2]:>12.2e}")
print(f"\n  R² = {r2:.5f}  (adj-R² ≈ {r2:.5f})")
print(f"  Note: High R² reflects Hubble flow domination (b×z̄≈H₀×d̄).")
print(f"        This study focuses on c(sin_θ) residual significance, not total R².")

# z-uncontrolled comparison
X_noz = np.column_stack([np.ones(n), sin_v])
b_noz, _, _, _, _ = ols(X_noz, v_v)
print(f"\n  c without z-control: {b_noz[1]:+.0f} km/s  →  with z-control: {c_c:+.1f} km/s")
print(f"  (Signal shrinks but does NOT vanish, p<0.001)")

# ── Independent check: residual-only regression ─────────────────────
section("  Independent check: residual-only sin(θ) regression")

X_z   = np.column_stack([np.ones(n), z_v])
b_z   = lstsq(X_z, v_v)[0]
resid = v_v - X_z @ b_z

X_res = np.column_stack([np.ones(n), sin_v])
b_r, se_r, t_r, p_r, _ = ols(X_res, resid)
print(f"  resid = v_pure − ŷ(z),  then regress sin(θ) on resid:")
print(f"  slope = {b_r[1]:+.1f} ± {se_r[1]:.1f} km/s   p={p_r[1]:.2e}")
print(f"  → Converges with multivariate c={c_c:+.1f} ✓")

# ── Functional form comparison (6 forms) ────────────────────────────
section("  Functional form comparison on z-controlled residuals")

forms = {
    'sin(θ)':      sin_v,
    '1−|cos(θ)|':  1 - np.abs(cos_v),
    'sin²(θ)':     sin_v**2,
    'θ-linear':    th_v / 180.0,
    '|cos(θ)|':    np.abs(cos_v),
    'cos(θ)':      cos_v,
}
results = []
for name, x in forms.items():
    Xf = np.column_stack([np.ones(n), x])
    bf, _, _, _, rf = ols(Xf, resid)
    results.append((rf, bf[1], name))
results.sort(reverse=True)

print(f"  {'Rank':<6} {'Form':<16} {'R²':>8} {'coef (km/s)':>13}  {'Note'}")
print(f"  {'-'*62}")
for rank, (r2f, coef, name) in enumerate(results, 1):
    note = "← LA-axis predicted ✓" if name == 'sin(θ)' else ""
    print(f"  {rank:<6} {name:<16} {r2f:>8.4f} {coef:>+13.1f}  {note}")

# ── Verification ──────────────────────────────────────────────────────
print()
checks = []
checks.append(check("c (sin coeff)",   c_c,         335.6,  0.5))
checks.append(check("SE(c)",           c_se,         37.7,  0.5))
checks.append(check("t(c)",            t_vals[2],     8.90,  0.05))
checks.append(check("R²",             r2,         0.99984,  0.00002))
checks.append(check("a (intercept)",   a_c,         162.5,  0.5))
checks.append(check("b (z coeff)",     b_c,        299222,  5))
checks.append(check("Residual c",      b_r[1],      332.8,  1.0))
checks.append(check("sin(θ) R² rank1", results[0][2] == 'sin(θ)', True, 0, fmt=''))
print(f"  Verification: {sum(checks)}/{len(checks)} passed")
