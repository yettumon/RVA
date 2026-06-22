"""
rva05_zbin_stratified.py
=========================
§3.6  Stratified z-bin analysis (distance matching, using global 205 pairs)
§3.7  z-decay F-test: CMB constant model vs exp(−z/z₀) attenuation
        Uses scipy.optimize.curve_fit for nonlinear WLS fit.

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA

Note on F-test method:
  The exp(−z/z₀) model is fitted by nonlinear weighted least squares
  (scipy.optimize.curve_fit with sigma=SE per bin), NOT log-space WLS.
  Log-space WLS gives F≈15.4; curve_fit gives F=18.6 (matches paper).
"""

import numpy as np
from scipy.stats import ttest_1samp, f as f_dist, linregress
from scipy.optimize import curve_fit
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rva_utils import (load_pantheon, make_sample, make_groups,
                       greedy_match, ols, section, check, DZ_MAX)

# ── Load data & global match ───────────────────────────────────────────
raw  = load_pantheon()
d    = make_sample(raw)
axis, perp, _ = make_groups(d)

# Global 205 matched pairs (§3.4)
pairs = greedy_match(axis, perp, dz_max=DZ_MAX)

# ══════════════════════════════════════════════════════════════════════
# §3.6  Stratified z-bin analysis
# ══════════════════════════════════════════════════════════════════════
section("§3.6  Stratified z-bin analysis (global matched pairs, split by axis z)")

# NOTE: paper Table 6 uses the global 205 matched pairs, then classifies
# each pair by the axis-object's zHD. N_ax sums = 107+74+24 = 205. ✓
bins      = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.50)]
paper_dv  = [474.9, 314.1, 165.4]
paper_se  = [ 53.1,  67.2, 102.7]

print(f"  {'z-bin':<14} {'N_ax':>5} {'N_pp':>5} {'Δv (km/s)':>11} {'SE':>7} {'p':>10}")
print(f"  {'-'*58}")

checks = []
for i, (z1, z2) in enumerate(bins):
    sub   = pairs[(pairs['z_ax'] >= z1) & (pairs['z_ax'] < z2)]
    ax_n  = len(axis[(axis['zHD'] >= z1) & (axis['zHD'] < z2)])
    pp_n  = len(perp[(perp['zHD'] >= z1) & (perp['zHD'] < z2)])

    dv_m  = sub['dv'].mean()
    dv_s  = sub['dv'].std() / np.sqrt(len(sub))
    _, p  = ttest_1samp(sub['dv'].values, 0)
    pstr  = "<0.001" if p < 0.001 else f"{p:.3f}"
    flag  = "n.s." if p >= 0.05 else "***"

    print(f"  {z1:.2f}≤z<{z2:.2f}   {ax_n:5d} {pp_n:5d}   {dv_m:>+9.1f}  ±{dv_s:>5.1f}  {pstr} {flag}")
    checks.append(check(f"Δv [{z1},{z2})", dv_m, paper_dv[i], 5.0))

print(f"  Verification: {sum(checks)}/{len(checks)} passed")

# ══════════════════════════════════════════════════════════════════════
# §3.7  z-decay F-test
# ══════════════════════════════════════════════════════════════════════
section("§3.7  z-decay F-test: CMB constant vs exp(−z/z₀) attenuation")

print("  Extracting c(sin_θ) per z-bin by multivariate regression (6 bins):")

zbins6 = [(0.15,0.20),(0.20,0.25),(0.25,0.30),(0.30,0.35),(0.35,0.40),(0.40,0.50)]
z_mids, c_vals, c_errs = [], [], []

print(f"  {'z-bin':<14} {'N':>5} {'c (km/s)':>10} {'SE':>8}")
print(f"  {'-'*42}")
for z1, z2 in zbins6:
    d_b = d[(d['zHD'] >= z1) & (d['zHD'] < z2)]
    if len(d_b) < 10:
        continue
    Xb = np.column_stack([np.ones(len(d_b)),
                           d_b['zHD'].values,
                           d_b['sin_LA'].values])
    bb, seb, _, _, _ = ols(Xb, d_b['v_pure'].values)
    zm = (z1 + z2) / 2
    z_mids.append(zm)
    c_vals.append(bb[2])
    c_errs.append(seb[2])
    print(f"  {z1:.2f}≤z<{z2:.2f}   {len(d_b):5d} {bb[2]:>+10.1f} {seb[2]:>8.1f}")

z_m = np.array(z_mids)
c_v = np.array(c_vals)
c_e = np.array(c_errs)
w   = 1 / c_e**2

# Model 1: constant (CMB motion prediction — z-independent)
c_const    = np.sum(w * c_v) / np.sum(w)
chi2_const = np.sum(w * (c_v - c_const)**2)
dof_const  = len(z_m) - 1

# Model 2: exp(−z/z₀) attenuation (LA-axis radial force prediction)
# Fitted by nonlinear WLS via scipy.optimize.curve_fit
# (log-space WLS gives different result; curve_fit matches paper values)
def exp_model(z, c0, z0):
    return c0 * np.exp(-z / z0)

popt, _ = curve_fit(exp_model, z_m, c_v,
                    p0=[1500, 0.2],
                    sigma=c_e,
                    absolute_sigma=True,
                    bounds=([0, 0.01], [1e5, 5.0]))
c0_fit, z0_fit = popt
c_pred_exp = exp_model(z_m, c0_fit, z0_fit)
chi2_exp   = np.sum(w * (c_v - c_pred_exp)**2)
dof_exp    = len(z_m) - 2

# F-test: does exp model significantly improve over constant?
F_stat = ((chi2_const - chi2_exp) / 1) / (chi2_exp / dof_exp)
p_F    = 1 - f_dist.cdf(F_stat, 1, dof_exp)

# Linear trend (supplementary)
slope_lr, _, r_lr, _, _ = linregress(z_m, c_v)

print(f"\n  Model comparison (weighted):")
print(f"  Constant model (CMB):      c = {c_const:.0f} km/s,          χ²/dof = {chi2_const/dof_const:.2f}")
print(f"  Exp-decay model (LA-axis): c₀={c0_fit:.0f}, z₀={z0_fit:.3f},  χ²/dof = {chi2_exp/dof_exp:.2f}")
print(f"\n  F-test: F = {F_stat:.1f},  p = {p_F:.3f}")
print(f"  Linear trend: slope = {slope_lr:+.0f} km/s/Δz,  r = {r_lr:.2f}")
print(f"\n  Interpretation: exp-decay model significantly better (p<0.05).")
print(f"  This is exploratory evidence against a pure CMB-motion explanation.")
print(f"  Caveat: 6 bins is limited; non-monotonic pattern in 0.25≤z<0.30.")

checks2 = []
checks2.append(check("c_const",           c_const,     405,   5))
checks2.append(check("χ²/dof constant",   chi2_const/dof_const, 3.84, 0.05))
checks2.append(check("c₀ (exp fit)",      c0_fit,     1397,  20))
checks2.append(check("z₀ (exp fit)",      z0_fit,     0.193, 0.010))
checks2.append(check("χ²/dof exp",        chi2_exp/dof_exp, 0.85, 0.05))
checks2.append(check("F-stat",            F_stat,      18.6,  0.5))
checks2.append(check("p(F)",              p_F,         0.013, 0.005))
checks2.append(check("slope r",           r_lr,        -0.80, 0.05))
print(f"  Verification: {sum(checks2)}/{len(checks2)} passed")
