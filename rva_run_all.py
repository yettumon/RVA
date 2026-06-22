"""
rva_run_all.py
===============
Run all RVA analysis scripts in sequence and print a final summary.

Usage:
    python rva_run_all.py

All scripts must be in the same directory as this file.
Pantheon_SH0ES.dat must be accessible (see rva_utils.py DATA_PATHS).

Paper: Kim, GunSik (Yettumon) 2026c — RVA
GitHub: github.com/yettumon/RVA
"""

import subprocess, sys, os, time

SCRIPTS = [
    ("rva01_sample_correction.py",  "§2.1-2.3  Data, correction, groups"),
    ("rva02_raw_anisotropy.py",     "§3.1      Raw anisotropy"),
    ("rva03_regression.py",         "§3.3      Multivariate regression (Method 2)"),
    ("rva04_distance_matching.py",  "§3.4-3.5  Distance matching + decomposition (Method 3)"),
    ("rva05_zbin_stratified.py",    "§3.6-3.7  z-bin stratified + F-test"),
    ("rva06_mock_catalog.py",       "§3.8      Mock Catalog verification"),
    ("rva07_confounders.py",        "§4.5-4.6  Confounder control + fit quality"),
    ("rva08_loso.py",               "App B.1   LOSO"),
    ("rva09_cmb_propagation.py",    "App B.2   CMB uncertainty propagation"),
]

DIR = os.path.dirname(os.path.abspath(__file__))
PASS_MARKER = "✅"
FAIL_MARKER = "❌"

print("=" * 70)
print("  RVA Full Analysis Pipeline")
print("  Kim, GunSik (Yettumon) 2026c | github.com/yettumon/RVA")
print("=" * 70)

summary = []
total_start = time.time()

for script, desc in SCRIPTS:
    path = os.path.join(DIR, script)
    print(f"\n{'─'*70}")
    print(f"  ▶  {script}")
    print(f"     {desc}")
    print('─'*70)
    t0  = time.time()
    res = subprocess.run([sys.executable, path], capture_output=False, text=True)
    dt  = time.time() - t0
    ok  = res.returncode == 0
    summary.append((script, desc, ok, dt))

elapsed = time.time() - total_start

print(f"\n{'='*70}")
print(f"  PIPELINE SUMMARY  ({elapsed:.1f}s total)")
print('='*70)
print(f"  {'Script':<38} {'Status':>8} {'Time':>7}")
print(f"  {'-'*56}")
for script, desc, ok, dt in summary:
    status = "✅ OK" if ok else "❌ FAIL"
    print(f"  {script:<38} {status:>8} {dt:>6.1f}s")

n_pass = sum(ok for _, _, ok, _ in summary)
n_total = len(summary)
print(f"\n  Result: {n_pass}/{n_total} scripts completed successfully.")
if n_pass == n_total:
    print("  All analyses reproduced. Ready for GitHub upload.")
else:
    print("  Check failed scripts above.")
print('='*70)
