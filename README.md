# RVA: Recession Velocity Anisotropy

**Quantitative Verification of Direction-Dependent Pure Radial Signal Along the LA-axis**

Kim, GunSik (Yettumon) 2026 | GCOD-III  
DOI: *(Zenodo, forthcoming)*

---

## Overview

Full analysis pipeline for the RVA paper (Kim 2026c), which quantifies recession velocity anisotropy along the LA-axis (RA=330°, Dec=0°) using the Pantheon+SH0ES supernova dataset.

**Core results:**
| Method | Result | p-value |
|--------|--------|---------|
| Distance matching (primary) | Δv = +380.6 ± 39.5 km/s | < 10⁻¹⁶ |
| Multivariate regression (auxiliary) | c = +335.6 ± 37.7 km/s | < 0.001 |
| Mock Catalog (isotropic null) | 5.2σ, 0/2,000 exceedances | < 10⁻⁵ |

All paper values are fully reproduced (51/51 checks pass).

---

## Repository structure

```
RVA/
├── README.md
├── rva_utils.py                  # Shared constants, helpers, data loader
│                                 #   Constants: THETA_AXIS_MAX=45°, THETA_ANTI_MIN=135°,
│                                 #             DZ_MAX=0.01, CMB/LG params, Z_LO/Z_HI
├── rva01_sample_correction.py    # §2.1–2.3  Data load, CMB+LG correction, groups
├── rva02_raw_anisotropy.py       # §3.1      Raw Δv, Welch t-test, Mann-Whitney U
├── rva03_regression.py           # §3.3      Multivariate regression (Method 2, auxiliary)
├── rva04_distance_matching.py    # §3.4–3.5  Distance matching (Method 3, primary) + decomposition
├── rva05_zbin_stratified.py      # §3.6–3.7  Stratified z-bin analysis + F-test
├── rva06_mock_catalog.py         # §3.8      Mock Catalog isotropic null hypothesis test
├── rva07_confounders.py          # §4.5–4.6  Sequential confounder control + fit quality
├── rva08_loso.py                 # App B.1   Leave-One-Survey-Out (LOSO)
├── rva09_cmb_propagation.py      # App B.2   CMB correction uncertainty propagation
└── rva_run_all.py                # Pipeline runner — executes rva01–rva09 in order
```

`rva_run_all.py` runs all nine scripts sequentially and prints a pass/fail summary.  
Each script can also be run independently.

See also: `validation7_CMB_collinearity.py` (§4.3 CMB multicollinearity, cited in paper).

---

## Data

**Pantheon+SH0ES** (Brout et al. 2022, ApJ 938, 110)  
Download: https://pantheonplussh0es.github.io/  
Place `Pantheon_SH0ES.dat` in the same directory as the scripts.

**CF3** (Tully et al. 2016, AJ 152, 50) — used in validation scripts (Kim 2026a):  
`CF3_all_distance.csv` — required for `validation4_CF3_regression.py` and `validation6_CF3_grid_scan.py`.

---

## Dependencies

```bash
pip install numpy pandas scipy
```

Optional (for `validation7_CMB_collinearity.py`):
```bash
pip install astropy
```

Python 3.8+ recommended.

---

## Usage

**Run entire pipeline:**
```bash
python rva_run_all.py
```

**Run individual section:**
```bash
python rva01_sample_correction.py   # §2.1–2.3 data & groups
python rva03_regression.py          # §3.3 multivariate regression
python rva04_distance_matching.py   # §3.4 distance matching (primary test)
python rva05_zbin_stratified.py     # §3.6–3.7 z-bin + F-test
python rva06_mock_catalog.py        # §3.8 mock catalog
python rva08_loso.py                # App B.1 LOSO
```

Each script prints computed values alongside paper reference values with ✅/❌ markers.

---

## Implementation notes

### F-test in §3.7 (rva05)
The exp(−z/z₀) attenuation model is fitted by **nonlinear weighted least squares** via `scipy.optimize.curve_fit` with `sigma=SE_per_bin`. Log-space linear WLS gives a different result (F≈15.4); `curve_fit` reproduces the paper value (F=18.6, p=0.013). See docstring in `rva05_zbin_stratified.py`.

### §3.6 z-bin stratification (rva05)
Paper Table 6 uses the **global 205 matched pairs** (from §3.4), then classifies each pair by the axis-object's zHD — not independent per-bin matching. N_ax sums: 107+74+24 = 205.

### No hardcoded intermediate values
All analysis constants are defined in `rva_utils.py` — angle thresholds (`THETA_AXIS_MAX=45°`, `THETA_ANTI_MIN=135°`), matching threshold (`DZ_MAX=0.01`), CMB/LG parameters, and redshift cuts (`Z_LO`, `Z_HI`). Changing any of these propagates automatically to all scripts. Paper reference values appear only inside `check()` calls for verification and do not affect computation. Each script loads data independently and can be run standalone without `rva_run_all.py`.

---

## Note on validation scripts (validation1–7)

Scripts `validation1_DR_angle.py` through `validation7_CMB_collinearity.py` reproduce results from the LA-axis paper (Kim 2026a) and TRT paper (Kim 2026b). `validation7_CMB_collinearity.py` is directly cited in §4.3 of the RVA paper.

| Script | Paper | Section |
|--------|-------|---------|
| validation1_DR_angle.py | Kim 2026a (LA-axis) | §4.1 |
| validation2_CMB_vector.py | Kim 2026a (LA-axis) | §4.2 |
| validation3_calibrator_bias.py | Kim 2026b (TRT) | §4.3 |
| validation4_CF3_regression.py | Kim 2026a (LA-axis) | §4.4 |
| validation5_fisher_combined.py | Kim 2026a (LA-axis) | §6 |
| validation6_CF3_grid_scan.py | Kim 2026a (LA-axis) | §3.2 |
| validation7_CMB_collinearity.py | **Kim 2026c (RVA)** | **§4.3** |

---

## Citation

```
Kim, G.S. (Yettumon) (2026c). Recession Velocity Anisotropy:
Quantitative Verification of Direction-Dependent Pure Radial Signal
Along the LA-axis. Zenodo. doi:forthcoming
```

Related papers:
- Kim 2026a (LA-axis): https://doi.org/10.5281/zenodo.20695683
- Kim 2026b (TRT): https://doi.org/10.5281/zenodo.20572072

---

*Independent researcher | nearcop3@gmail.com | github.com/yettumon*
