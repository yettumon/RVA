"""
Validation 7: CMB Collinearity Analysis — sin vs cos basis
===========================================================
GCOD-III: Recession Velocity Anisotropy (RVA)
Section 4.3: CMB Multicollinearity

Key Results:
  cos basis: cos(θ_GCOD) vs cos(θ_CMB) r=−0.97, VIF=17 (geometric fact)
  sin basis: sin(θ_GCOD) vs cos(θ_CMB) r=+0.069, VIF=1.01 (no collinearity)
  CMB simultaneous control: c=+359.6±31.7 km/s (strengthened, not weakened)

Physical interpretation:
  LA-axis Aquarius end (330°) ↔ Dipole Repeller (337.5°): 7.5°
  LA-axis Leo end     (150°) ↔ CMB dipole    (168°,−7°): 19.3°
  CMB motion vector (369 km/s) projected onto Leo direction: 348.3 km/s (94.4%)

Author: GunSik Kim (Yettumon) | 2026
Data:   Pantheon+SH0ES.dat (place in same directory)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq
import os

# ── Data load ──────────────────────────────────────────────
_PATHS = [
    'Pantheon_SH0ES.dat',
    'Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon_SH0ES.dat',
]
df = None
for path in _PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path, sep=r'\s+', comment='#')
        print(f"Loaded: {path} (N={len(df)})")
        break
if df is None:
    raise FileNotFoundError("Pantheon+SH0ES.dat not found.")

C = 299792.458
AXIS_RA, AXIS_DEC = 330.0, 0.0
CMB_RA,  CMB_DEC  = 168.0, -7.0
CMB_V, LG_V = 369.0, 627.0

def dot_sky(ra, dec, ra0, dec0):
    return (np.sin(np.radians(dec))*np.sin(np.radians(dec0)) +
            np.cos(np.radians(dec))*np.cos(np.radians(dec0))*
            np.cos(np.radians(ra)-np.radians(ra0)))

# ── Sample filter ──────────────────────────────────────────
d = df[(df['IS_CALIBRATOR']==0)&(df['zHD']>0.15)&(df['zHD']<0.50)].copy()
d['v_pure'] = (C*d['zHD']
               - CMB_V*dot_sky(d['RA'],d['DEC'],168.,-7.)
               - LG_V *dot_sky(d['RA'],d['DEC'],276.,30.))
d['cos_GCOD'] = dot_sky(d['RA'],d['DEC'],AXIS_RA,AXIS_DEC)
d['cos_CMB']  = dot_sky(d['RA'],d['DEC'],CMB_RA, CMB_DEC)
d['sin_GCOD'] = np.sqrt(np.clip(1-d['cos_GCOD']**2,0,1))
n = len(d)
print(f"Sample: N={n}, z=0.15~0.50, IS_CALIBRATOR=0\n")

def ols(X, y):
    b,_,_,_ = lstsq(X, y)
    r = y - X@b
    k = X.shape[1]
    s2 = np.sum(r**2)/(len(y)-k)
    se = np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    t = b/se
    p = 2*(1-stats.t.cdf(np.abs(t), df=len(y)-k))
    return b, se, t, p

# ── Analysis 1: cos basis collinearity ─────────────────────
print("="*60)
print("Analysis 1: cos basis — geometric collinearity")
print("="*60)
r_cos = np.corrcoef(d['cos_GCOD'], d['cos_CMB'])[0,1]
X_aux_cos = np.column_stack([np.ones(n), d['zHD'], d['cos_CMB']])
b_aux,_,_,_ = lstsq(X_aux_cos, d['cos_GCOD'])
r2_cos = 1-np.sum((d['cos_GCOD']-X_aux_cos@b_aux)**2)/np.sum((d['cos_GCOD']-d['cos_GCOD'].mean())**2)
vif_cos = 1/(1-r2_cos)
print(f"cos(θ_GCOD) vs cos(θ_CMB): r={r_cos:.4f}")
print(f"VIF(cos_GCOD) = {vif_cos:.1f}")
print("→ Geometric consequence of 160.7° angular separation")

# ── Analysis 2: sin basis — actual main test ────────────────
print("\n" + "="*60)
print("Analysis 2: sin basis — actual main test (no collinearity)")
print("="*60)
r_sin = np.corrcoef(d['sin_GCOD'], d['cos_CMB'])[0,1]
X_aux_sin = np.column_stack([np.ones(n), d['zHD'], d['cos_CMB']])
b_aux2,_,_,_ = lstsq(X_aux_sin, d['sin_GCOD'])
r2_sin = 1-np.sum((d['sin_GCOD']-X_aux_sin@b_aux2)**2)/np.sum((d['sin_GCOD']-d['sin_GCOD'].mean())**2)
vif_sin = 1/(1-r2_sin)
print(f"sin(θ_GCOD) vs cos(θ_CMB): r={r_sin:.4f}")
print(f"VIF(sin_GCOD) = {vif_sin:.3f}")
print("→ No multicollinearity in actual main test")

# ── Analysis 3: CMB simultaneous control ───────────────────
print("\n" + "="*60)
print("Analysis 3: CMB simultaneous control — signal strengthened")
print("="*60)
y = d['v_pure'].values
z = d['zHD'].values
sin_g = d['sin_GCOD'].values
cos_c = d['cos_CMB'].values

X_base = np.column_stack([np.ones(n), z, sin_g])
b1, se1, t1, p1 = ols(X_base, y)
print(f"Base (z + sin_GCOD):             c={b1[2]:+.1f}±{se1[2]:.1f}, t={t1[2]:.2f}, p={p1[2]:.2e}")

X_both = np.column_stack([np.ones(n), z, sin_g, cos_c])
b2, se2, t2, p2 = ols(X_both, y)
print(f"+ CMB (z + sin_GCOD + cos_CMB):  c={b2[2]:+.1f}±{se2[2]:.1f}, t={t2[2]:.2f}, p={p2[2]:.2e}")
print(f"CMB coefficient:                 d={b2[3]:+.1f}±{se2[3]:.1f}")
print(f"→ Signal strengthened (+{b2[2]-b1[2]:.1f} km/s), not weakened")

# ── Analysis 4: LA-axis bipolar alignment ──────────────────
print("\n" + "="*60)
print("Analysis 4: LA-axis bipolar alignment")
print("="*60)
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    def sep(r1,d1,r2,d2):
        return SkyCoord(ra=r1*u.deg,dec=d1*u.deg).separation(
               SkyCoord(ra=r2*u.deg,dec=d2*u.deg)).deg
    s_dr  = sep(330,0,337.5,0)
    s_cmb = sep(150,0,168,-7)
    print(f"LA Aquarius(330°) ↔ Dipole Repeller(337.5°): {s_dr:.2f}°")
    print(f"LA Leo(150°)      ↔ CMB dipole(168°,−7°):    {s_cmb:.2f}°")
except ImportError:
    s_cmb = 19.27
    print("astropy not available; using pre-computed values")
    print(f"LA Aquarius(330°) ↔ Dipole Repeller(337.5°): 7.50°")
    print(f"LA Leo(150°)      ↔ CMB dipole(168°,−7°):    19.27°")

la_leo = np.array([np.cos(0)*np.cos(np.radians(150)),
                   np.cos(0)*np.sin(np.radians(150)), 0.])
cmb_vec = np.array([np.cos(np.radians(-7))*np.cos(np.radians(168)),
                    np.cos(np.radians(-7))*np.sin(np.radians(168)),
                    np.sin(np.radians(-7))])
proj = np.dot(cmb_vec, la_leo)
print(f"CMB vector (369 km/s) → Leo projection: {proj*369:.1f} km/s ({abs(proj)*100:.1f}%)")

# ── Verification ───────────────────────────────────────────
print("\n" + "="*60)
print("Verification against paper values")
print("="*60)
checks = [
    ("sin_GCOD vs cos_CMB r",   r_sin,         0.069,  0.01),
    ("VIF(sin_GCOD)",            vif_sin,       1.01,   0.05),
    ("c base",                   b1[2],         335.6,  2.0),
    ("c + CMB",                  b2[2],         359.6,  2.0),
    ("CMB Leo projection %",     abs(proj)*100, 94.4,   0.5),
]
for label, val, ref, tol in checks:
    ok = abs(val-ref) <= tol
    print(f"  {'✅' if ok else '❌'} {label}: {val:.3f} (paper: {ref})")
