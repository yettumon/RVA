"""
rva_utils.py — Shared constants and helper functions
=====================================================
GCOD-III: Recession Velocity Anisotropy (Kim 2026c)
GitHub: github.com/yettumon/RVA

Used by: all rva0X_*.py scripts
"""

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy import stats
import os

# ── Physical constants ─────────────────────────────────────────────────
C       = 299792.458   # speed of light (km/s)

# ── LA-axis definition (Kim 2026a) ─────────────────────────────────────
AXIS_RA  = 330.0       # RA  (deg)
AXIS_DEC = 0.0         # Dec (deg)

# ── CMB dipole (Planck 2018) ────────────────────────────────────────────
CMB_RA   = 168.0       # RA  (deg)
CMB_DEC  = -7.0        # Dec (deg)
CMB_V    = 369.0       # speed (km/s)

# ── Local Group motion ──────────────────────────────────────────────────
LG_RA    = 276.0       # RA  (deg)
LG_DEC   = 30.0        # Dec (deg)
LG_V     = 627.0       # speed (km/s)

# ── Sample cuts ────────────────────────────────────────────────────────
Z_LO     = 0.15
Z_HI     = 0.50

# ── Default random seed ────────────────────────────────────────────────
SEED     = 42

# ── Data file search paths ─────────────────────────────────────────────
DATA_PATHS = [
    'Pantheon_SH0ES.dat',
    '../Pantheon_SH0ES.dat',
    '/mnt/project/Pantheon_SH0ES.dat',
    os.path.expanduser('~/Pantheon_SH0ES.dat'),
]


def load_pantheon(paths=None):
    """Load Pantheon+SH0ES catalogue. Returns full DataFrame."""
    if paths is None:
        paths = DATA_PATHS
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, sep=r'\s+', comment='#')
            print(f"[load] {p}  (N={len(df)})")
            return df
    raise FileNotFoundError(
        "Pantheon_SH0ES.dat not found. "
        "Download from https://pantheonplussh0es.github.io/ "
        "and place in the RVA directory."
    )


def make_sample(df):
    """
    Apply analysis filters:
      IS_CALIBRATOR=0, Z_LO < zHD < Z_HI
    Returns filtered copy with v_pure, cos_LA, sin_LA, theta columns added.
    """
    d = df[(df['IS_CALIBRATOR'] == 0) &
           (df['zHD'] > Z_LO) &
           (df['zHD'] < Z_HI)].copy()

    # CMB + LG peculiar velocity correction
    d['v_pure'] = (C * d['zHD']
                   - CMB_V * dot_sky(d['RA'], d['DEC'], CMB_RA, CMB_DEC)
                   - LG_V  * dot_sky(d['RA'], d['DEC'], LG_RA,  LG_DEC))

    # Angular distance from LA-axis
    d['cos_LA'] = dot_sky(d['RA'], d['DEC'], AXIS_RA, AXIS_DEC)
    d['sin_LA'] = np.sqrt(np.clip(1 - d['cos_LA']**2, 0, 1))
    d['theta']  = np.degrees(np.arccos(np.clip(d['cos_LA'], -1, 1)))

    return d


def make_groups(d):
    """
    Split sample into axis / perpendicular / anti-axis groups.
    Returns (axis_df, perp_df, anti_df).
    """
    axis = d[d['theta'] < THETA_AXIS_MAX].copy()
    perp = d[(d['theta'] >= THETA_AXIS_MAX) & (d['theta'] <= THETA_ANTI_MIN)].copy()
    anti = d[d['theta'] > THETA_ANTI_MIN].copy()
    return axis, perp, anti


def dot_sky(ra, dec, ra0, dec0):
    """
    Cosine of angular distance between sky positions (ra,dec) and (ra0,dec0).
    All inputs in degrees. Vectorised over arrays.
    """
    return (np.sin(np.radians(dec))  * np.sin(np.radians(dec0)) +
            np.cos(np.radians(dec))  * np.cos(np.radians(dec0)) *
            np.cos(np.radians(ra)    - np.radians(ra0)))


def ols(X, y):
    """
    OLS regression via scipy.linalg.lstsq.
    Returns (coefficients, standard_errors, t_values, p_values, R²).
    """
    b, _, _, _ = lstsq(X, y)
    r   = y - X @ b
    k   = X.shape[1]
    n   = len(y)
    s2  = np.sum(r**2) / (n - k)
    se  = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t   = b / se
    p   = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    r2  = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    return b, se, t, p, r2


def greedy_match(axis_df, perp_df, dz_max=0.01):
    """
    1:1 greedy nearest-neighbour matching on zHD.
    Each axis object is matched to the closest unmatched perp object
    within dz_max. Returns DataFrame with columns [z_ax, z_pp, dv, dz].
    """
    ax = axis_df[['zHD', 'v_pure']].reset_index(drop=True)
    pp = perp_df[['zHD', 'v_pure']].reset_index(drop=True)
    used = set()
    rows = []
    for _, row_ax in ax.iterrows():
        best_i, best_dz = None, 1e9
        for i, row_pp in pp.iterrows():
            if i in used:
                continue
            dz = abs(row_ax['zHD'] - row_pp['zHD'])
            if dz < best_dz:
                best_dz, best_i = dz, i
        if best_i is not None and best_dz < dz_max:
            used.add(best_i)
            rows.append({
                'z_ax': row_ax['zHD'],
                'z_pp': pp.loc[best_i, 'zHD'],
                'dv':   pp.loc[best_i, 'v_pure'] - row_ax['v_pure'],
                'dz':   best_dz,
            })
    return pd.DataFrame(rows)


def section(title):
    """Print a section header."""
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def check(label, val, ref, tol, fmt='.3f'):
    """Print a paper-value verification line. Returns True if passed."""
    ok  = abs(val - ref) <= tol
    sym = '✅' if ok else '❌'
    print(f"  {sym}  {label}: {val:{fmt}}  (paper: {ref})")
    return ok

# ── Group angle boundaries ─────────────────────────────────────────────
THETA_AXIS_MAX = 45.0    # principal axis: θ < 45°
THETA_ANTI_MIN = 135.0   # anti-axis:      θ > 135°
# perpendicular: THETA_AXIS_MAX ≤ θ ≤ THETA_ANTI_MIN

# ── Distance matching threshold ────────────────────────────────────────
DZ_MAX = 0.01            # |Δz| < 0.01 (§3.4)
