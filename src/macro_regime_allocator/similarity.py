import numpy as np
import pandas as pd
from .config import TOLS, REGIME_COL

def similar_period(mu, sigma, Z, macro_hist, x_level, k, regime_choice: str):
    mask_regime = macro_hist[REGIME_COL] == regime_choice
    if not mask_regime.any():
        return None

    Z_reg = Z.loc[mask_regime]
    if Z_reg.empty:
        return None

    zx = (x_level - mu) / sigma
    common_cols = Z_reg.columns.intersection(zx.index)
    if len(common_cols) == 0:
        return None

    tol_scales = [1.0, 1.5, 2.0, 3.0, 5.0]
    similar = None
    for scale in tol_scales:
        mask = np.ones(len(Z_reg), dtype=bool)
        for f, base_tol in TOLS.items():
            if f not in Z_reg.columns or f not in zx.index:
                continue
            tol = base_tol * scale
            mask &= (Z_reg[f] - zx[f]).abs() <= tol
        similar = Z_reg[mask]
        if not similar.empty:
            break

    if similar is None or similar.empty:
        return None

    similar_num = similar[common_cols].astype(float)
    zx_num = zx[common_cols].astype(float)
    diff = similar_num.sub(zx_num, axis=1)
    dist_array = np.sqrt(np.sum(diff.values ** 2, axis=1))
    dist = pd.Series(dist_array, index=similar.index)

    k_eff = min(k, len(dist))
    nearest = dist.nsmallest(k_eff)
    return pd.DatetimeIndex(nearest.index)
