import numpy as np
import pandas as pd

def solve(A, b):
    return np.linalg.solve(A, b)

def annual_vol_to_daily(sigma_annual, periods=252):
    return sigma_annual / np.sqrt(periods)

def ann_from_daily(mu_d, sig_d, periods=252):
    mu_a = (1 + mu_d)**periods - 1
    sig_a = sig_d * np.sqrt(periods)
    return mu_a, sig_a

def shrink_covariance(Sigma: pd.DataFrame, lam: float = 0.25) -> pd.DataFrame:
    diag_vals = np.diag(np.diag(Sigma.values))
    T = pd.DataFrame(diag_vals, index=Sigma.index, columns=Sigma.columns)
    Sigma_shrink = lam * T + (1.0 - lam) * Sigma
    return Sigma_shrink

def project_long_only_capped(w: pd.Series, cap: float = 0.5) -> pd.Series:
    w2 = w.clip(lower=0.0)
    w2 = w2.clip(upper=cap)
    s = w2.sum()
    return w2 / s if s > 0 else w2

def risk_parity_weights(Sigma: pd.DataFrame,
                        max_iter: int = 1000,
                        tol: float = 1e-8) -> pd.Series:
    Sigma_mat = Sigma.values.astype(float)
    n = Sigma_mat.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_p = float(np.sqrt(w @ Sigma_mat @ w))
        if sigma_p == 0:
            break
        mrc = Sigma_mat @ w / sigma_p
        rc = w * mrc
        target = sigma_p / n
        diff = rc - target
        if np.max(np.abs(diff)) < tol:
            break
        w = w * (target / np.maximum(rc, 1e-12))
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = np.ones(n) / n if s <= 0 else w / s
    return pd.Series(w, index=Sigma.index)
