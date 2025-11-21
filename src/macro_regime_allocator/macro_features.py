import numpy as np
import pandas as pd
from .config import MOM_BASE, FEATURES_LEVEL, REGIME_COL, TOLS, FEATURES_MOM

def load_macro_with_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    # Momentum 3M
    for col in MOM_BASE:
        if col in df.columns:
            df[f"{col}_mom3"] = df[col].diff(3)

    def label_regime(row):
        gdp_m = row.get("GDP_YoY_mom3", np.nan)
        unemp_m = row.get("Unemployment_mom3", np.nan)
        eps_gdp, eps_unemp = 0.3, 0.1
        if pd.isna(gdp_m) or pd.isna(unemp_m):
            return "Stable"
        if (gdp_m >= eps_gdp) and (unemp_m <= -eps_unemp):
            return "Croissance"
        elif (gdp_m <= -eps_gdp) and (unemp_m >= eps_unemp):
            return "Récession"
        return "Stable"

    df[REGIME_COL] = df.apply(label_regime, axis=1)
    return df

def vectorisation(macro_hist: pd.DataFrame):
    df = macro_hist[FEATURES_LEVEL].dropna(how="any").copy()
    mu = df.mean()
    sigma = df.std(ddof=0).replace(0, np.nan)
    Z = (df - mu) / sigma
    return Z, mu, sigma

def compute_momentum_score(macro_hist, similar_dates, mom_features):
    available_mom = [c for c in mom_features if c in macro_hist.columns]
    if not available_mom:
        return 0.0
    mom_hist = macro_hist.loc[similar_dates, available_mom].dropna(how="any")
    if mom_hist.empty:
        return 0.0
    global_mom = macro_hist[available_mom].dropna(how="any")
    mu_m = global_mom.mean()
    sigma_m = global_mom.std(ddof=0).replace(0, np.nan)
    Z_mom = (mom_hist - mu_m) / sigma_m
    score = Z_mom.abs().mean().mean()
    return float(score)

def adjust_mu_sigma(m, Sigma, regime: str, momentum_score: float):
    base_beta = 0.3
    if regime == "Récession":
        gamma = 0.7
    elif regime == "Croissance":
        gamma = 0.3
    else:
        gamma = 0.5

    factor_sigma = 1.0 + base_beta * momentum_score
    factor_mu = 1.0 + gamma * momentum_score

    Sigma_adj = Sigma * factor_sigma
    m_adj = m / factor_mu
    return m_adj, Sigma_adj
