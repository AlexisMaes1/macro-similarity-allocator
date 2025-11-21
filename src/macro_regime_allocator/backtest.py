import pandas as pd
import numpy as np

from .config import (
    MACRO_CSV, PRICES_CSV, BACKTEST_START, CUTOFF_MARKET,
    ASSETS, TARGET_VOL_ANN, LONG_ONLY, MAX_WEIGHT,
    FEATURES_LEVEL, FEATURES_MOM, REGIME_COL, K_TOP,
    LAMBDA_SHRINK,
)
from .data import load_prices
from .macro_features import load_macro_with_features, vectorisation, compute_momentum_score, adjust_mu_sigma
from .similarity import similar_period
from .optimization import (
    shrink_covariance, annual_vol_to_daily, ann_from_daily,
    project_long_only_capped, solve,
)
from .plotting import plot_performance_comparison

def compute_allocation_at_date(decision_date: pd.Timestamp,
                               macro_df: pd.DataFrame,
                               prices_df: pd.DataFrame):
    macro_hist = macro_df.loc[macro_df.index < decision_date].copy()
    if macro_hist.empty:
        return None, None, None

    if not set(FEATURES_LEVEL).issubset(macro_hist.columns):
        return None, None, None

    if decision_date not in macro_df.index:
        return None, None, None

    row_t = macro_df.loc[decision_date]
    regime_t = row_t[REGIME_COL]
    x_level = row_t[FEATURES_LEVEL]

    Z_hist, mu_level, sigma_level = vectorisation(macro_hist)
    similar_dates = similar_period(
        mu_level, sigma_level, Z_hist, macro_hist, x_level, K_TOP, regime_t
    )
    if similar_dates is None or len(similar_dates) == 0:
        return None, None, None

    cutoff = pd.Timestamp(CUTOFF_MARKET)
    similar_dates = similar_dates[similar_dates >= cutoff]
    if len(similar_dates) == 0:
        return None, None, None

    momentum_score = compute_momentum_score(macro_hist, similar_dates, FEATURES_MOM)

    months = similar_dates.to_period("M").unique().sort_values()
    extended_periods = sorted({p + i for p in months for i in range(0, 4)})
    extended_periods = [p for p in extended_periods if p.end_time < decision_date]
    if not extended_periods:
        return None, None, None

    global_start = extended_periods[0].start_time.normalize()
    global_end = extended_periods[-1].end_time.normalize()
    prices_train = prices_df.loc[global_start:global_end].dropna(how="all")
    if prices_train.empty:
        return None, None, None

    chunks = []
    for p in extended_periods:
        m = (prices_train.index >= p.start_time) & (prices_train.index <= p.end_time)
        chunk = prices_train.loc[m]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return None, None, None

    daily_on_similar = pd.concat(chunks).sort_index()
    returns = daily_on_similar.pct_change().dropna(how="all")
    if returns.empty:
        return None, None, None

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    Sigma = cov_matrix.fillna(0)
    Sigma_shrink = shrink_covariance(Sigma, lam=LAMBDA_SHRINK)
    mu_s = mean_returns.fillna(0)
    mu_adj, Sigma_adj = adjust_mu_sigma(mu_s, Sigma_shrink, regime_t, momentum_score)

    assets = mu_adj.index
    S = Sigma_adj.values
    m_vec = mu_adj.values
    one = np.ones_like(m_vec)

    try:
        Sinv_1 = solve(S, one)
        Sinv_m = solve(S, m_vec)
    except Exception:
        return None, None, None

    a = float(one @ Sinv_1)
    b = float(one @ Sinv_m)
    c = float(m_vec @ Sinv_m)
    denom = (a * c - b**2)
    if denom <= 0 or a <= 0:
        return None, None, None

    w_gmv = Sinv_1 / a
    w_mrr = Sinv_m / b
    sigma_gmv = (1.0 / a)**0.5

    sigma_target_daily = annual_vol_to_daily(TARGET_VOL_ANN)
    if sigma_target_daily**2 < sigma_gmv**2:
        return None, None, None

    tau = ((a * (sigma_target_daily**2) - 1.0) / denom)**0.5
    w_vec = (b * tau) * w_mrr + (1.0 - b * tau) * w_gmv
    w = pd.Series(w_vec, index=assets)

    if LONG_ONLY:
        w = project_long_only_capped(w, cap=MAX_WEIGHT)

    return w, momentum_score, sigma_target_daily


def run_backtest(macro_path: str | None = None,
                 prices_path: str | None = None):
    macro_path = macro_path or MACRO_CSV
    prices_path = prices_path or PRICES_CSV

    macro_df = load_macro_with_features(macro_path)
    prices_df = load_prices(prices_path)

    macro_df = macro_df.sort_index()
    decision_dates = macro_df.index[macro_df.index >= BACKTEST_START]

    prices_df = prices_df.loc[pd.Timestamp(CUTOFF_MARKET):].copy()

    equity_curve = []
    all_portfolios = []
    all_daily_returns = []

    current_portfolio = None
    current_value = 1.0

    for i in range(len(decision_dates) - 1):
        t = decision_dates[i]
        t_next = decision_dates[i+1]

        w, mom_score, sigma_target_daily = compute_allocation_at_date(t, macro_df, prices_df)
        if w is None:
            print(f"[{t.date()}] Impossible de calculer les poids, on conserve le portefeuille précédent.")
            if current_portfolio is None:
                continue
        else:
            current_portfolio = w
            port_row = {
                "Date": t,
                "Regime": macro_df.loc[t, REGIME_COL],
                "MomentumScore": mom_score,
                "SigmaTargetDaily": sigma_target_daily,
            }
            for asset in ASSETS:
                port_row[f"w_{asset}"] = current_portfolio.get(asset, 0.0)
            all_portfolios.append(port_row)

        if current_portfolio is not None:
            window_prices = prices_df.loc[(prices_df.index > t) & (prices_df.index <= t_next)]
            if window_prices.empty:
                continue

            window_rets = window_prices.pct_change().dropna(how="all")
            valid_assets = [a for a in ASSETS if a in window_rets.columns]
            if not valid_assets:
                continue

            w_vec = current_portfolio.reindex(valid_assets).fillna(0.0)
            port_rets = (window_rets[valid_assets] * w_vec).sum(axis=1)

            for dt, r in port_rets.items():
                current_value *= (1.0 + r)
                equity_curve.append((dt, current_value))
                all_daily_returns.append((dt, r))

    if not equity_curve:
        print("Aucun résultat de backtest.")
        return

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"]).set_index("Date")
    daily_ret_df = pd.DataFrame(all_daily_returns, columns=["Date", "Ret"]).set_index("Date")

    mu_d = daily_ret_df["Ret"].mean()
    sig_d = daily_ret_df["Ret"].std(ddof=0)
    mu_a, sig_a = ann_from_daily(mu_d, sig_d)
    sharpe = mu_a / sig_a if sig_a > 0 else 0.0

    print("=== Résultats backtest ===")
    print(f"Période : {equity_df.index.min().date()} → {equity_df.index.max().date()}")
    print(f"Rendement annualisé : {mu_a*100:.2f}%")
    print(f"Volatilité annualisée : {sig_a*100:.2f}%")
    print(f"Sharpe (rf≈0) : {sharpe:.2f}")

    equity_df.to_csv("backtest_equity_curve.csv", index=True)
    daily_ret_df.to_csv("backtest_daily_returns.csv", index=True)
    if all_portfolios:
        ports_df = pd.DataFrame(all_portfolios).set_index("Date")
        ports_df.to_csv("backtest_portfolios.csv", index=True)

    plot_performance_comparison(equity_df, prices_df, output_path="backtest_comparison.png")

    return equity_df, daily_ret_df
