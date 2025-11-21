import pandas as pd
import numpy as np

from .config import ASSETS       
from .optimization import risk_parity_weights


def plot_performance_comparison(equity_df: pd.DataFrame,
                                prices_df: pd.DataFrame,
                                output_path: str = "backtest_comparison.png"):
    """
    Construit un graphique comparant :
    - ta stratégie (equity_df)
    - S&P500 seul
    - 60/40 (SP500/TLT)
    - 1/N multi-asset (tous les ETF sauf Cash_TBills)
    - Risk Parity (Equal Risk Contribution) sur les mêmes ETF
    """
    # Courbe de ta stratégie (déjà une equity)
    strat_eq = equity_df["Equity"].copy()
    strat_eq = strat_eq / strat_eq.iloc[0]

    # On restreint les prix à la même période
    start, end = strat_eq.index.min(), strat_eq.index.max()
    px = prices_df.loc[start:end].copy()

    # Rendements journaliers
    rets = px.pct_change().dropna(how="all")

    # --- S&P 500 ---
    sp_eq = None
    if "Equity_SP500" in rets.columns:
        sp_rets = rets["Equity_SP500"]
        sp_eq = (1.0 + sp_rets).cumprod()

    # --- 60/40 SP500 / GovBonds_10y+ ---
    eq_6040 = None
    if ("Equity_SP500" in rets.columns) and ("GovBonds_10y+" in rets.columns):
        ret_6040 = 0.6 * rets["Equity_SP500"] + 0.4 * rets["GovBonds_10y+"]
        eq_6040 = (1.0 + ret_6040).cumprod()

    # --- 1/N multi-asset (hors cash) ---
    eq_1N = None
    risky_assets = [a for a in ASSETS if a in rets.columns and a != "Cash_TBills"]
    if len(risky_assets) > 0:
        w_equal = np.ones(len(risky_assets)) / len(risky_assets)
        ret_1N = (rets[risky_assets] * w_equal).sum(axis=1)
        eq_1N = (1.0 + ret_1N).cumprod()

    # --- Risk Parity (ERC) ---
    eq_rp = None
    if len(risky_assets) > 0:
        Sigma = rets[risky_assets].cov()
        try:
            w_rp = risk_parity_weights(Sigma)
            ret_rp = (rets[risky_assets] * w_rp).sum(axis=1)
            eq_rp = (1.0 + ret_rp).cumprod()
        except Exception as e:
            print(f"[WARN] Risk parity a échoué : {e}")

    # --- Assemblage ---
    curves = {"Ma stratégie": strat_eq}
    if sp_eq is not None:
        curves["S&P 500 seul"] = sp_eq
    if eq_6040 is not None:
        curves["60/40 SP500/ETF 20y+ treasury bond"] = eq_6040
    if eq_1N is not None:
        curves["1/N multi-ETF (pas full actions)"] = eq_1N
    if eq_rp is not None:
        curves["Risk Parity (ERC)"] = eq_rp

    curves_df = pd.DataFrame(curves)

    # Normalisation INDIVIDUELLE base 1 pour chaque courbe
    for col in curves_df.columns:
        s = curves_df[col].dropna()
        if len(s) == 0:
            continue
        curves_df[col] = curves_df[col] / s.iloc[0]

    curves_df = curves_df.dropna(how="all")

    # --- Plot ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for col in curves_df.columns:
        plt.plot(curves_df.index, curves_df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Valeur du portefeuille (1 = capital initial)")

    plt.title("Comparaison des performances – Stratégie vs Benchmarks")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graphique de comparaison sauvegardé dans : {output_path}")