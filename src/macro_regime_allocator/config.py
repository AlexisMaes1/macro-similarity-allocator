MACRO_CSV = "data/macro_indicators.csv"
PRICES_CSV = "data/asset_prices_daily_12ETF.csv"

BACKTEST_START = "2015-01-01"
CUTOFF_MARKET = "2008-01-01"

ASSETS = [
    "Equity_SP500",
    "GovBonds_10y+",
    "Gold",
    "Commodities",
    "REITs_US",
    "Cash_TBills",
    "Equity_DevExUS_VEA",
    "Equity_EM_VWO",
    "Equity_Japan_EWJ",
    "Equity_Eurozone_EZU",
    "Equity_EM_EEM",
    "Equity_Europe_FEZ",
]

TARGET_VOL_ANN = 0.10
LONG_ONLY = True
MAX_WEIGHT = 0.5
LAMBDA_SHRINK = 0.25
K_TOP = 12

FEATURES_LEVEL = [
    "GDP_YoY",
    "CPI_YoY",
    "FedFunds",
    "UST10Y",
    "Spread_10Y_2Y",
    "Unemployment",
    "VIX",
    "INDPRO_YoY",
    "RetailSales_YoY",
    "HousingStarts_YoY",
    "BAA10Y_Spread",
    "UMich_Sentiment",
]
MOM_BASE = [
    "GDP_YoY",
    "CPI_YoY",
    "FedFunds",
    "UST10Y",
    "Spread_10Y_2Y",
    "Unemployment",
    "VIX",
    ]
FEATURES_MOM = [
    f"{c}_mom3" for c in MOM_BASE
]
TOLS = {
    "GDP_YoY":   1.2,
    "CPI_YoY":   1.2,
    "FedFunds":  1.0,
    "UST10Y":    1.0,
    "Spread_10Y_2Y":     2.0,
    "Unemployment":      2.5,
    "VIX":               4.0,
    "INDPRO_YoY":        1.5,
    "RetailSales_YoY":   1.5,
    "HousingStarts_YoY": 2.5,
    "BAA10Y_Spread":     1.5,
    "UMich_Sentiment":   1.5,
}

REGIME_COL = "Regime"
REGIME_CHOICES = ["Croissance", "Stable", "Récession"]
