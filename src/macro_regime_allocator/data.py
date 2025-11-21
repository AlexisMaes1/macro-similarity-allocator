import pandas as pd
from .config import ASSETS

def load_macro(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df[ASSETS].dropna(how="all")
