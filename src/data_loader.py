# data_loader.py
# Dynamically loads a CSV file containing either prices or returns for multiple tickers.
# Routes to calculations.py for log or linear return computation when only prices are given.

import pandas as pd
from calculations import compute_log_returns, compute_linear_returns


#Column-Name Aliases

_DATE_ALIASES: list[str] = [
    "date", 
    "time", 
    "timestamp", 
    "datetime", 
    "period", 
    "trading_date",
    "trade_date", 
    "valuedate", 
    "value_date",
]
 
_RETURN_KEYWORDS: list[str] = [
    "return", 
    "ret", 
    "rtn", 
    "chg", 
    "change", 
    "pct", 
    "pct_change",
    "daily_return", 
    "log_return", 
    "log_ret",
]
 
_PRICE_KEYWORDS: list[str] = [
    "close", 
    "adj", 
    "price", 
    "last", 
    "settle", 
    "settlement",
    "adj_close", 
    "adjusted", 
    "nav",
]

#Interal helpers
def _load_csv(filepath: str) -> pd.DataFrame:
    #Read CSV and normalise all column names (strip + lowercase).
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df
 
 
def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    #Detect a date/time column by checking known aliases.
    #The detection is done AFTER column normalisation, so user variations like 'Date', ' DATE ', 'Trading Date' all match.
    for alias in _DATE_ALIASES:
        if alias in df.columns:
            df[alias] = pd.to_datetime(df[alias])
            df = df.sort_values(alias).set_index(alias)
            df.index.name = "date"
            return df
 
    #Error catching: if the first column parses as dates, use it
    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col])
        df[first_col] = parsed
        df = df.sort_values(first_col).set_index(first_col)
        df.index.name = "date"
        return df
    except Exception:
        pass
 
    raise ValueError(
        f"No valid date column found. "
        f"Known aliases: {_DATE_ALIASES}. "
        f"Columns present: {list(df.columns)}"
    )

def _classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:

    #Classify every non-index column as:
    #   - return_cols  : already contains return data
    #   - price_cols   : contains price / level data
    #   - unknown_cols : numeric but ambiguous (treated as prices if no price_cols found)

    return_cols: list[str] = []
    price_cols: list[str] = []
    unknown_cols: list[str] = []
 
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in _RETURN_KEYWORDS):
            return_cols.append(col)
        elif any(kw in col_lower for kw in _PRICE_KEYWORDS):
            price_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unknown_cols.append(col)
 
    return return_cols, price_cols, unknown_cols

def _extract_ticker(col: str) -> str:
    #Derive a clean ticker symbol from a column name.
    #Examples:
    #   'aapl_close'     -> 'aapl'
    #   'msft_adj_close' -> 'msft'
    #   'aapl_return'    -> 'aapl'
    #   'aapl'           -> 'aapl'

    parts = col.split("_")
    noise = set(_RETURN_KEYWORDS + _PRICE_KEYWORDS + ["log", "daily", "weekly", "monthly"])
    cleaned = [p for p in parts if p not in noise]
    return "_".join(cleaned) if cleaned else col

def _rename_to_tickers(df: pd.DataFrame) -> pd.DataFrame:
    #Rename columns to clean ticker symbols.
    return df.rename(columns={col: _extract_ticker(col) for col in df.columns})

#Public API
def init_dataframe(
    filepath: str,
    return_type: str = "linear",
) -> pd.DataFrame:
    #Load a CSV and return a clean DataFrame of **returns** indexed by date.

    """
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    return_type : {"linear", "log"}
        Which return formula to use when the CSV contains prices but not returns.
        - "linear" : r_t = (P_t / P_{t-1}) - 1   (simple / arithmetic returns)
        - "log"    : r_t = ln(P_t / P_{t-1})       (log / continuously-compounded)
        Ignored when the CSV already contains return columns.
 
    Returns
    -------
    pd.DataFrame
        Columns are clean ticker symbols; index is a DatetimeIndex named 'date'.
        NaN rows are dropped.
    """

    if return_type not in ("linear", "log"):
        raise ValueError(f"return_type must be 'linear' or 'log', got '{return_type}'.")
 
    df = _load_csv(filepath)
    df = _set_datetime_index(df)
 
    return_cols, price_cols, unknown_cols = _classify_columns(df)
 
    if return_cols:
        #CSV already has returns -> use them directly
        result = df[return_cols].copy()
        result = _rename_to_tickers(result)
        return result.dropna()
 
    candidate_price_cols = price_cols if price_cols else unknown_cols
 
    if not candidate_price_cols:
        raise ValueError(
            "CSV contains no identifiable return or price columns. "
            f"Columns found: {list(df.columns)}"
        )
 
    prices = df[candidate_price_cols].copy()
 
    #Route to calculations.py (Numba-accelerated)
    if return_type == "log":
        returns = compute_log_returns(prices)
    else:
        returns = compute_linear_returns(prices)
 
    returns = _rename_to_tickers(returns)
    return returns.dropna()
 
 
def load_prices(filepath: str) -> pd.DataFrame:
    #Load and return the raw price DataFrame (no return computation).
    #Useful for charting or pre-processing before calling init_dataframe.
 
    """
    Returns
    -------
    pd.DataFrame
        Columns are clean ticker symbols; index is a DatetimeIndex named 'date'.
    """
    
    df = _load_csv(filepath)
    df = _set_datetime_index(df)
 
    _, price_cols, unknown_cols = _classify_columns(df)
    candidate_cols = price_cols if price_cols else unknown_cols
 
    if not candidate_cols:
        raise ValueError("No numeric price columns found in the CSV.")
 
    prices = df[candidate_cols].copy()
    prices = _rename_to_tickers(prices)
    return prices.dropna()