# handle data loading

import pandas as pd

returns_df = None

#Load csv and clean column headers
def _load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.columns = [col.strip().lower() for col in df.columns]

    return df

#Set timestamp correctly for later matching
def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    possible_date_cols = ["date", "time", "timestamp"]

    for col in possible_date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.sort_values(col)
            df = df.set_index(col)
            return df
        raise ValueError("No valid date column found.")

#Fill Retrun Column if already exists or fill Price Column
def _classify_columns(df: pd.DataFrame):
    return_cols = []
    price_cols = []

    for col in df.columns:
        if "return" in col:
            return_cols.append(col)
        elif any(x in col for x in ["close", "adj", "price"]):
            price_cols.append(col)

    return return_cols, price_cols


#Calculate Returns if not already in csv
def _compute_returns(df: pd.DataFrame, return_cols, price_cols):
    result = pd.DataFrame(index=df.index)

    if return_cols:
        # Use existing returns
        for col in return_cols:
            result[col] = df[col]

    elif price_cols:
        # Compute returns from price columns
        for col in price_cols:
            returns = df[col].pct_change()
            result[f"{col}_return"] = returns

    else:
        raise ValueError("No return or price columns found.")

    return result.dropna()

#Clean Column Name for later usage
def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}

    for col in df.columns:
        # Example: aapl_close → aapl
        if "_" in col:
            parts = col.split("_")
            new_cols[col] = parts[0]
        else:
            new_cols[col] = col

    return df.rename(columns=new_cols)

#Initialize Dataframe for global usage
def init_dataframe(filepath: str):
    df = _load_csv(filepath)
    df = _set_datetime_index(df)
    df = _rename_columns(df)

    return_cols, price_cols = _classify_columns(df)
    returns_df = _compute_returns(df, return_cols, price_cols)

    return returns_df