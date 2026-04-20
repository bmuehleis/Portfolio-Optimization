import pandas as pd

def calculate_portfolio_returns(df: pd.DataFrame, weights: dict) -> pd.Series:
    
    # Ensure all weights exist in dataframe
    missing_cols = [col for col in weights if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")
    
    # Normalize weights (safety check)
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero.")

    normalized_weights = {k: v / total_weight for k, v in weights.items()}

     # Start portfolio return series
    portfolio_return = pd.Series(0.0, index=df.index)

    # Weighted sum of returns
    for col, weight in normalized_weights.items():
        portfolio_return += df[col] * weight

    return portfolio_return