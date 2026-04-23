#calculations.py
#Core numerical calculations for portfolio optimization.
#Heavy loops are JIT-compiled with Numba (@njit) for near-C performance.

import numpy as np
import pandas as pd
from numba import njit


#Numba-accelerated kernels (operate on raw NumPy arrays)

@njit(cache=True)
def _linear_returns_kernel(prices: np.ndarray) -> np.ndarray:
    #Compute simple (linear / arithmetic) returns for a 2-D price matrix.

    """
    r[t, j] = prices[t, j] / prices[t-1, j] - 1

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)  float64

    Returns
    -------
    np.ndarray, shape (T-1, N)  float64
    """

    T, N = prices.shape
    out = np.empty((T - 1, N), dtype=np.float64)
    for t in range(1, T):
        for j in range(N):
            if prices[t - 1, j] != 0.0:
                out[t - 1, j] = prices[t, j] / prices[t - 1, j] - 1.0
            else:
                out[t - 1, j] = np.nan
    return out


@njit(cache=True)
def _log_returns_kernel(prices: np.ndarray) -> np.ndarray:
    #Compute log (continuously-compounded) returns for a 2-D price matrix.

    """
    r[t, j] = ln(prices[t, j] / prices[t-1, j])

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)  float64

    Returns
    -------
    np.ndarray, shape (T-1, N)  float64
    """

    T, N = prices.shape
    out = np.empty((T - 1, N), dtype=np.float64)
    for t in range(1, T):
        for j in range(N):
            ratio = prices[t, j] / prices[t - 1, j]
            if ratio > 0.0:
                out[t - 1, j] = np.log(ratio)
            else:
                out[t - 1, j] = np.nan
    return out


@njit(cache=True)
def _portfolio_return_kernel(
    returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    #Compute the weighted portfolio return series.

    """
    Parameters
    ----------
    returns : np.ndarray, shape (T, N)  float64
    weights : np.ndarray, shape (N,)    float64  (must sum to 1)

    Returns
    -------
    np.ndarray, shape (T,)  float64
    """

    T, N = returns.shape
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        s = 0.0
        for j in range(N):
            s += returns[t, j] * weights[j]
        out[t] = s
    return out


@njit(cache=True)
def _covariance_kernel(returns: np.ndarray) -> np.ndarray:
    #Compute the sample covariance matrix from a (T x N) returns array.
    #Uses the unbiased estimator (divides by T-1).

    """
    Parameters
    ----------
    returns : np.ndarray, shape (T, N)  float64

    Returns
    -------
    np.ndarray, shape (N, N)  float64
    """

    T, N = returns.shape
    means = np.zeros(N, dtype=np.float64)
    for j in range(N):
        s = 0.0
        for t in range(T):
            s += returns[t, j]
        means[j] = s / T

    cov = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (returns[t, i] - means[i]) * (returns[t, j] - means[j])
            cov[i, j] = s / (T - 1)
            cov[j, i] = cov[i, j]
    return cov

# Public pandas-facing wrappers

def compute_linear_returns(prices: pd.DataFrame) -> pd.DataFrame:
    #Compute simple (linear) returns from a price DataFrame.

    """
    r_t = P_t / P_{t-1} - 1

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are tickers; index is a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Same columns; index starts one period later.
    """

    arr = prices.to_numpy(dtype=np.float64)
    result = _linear_returns_kernel(arr)
    return pd.DataFrame(result, index=prices.index[1:], columns=prices.columns)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    #Compute log (continuously-compounded) returns from a price DataFrame.

    """
    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are tickers; index is a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Same columns; index starts one period later.
    """

    arr = prices.to_numpy(dtype=np.float64)
    result = _log_returns_kernel(arr)
    return pd.DataFrame(result, index=prices.index[1:], columns=prices.columns)


def calculate_portfolio_returns(
    df: pd.DataFrame,
    weights: dict,
) -> pd.Series:
    #Compute the weighted portfolio return series.

    """
    Parameters
    ----------
    df : pd.DataFrame
        Returns DataFrame with ticker columns.
    weights : dict
        {ticker: weight} mapping. Weights are auto-normalised to sum to 1.

    Returns
    -------
    pd.Series
        Portfolio return series with the same DatetimeIndex as *df*.
    """

    missing = [col for col in weights if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    total = sum(weights.values())
    if total == 0:
        raise ValueError("Sum of weights cannot be zero.")

    tickers = list(weights.keys())
    w = np.array([weights[t] / total for t in tickers], dtype=np.float64)
    arr = df[tickers].to_numpy(dtype=np.float64)

    result = _portfolio_return_kernel(arr, w)
    return pd.Series(result, index=df.index, name="portfolio_return")


def compute_covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    #Compute the sample covariance matrix of returns (Numba-accelerated).

    """
    Parameters
    ----------
    df : pd.DataFrame
        Returns DataFrame with ticker columns.

    Returns
    -------
    pd.DataFrame
        (N x N) covariance matrix.
    """

    arr = df.to_numpy(dtype=np.float64)
    cov = _covariance_kernel(arr)
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def compute_mean_returns(df: pd.DataFrame) -> pd.Series:
    #Compute the mean return for each ticker.

    """
    Parameters
    ----------
    df : pd.DataFrame
        Returns DataFrame.

    Returns
    -------
    pd.Series
        Mean returns indexed by ticker.
    """

    return df.mean()