import numpy as np

def compute_returns(price_df):
    return price_df.pct_change().dropna()

def annualized_return(returns, periods=252):
    return returns.mean() * periods

def annualized_volatility(returns, periods=252):
    return returns.std() * np.sqrt(periods)

def sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    excess = annualized_return(returns, periods) - risk_free_rate
    vol = annualized_volatility(returns, periods)
    return excess / vol

def portfolio_performance(weights, returns):
    port_return = np.dot(weights, annualized_return(returns))
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return / port_vol
    return port_return, port_vol, sharpe
