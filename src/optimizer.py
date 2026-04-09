import numpy as np
from scipy.optimize import minimize
from metrics import portfolio_performance

def optimize_portfolio(returns, objective="sharpe", bounds=None, constraints=[]):
    n_assets = returns.shape[1]
    init_guess = np.ones(n_assets) / n_assets

    def neg_sharpe(weights):
        return -portfolio_performance(weights, returns)[2]

    def min_vol(weights):
        return portfolio_performance(weights, returns)[1]

    def max_return(weights):
        return -portfolio_performance(weights, returns)[0]

    if objective == "sharpe":
        func = neg_sharpe
    elif objective == "volatility":
        func = min_vol
    elif objective == "return":
        func = max_return
    else:
        raise ValueError("Unknown objective")

    result = minimize(
        func,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x
