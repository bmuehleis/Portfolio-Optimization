from data_loader import load_price_data
from metrics import compute_returns, portfolio_performance
from optimizer import optimize_portfolio
from constraints import weight_sum_constraint, long_only_constraint, bounds
from config import get_user_preferences

def main():
    prices = load_price_data("../data")
    returns = compute_returns(prices)

    prefs = get_user_preferences()

    n_assets = returns.shape[1]

    cons = [weight_sum_constraint()]

    if prefs["long_only"]:
        cons.append(long_only_constraint())

    bnds = bounds(
        n_assets,
        prefs["min_weight"],
        prefs["max_weight"]
    )

    weights = optimize_portfolio(
        returns,
        objective=prefs["objective"],
        bounds=bnds,
        constraints=cons
    )

    port_return, port_vol, sharpe = portfolio_performance(weights, returns)

    print("\nOptimal Portfolio:")
    for asset, weight in zip(prices.columns, weights):
        print(f"{asset}: {weight:.2%}")

    print("\nPerformance:")
    print(f"Return: {port_return:.2%}")
    print(f"Volatility: {port_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    main()
