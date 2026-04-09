def get_user_preferences():
    print("Choose optimization goal:")
    print("1 - Max Sharpe Ratio")
    print("2 - Min Volatility")
    print("3 - Max Return")

    choice = input("Enter choice: ")

    mapping = {
        "1": "sharpe",
        "2": "volatility",
        "3": "return"
    }

    min_weight = float(input("Minimum allocation per asset (e.g. 0.0): "))
    max_weight = float(input("Maximum allocation per asset (e.g. 0.5): "))

    allow_short = input("Allow short selling? (y/n): ") == "y"

    return {
        "objective": mapping.get(choice, "sharpe"),
        "min_weight": min_weight,
        "max_weight": max_weight,
        "long_only": not allow_short
    }
