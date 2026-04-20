#optimizing logic using cvxpy

import cvxpy as cp
import numpy as np

from data_loader import init_dataframe, returns_df

#Initialize Dataframe
init_dataframe("data/test.csv")

print(returns_df)