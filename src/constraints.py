def weight_sum_constraint():
    return {'type': 'eq', 'fun': lambda w: sum(w) - 1}

def long_only_constraint():
    return {'type': 'ineq', 'fun': lambda w: w}

def bounds(n_assets, min_w=0.0, max_w=1.0):
    return tuple((min_w, max_w) for _ in range(n_assets))
