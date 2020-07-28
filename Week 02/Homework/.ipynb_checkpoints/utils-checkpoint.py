def mse(y1, y2):
    n = len(y1)
    se = (y1 - y2) ** 2
    return np.sum(se) / n