import numpy as np


def mse(y1, y2):
    n = len(y1)
    se = (y1 - y2) ** 2
    return np.average(se)


def add_bias_feature(x):
    ones = np.ones((len(x), 1))
    return np.hstack((ones, x))
