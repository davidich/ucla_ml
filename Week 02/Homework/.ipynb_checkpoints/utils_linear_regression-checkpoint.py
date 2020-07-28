import numpy as np


def h(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Linear regression hypothesis function.
    :param x: Features. N*M matrix. N-instances, M-features
    :param theta: Bias + Weights; M*1 column vector
    :return: Prediction (y_hat). N*1 column vector
    """
    return x @ theta


def cost(y_h: np.ndarray, y: np.ndarray) -> float:
    """
    Cost function for linear regression.
    :param y_h: Hypothesis value; M*1 column vector
    :param y: Target value; M*1 column vector
    :return:
    """
    return np.average((y_h - y) ** 2)