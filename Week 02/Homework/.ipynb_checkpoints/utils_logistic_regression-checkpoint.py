import numpy as np


def _sigmoid(t):
    return 1 / (1 + np.exp(-t))


def h(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Logistic regression hypothesis function.
    :param x: Features. N*M matrix. N-instances, M-features
    :param theta: Bias + Weights; M*1 column vector
    :return: Logistic probability (p_hat). N*1 column vector
    """
    return _sigmoid(x @ theta)


def cost(p_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Cost function for logistic regression ("log loss").
    :param p_hat: Hypothesis value; M*1 column vector
    :param y: Target value; M*1 column vector
    :return:
    """

    term1 = y * np.log(p_hat)
    term2 = (1 - y) * np.log(1 - p_hat)
    return -np.average(term1 + term2)
