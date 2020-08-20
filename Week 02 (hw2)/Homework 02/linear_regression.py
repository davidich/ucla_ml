import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler
from utils import add_bias_feature
from utils_linear_regression import h
from gradient_descent import GradientDescent


class MyLinearRegression(BaseEstimator):

    def __init__(self, max_iterations=1000, learning_rate=0.001, penalty='', alpha=1.0, scale_data=False):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.scale_data = scale_data

        if scale_data:
            self.scaler = StandardScaler()

    def fit(self, x, y, plot_cost=False):
        x, y = check_X_y(x, y)

        if hasattr(self, "scaler"):
            x = self.scaler.fit_transform(x)

        x = add_bias_feature(x)

        gd = GradientDescent(mode='linear',
                             track_cost=plot_cost,
                             learning_rate=self.learning_rate,
                             penalty=self.penalty,
                             alpha=self.alpha,
                             max_iterations=self.max_iterations)

        theta = gd.find_theta(x, y)

        if plot_cost:
            cost_x, cost_y = gd.get_cost_history()
            plt.plot(cost_x, cost_y, "r-")
            plt.title(F"Cost (last value={gd.last_cost:0.6f})")
            plt.show()

        self._theta = theta
        self.intercept_ = theta[0, 0]
        self.coef_ = theta[1:].reshape(len(theta) - 1)

        check_is_fitted(self, attributes=['intercept_', 'coef_'])
        return self

    def predict(self, x):
        x = check_array(x)

        if hasattr(self, "scaler"):
            x = self.scaler.transform(x)

        x = add_bias_feature(x)

        return h(x, self._theta).reshape(len(x))
