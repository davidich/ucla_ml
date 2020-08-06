import numpy as np
import matplotlib.pyplot as plt
import math
from utils import add_bias_feature
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from utils_logistic_regression import h
from gradient_descent import GradientDescent


class MyLogisticRegression(BaseEstimator):
    def __init__(self, max_iterations=1000, learning_rate=0.001, penalty='', alpha=1.0, scale_data=False):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.scale_data = scale_data
        self._scaler = None
        self._class_data = {}

    def fit(self, x, y, plot_cost=False):
        x, y = check_X_y(x, y)

        if self.scale_data:
            self._scaler = StandardScaler()
            x = self._scaler.fit_transform(x)

        x = add_bias_feature(x)

        # one-vs-all
        for class_type in set(y):
            cur_y = np.ones(len(y))
            cur_y[y != class_type] = 0
            self._class_data[class_type] = {"y": cur_y}

            gd = GradientDescent(mode='logistic',
                                 track_cost=plot_cost,
                                 learning_rate=self.learning_rate,
                                 penalty=self.penalty,
                                 alpha=self.alpha,
                                 max_iterations=self.max_iterations)

            cur_theta = gd.find_theta(x, cur_y)
            self._class_data[class_type] = {"theta": cur_theta}

            if plot_cost:
                cost_x, cost_y = gd.get_cost_history()
                plt.plot(cost_x, cost_y, "r-")
                plt.title(F"Cost for class '{class_type}' (last value={gd.last_cost:0.6f})")
                plt.show()

        coef = [self._class_data[key]["theta"][1:, 0] for key in self._class_data]
        intercept = [self._class_data[key]["theta"][0, 0] for key in self._class_data]

        self.coef_ = np.array(coef[1:]) if len(coef) == 2 else np.array(coef)
        self.intercept_ = np.array(intercept[1:]) if len(intercept) == 2 else np.array(intercept)

        check_is_fitted(self, attributes=['intercept_', 'coef_'])
        return self

    def predict(self, x):
        if self.scale_data:
            x = self._scaler.transform(x)

        x = add_bias_feature(x)

        return np.array([self.predict_single_(x_) for x_ in x])

    def predict_proba(self, x):
        if self.scale_data:
            x = self._scaler.transform(x)

        x = add_bias_feature(x)

        p = np.empty((len(x), len(self._class_data)))
        for i in range(len(x)):
            x_ = x[i]
            p[i, :] = [h(x_, self._class_data[c]["theta"])[0] for c in self._class_data]
        return p

    def predict_single_(self, x):
        max_proba = -math.inf
        best_match = None
        for (ind, class_type) in enumerate(self._class_data):
            theta = self._class_data[class_type]["theta"]
            proba = h(x, theta)
            if max_proba < proba:
                max_proba = proba
                best_match = class_type

        return best_match
