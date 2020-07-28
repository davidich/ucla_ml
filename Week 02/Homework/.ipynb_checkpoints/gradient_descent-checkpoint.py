import numpy as np
import math
import utils_linear_regression as lin_utils
import utils_logistic_regression as log_utils


class GradientDescent:
    def __init__(self,
                 mode,  # 'linear', 'logistic'
                 learning_rate=0.001,
                 max_iterations=1000,
                 track_cost=False,
                 track_cost_interval=None,
                 penalty='l2',
                 alpha=1.0):
        # validate parameters
        if penalty != '' and penalty != 'l1' and penalty != 'l2':
            raise ValueError(F"'penalty' should be either '', 'l1' or 'l2'")
        if mode != 'linear' and mode != 'logistic':
            raise ValueError(F"'mode' should be either 'linear' or 'logistic'")

        # store parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.track_cost = track_cost
        self.track_cost_interval = track_cost_interval
        self.penalty = penalty
        self.alpha = alpha
        self.mode = mode
        self.utils = lin_utils if mode == 'linear' else log_utils
        self.cost_history = []
        self.last_cost = math.inf
        self.last_cost_improvement = math.nan

        if track_cost_interval is None:
            self.track_cost_interval = int(max_iterations / 100)

    def find_theta(self, x, y):
        y = np.array(y).reshape(len(y), 1)
        n, m = x.shape
        theta = np.zeros((m, 1))
        x_t = x.T

        self.last_cost = math.inf
        for i in range(0, self.max_iterations):
            hypothesis = self.utils.h(x, theta)
            error = hypothesis - y
            gradient = 1 / n * (x_t @ error) + self._calculate_penalty(theta, n)
            theta = theta - self.learning_rate * gradient
            cost = self.utils.cost(hypothesis, y)
            cost_improvement = abs(self.last_cost - cost)

            if cost_improvement < 1e-7:
                print(F"Successful convergence at step #{i}")
                break
            elif math.isnan(cost):
                print(F"Early Convergence step #{i} due broken math. "
                      F"Last cost: {self.last_cost}"
                      F"Last cost improvement: {self.last_cost_improvement}")
                break

            self.last_cost = cost
            self.last_cost_improvement = cost_improvement
            if self.track_cost and i % self.track_cost_interval == 0:
                self.cost_history.append(self.last_cost)

        return theta

    def _calculate_penalty(self, theta, n):
        m = len(theta)

        # Lasso regression
        if self.penalty == 'l1':
            result = np.sign(theta)
            result[0] = 0  # don't penalize bias
            return self.alpha * result
        # Ridge regression (Tikhonov regularization)
        elif self.penalty == 'l2':
            result = theta.copy()
            result[0, 0] = 0  # don't penalize bias
            return self.alpha / n * result

        else:
            return np.zeros((m, 1))

    def get_cost_history(self):
        if not self.track_cost:
            return

        x = np.linspace(0, self.max_iterations, len(self.cost_history))
        y = self.cost_history
        return x, y
