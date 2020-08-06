import numpy as np
import operator
from collections import Counter, namedtuple

XY = namedtuple("XY", ["x", "y"])
Distance = namedtuple("DistY", ["distance", "y"])


class MyKNN:

    def __init__(self,
                 type: str,
                 k=3,
                 distance='euclidean',
                 use_weights=False):
        """
        Constructs KNN model
        :param type: type of prediction. Supported values: 'classifier',  'regressor'
        :param k: number of nearest neighbors to account
        :param distance: name of the algorithm to calculate distance. Supported values: 'manhattan' and 'euclidean'
        """
        # validate values
        if type != 'classifier' and type != 'regressor':
            raise ValueError(F"'type' should be either 'classifier' or 'regressor'")
        if distance != 'manhattan' and distance != 'euclidean':
            raise ValueError(F"'distance' should be either 'manhattan' or 'euclidean'")

        # store hyper-parameters
        self.type = type
        self.k = k
        self.distance = distance
        self.use_weights = use_weights

        # prepare model data
        self.data = None

    def _dist(self,
              x1: np.array,
              x2: np.array) -> float:
        """
        Computes distance between two vectors
        :param x1: first vector
        :return: distance
        """
        if self.distance == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            return -1

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data = [XY(X[i], y[i]) for i in range(len(y))]

    def predict(self, x):
        return np.array([self.predict_single(xx) for xx in x])

    def predict_single(self, x):
        dist = [Distance(y=d.y, distance=self._dist(d.x, x)) for d in self.data]
        sort = sorted(dist, key=lambda d: d.distance)
        nn = [sort[i] for i in range(self.k)]

        if self.type == 'classifier':
            if self.use_weights:
                w = self._find_weights(nn)
                d = dict()
                for i in range(self.k):
                    if nn[i].y in d:
                        d[nn[i].y] += w[i]
                    else:
                        d[nn[i].y] = w[i]

                return max(d.items(), key=operator.itemgetter(1))[0]
            else:
                return Counter(nn).most_common(1)[0][0].y
        elif self.type == 'regressor':
            if self.use_weights:
                w = self._find_weights(nn)
                weighted_mean = np.sum([nn[i].y * w[i] for i in range(self.k)])
                return weighted_mean
            else:
                return np.mean([n.y for n in nn])

    def _find_weights(self, nn: np.array):
        # look if we have a perfect match first
        perfect_match = next((i for i in range(self.k) if nn[i].distance == 0), None)
        if perfect_match is not None:
            weights = np.zeros(self.k)
            weights[perfect_match] = 1
            return weights

        """       Example of calculating weights:
        ------------------------------------------------------
                                         K1      K2       K3
        distance (D):                   1.00    3.00     6.00
        distance total (DT):  10.00
        proximity (P=DT/D):            10.00    3.(3)    1.(6)
        proximity total (PT): 15.00
        weight (P/PT):                  0.(6)    0.(2)     0.(1)
        """
        dt = np.sum([n.distance for n in nn])
        p = np.array([dt / n.distance for n in nn])
        pt = np.sum(p)
        weights = p / pt
        return weights
