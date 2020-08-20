from decision_tree import DecisionTree
from sklearn.model_selection import ShuffleSplit
from collections import Counter


class RandomForest:
    def __init__(self, min_split=50, min_leaf=10, forest_size=100):
        self.min_split = min_split
        self.min_leaf = min_leaf
        self.forest_size = forest_size
        self.forest = []

    def fit(self, X, y):
        rs = ShuffleSplit(n_splits=self.forest_size, train_size=0.75, random_state=42)
        for train_index, _ in rs.split(X):
            X_cur = X[train_index]
            y_cur = y[train_index]
            tree = DecisionTree(self.min_split, self.min_leaf)
            tree.fit(X_cur, y_cur)
            self.forest.append(tree)

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        predictions = [tree.predict([x])[0] for tree in self.forest]
        most_common = Counter(predictions).most_common(1)[0][0]
        return most_common
