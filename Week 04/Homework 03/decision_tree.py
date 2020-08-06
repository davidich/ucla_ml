from collections import deque
from decision_tree_node import DecisionTreeNode


class DecisionTree:
    def __init__(self, min_split=50, min_leaf=10, verbose=False):
        self.min_split = min_split
        self.min_leaf = min_leaf
        self.head = None
        self.verbose = verbose

    def fit(self, X, y):
        self.head = DecisionTreeNode(X, y, self.min_split, self.min_leaf)
        queue = deque()
        queue.append(self.head)
        splits = 0
        while len(queue):  # Transverses the tree, level by level, using the BFS algorithm
            n = queue.popleft()
            if n.split():
                queue.append(n.left)
                queue.append(n.right)
                splits += 1
            if self.verbose:
                n.print_info()
        if self.verbose:
            print(f"Performed {splits} splits.")

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x, verbose=False):
        queue = deque()
        queue.append(self.head)

        while len(queue):
            n = queue.popleft()
            if n.is_leaf:  # if the node is a leaf, we make a prediction
                return n.node_class
            if x[n.split_feature] > n.split_value:  # if larger than the threshold, go to the right
                if verbose:
                    print(f"Went right. {x[n.split_feature]} larger than the {n.split_value} threshold.")
                queue.append(n.right)
            else:  # otherwise, we go to the left
                if verbose:
                    print(f"Went left. {x[n.split_feature]} larger than the {n.split_value} threshold.")
                queue.append(n.left)
            if verbose:
                n.print_info()
