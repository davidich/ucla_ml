from collections import Counter
import numpy as np
import math


class DecisionTreeNode:
    def __init__(self, data, target, min_split=50, min_leaf=1):
        self.left = None
        self.right = None
        self.data = data
        self.target = target
        self.split_feature = None
        self.split_value = None
        self.min_split = min_split
        self.min_leaf = min_leaf
        self.is_leaf = False
        self.node_class = None

    def print_info(self):
        print(f"""Gini: {_compute_gini(self.target)} Distribution: {Counter(self.target)} 
              Split: {self.split_feature} Value: {self.split_value} Leaf: {self.is_leaf}""")

    def split(self):
        counter = Counter(self.target)
        most_common = counter.most_common(1)
        self.node_class = most_common[0][0]  # get the most common class

        if self.min_split > len(self.data):  # If we can't split anymore, we mark the node as leaf and stop.
            self.is_leaf = True
            return False

        final_cost = math.inf
        final_left_indexes = None
        final_right_indexes = None

        feature_count = self.data.shape[1]
        for feature_index in range(feature_count):
            curr_feature = self.data[:, feature_index]
            order = np.argsort(curr_feature)
            ordered_target = [self.target[i] for i in order]

            minimal_cost, split_index = self._find_best_split(ordered_target)

            if split_index is None:
                print("here we go")
                minimal_cost, split_index = self._find_best_split(ordered_target)


            if final_cost >= minimal_cost:
                self.split_feature = feature_index  # Saves the name of the feature to split
                self.split_value = self.data[order[split_index]][feature_index]  # Saves the value fo the threshold
                final_cost = minimal_cost
                final_left_indexes = order[:split_index]  # saves the left split
                final_right_indexes = order[split_index:]  # saves the right split

        if final_left_indexes is None or final_right_indexes is None or final_cost == 0:
            self.is_leaf = True
            return False

        left_data = self.data[final_left_indexes]
        left_target = self.target[final_left_indexes]
        self.left = DecisionTreeNode(left_data, left_target, self.min_split, self.min_leaf)

        right_data = self.data[final_right_indexes]
        right_target = self.target[final_right_indexes]
        self.right = DecisionTreeNode(right_data, right_target, self.min_split, self.min_leaf)

        return True

    def _find_best_split(self, target):
        split_index = None
        minimal_cost = math.inf
        instance_count = len(target)
        start = self.min_leaf
        end = instance_count - start + 1
        for i in range(start, end):
            left = target[:i]
            right = target[i:]
            left_gini = _compute_gini(left)
            right_gini = _compute_gini(right)
            cost = _compute_cost(len(left), len(right), instance_count, left_gini, right_gini)

            if minimal_cost >= cost:
                minimal_cost = cost
                split_index = i

        return minimal_cost, split_index


def _compute_gini(target):
    n = len(target)
    proportions = [count / n for count in Counter(target).values()]
    gini = 1 - sum([p ** 2 for p in proportions])
    return gini


def _compute_cost(m_left, m_right, m, gini_score_left, gini_score_right):
    return (m_left / m) * gini_score_left + (m_right / m) * gini_score_right
