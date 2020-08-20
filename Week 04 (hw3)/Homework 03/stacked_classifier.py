import numpy as np
from logistic_regression import MyLogisticRegression
from knn import MyKNN
from decision_tree import DecisionTree
from random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class StackedClassifier:
    def __init__(self, train_size=0.7):
        self.predictors = [
            MyLogisticRegression(scale_data=True),
            RandomForest(6, 3, 200),
            SVC()]
        self.blender = MyKNN(type='classifier', use_weights=False)  # DecisionTree(2, 1)
        self.train_size = train_size

    def fit(self, X, y):
        X1, X2, y1, y2 = train_test_split(X, y, train_size=self.train_size)
        for pr in self.predictors:
            pr.fit(X1, y1)

        X_blender = []
        for x in X2:
            row = []
            for p in self.predictors:
                row.append(p.predict([x])[0])
            X_blender.append(row)

        X_blender = np.array(X_blender)
        self.blender.fit(X_blender, y2)

    def predict(self, X):
        blender_input = []
        for x in X:
            row = []
            for p in self.predictors:
                row.append(p.predict([x])[0])
            blender_input.append(row)

        return self.blender.predict(blender_input)

    def accuracy_score(self, X, y):
        y_pred = self.predict(X)
        print("Accuracy scores:")
        print(F" - Blender '{self.blender.__class__.__name__}': {accuracy_score(y, y_pred)}")

        cnt = 0
        for p in self.predictors:
            cnt += 1
            y_pred = p.predict(X)
            print(F" - Predictor {cnt} '{p.__class__.__name__}': {accuracy_score(y, y_pred)}")
