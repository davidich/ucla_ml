from sklearn.datasets import *
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

min_samples_split = 10
min_samples_leaf = 5

data = load_wine()
X = data.data
y = data.target
labels = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def test_predictor(predictor, X_train, X_test, y_train, y_test):
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    method = getattr(predictor, "accuracy_score", None)
    has_print_accuracy = callable(method)

    print("-" * 60)
    print(F"'{predictor.__class__.__name__}' REPORT:")
    print("-" * 60)
    if has_print_accuracy:
        method(X_test, y_test)
    else:
        print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("")


dt = DecisionTree(min_samples_split, min_samples_leaf)
test_predictor(dt, X_train, X_test, y_train, y_test)

from random_forest import RandomForest

rf = RandomForest(min_samples_split, min_samples_leaf, 100)
test_predictor(rf, X_train, X_test, y_train, y_test)

from stacked_classifier import StackedClassifier

sc = StackedClassifier(train_size=0.7)
test_predictor(sc, X_train, X_test, y_train, y_test)
