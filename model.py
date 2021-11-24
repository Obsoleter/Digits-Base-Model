from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def logistic_regression(X_train, X_test, y_train, y_test):
    # Cross Validation
    grid = {
        'solver': ['saga', 'sag'],
        'C': [0.5, 1, 10],
        'max_iter': [100, 250, 500, 1000]
    }

    cv = GridSearchCV(LogisticRegression(), grid, cv=10, verbose=2).fit(X_train, y_train)

    params = cv.best_params_
    print(params)

    # Result {'C': 10, 'max_iter': 100, 'solver': 'sag'}
    regr = LogisticRegression(**params)
    regr.fit(X_train, y_train)
    print(regr.score(X_test, y_test))

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(regr, X_test, y_test)
    plt.show()


def svm(X_train, X_test, y_train, y_test):
    # Normalize data
    X_train = scale(X_train)
    X_test = scale(X_test)

    # Cross Validation
    grid = {
        'C': [0.5, 1, 10],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }

    cv = GridSearchCV(SVC(), grid, cv=10, verbose=2).fit(X_train, y_train)

    params = cv.best_params_
    print(params)

    # Result {'C': 0.5, 'gamma': 0.1, 'kernel': 'poly'}
    svm = SVC(**params)
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))

    # CM
    ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test)
    plt.show()


def decision_tree(X_train, X_test, y_train, y_test):
    # Cross Validation
    path = DecisionTreeClassifier().cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    grid = {
        'ccp_alpha': alphas,
        'splitter': ['best', 'random'],
        'criterion': ['gini', 'entropy']
    }

    cv = GridSearchCV(DecisionTreeClassifier(), grid, cv=10, verbose=2).fit(X_train, y_train)

    params = cv.best_params_
    print(params)

    # Result {'ccp_alpha': 0.0007423904974016332}
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)
    print(dt.score(X_test, y_test))

    # CM
    ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test)
    plt.show()


def main():
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Separate data into Training and Testing
    data = train_test_split(X, y)

    # Apply methods
    # logistic_regression(*data)
    svm(*data)
    # decision_tree(*data)


if __name__ == "__main__":
    main()