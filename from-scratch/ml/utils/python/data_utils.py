from sklearn.datasets import make_regression, make_classification, make_blobs
import pandas as pd


def make_regression_example():
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]


def make_classification_example():
    X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]


def make_clusterization_example():
    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]
