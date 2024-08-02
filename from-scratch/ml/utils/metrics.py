import numpy as np


def mse(y, y_):
    return np.mean((y - y_) ** 2)


def mae(y, y_):
    return np.mean(np.abs(y - y_))


def rmse(y, y_):
    return np.sqrt(np.mean((y - y_) ** 2))


def r2(y, y_):
    y_mean = np.mean(y)
    return 1 - (np.sum((y - y_) ** 2) / np.sum((y - y_mean) ** 2))


def mape(y, y_):
    return (100 / len(y)) * np.sum(np.abs((y - y_) / y))
