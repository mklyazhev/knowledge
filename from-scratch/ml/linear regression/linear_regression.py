import numpy as np
import pandas as pd

from ml.utils.metrics import mse, mae, rmse, r2, mape
from ml.utils.data_utils import make_regression_sample


class LinearRegression:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.__metric_function = self.__get_metric_function(metric)
        self.__weights = None
        self.__loss = None
        self.__score = None

    def __str__(self):
        return f"LinearRegression class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        # Добавляем к матрице объектов-признаков фиктивный единичный признак перед первым столбцом
        ones_column = pd.Series([1] * X.shape[0])
        X = pd.concat([ones_column, X], axis=1)

        # Создаем вектор весов, состоящий из единиц
        self.__weights = pd.Series([1] * X.shape[1])

        # Делаем n_iter итераций
        for i in range(1, self.n_iter + 1):
            # Матрица объектов-признаков умножается на вектор весов
            # Матрица остается в исходном виде, а вектор весов рассматривается как вектор-столбец
            # (numpy делает это автоматически)
            y_ = self.__predict(X)
            # Считаем ошибку
            self.__loss = self.__calc_loss(y, y_)
            # Считаем вектор-градиент
            grad = self.__calc_grad(X, y, y_)
            # Обновляем веса
            self.__step(grad)

            if self.metric:
                # Снова делаем предсказание и считаем значение метрики, если она указана
                y_ = self.__predict(X)
                self.__score = self.__metric_function(y, y_)

            # Выводим лог, если указан verbose
            if verbose and (i % verbose) == 0:
                if self.metric:
                    print(f"{i} | loss: {self.__loss}")
                else:
                    print(f"{i} | loss: {self.__loss} | {self.metric}: {self.__score}")

    def __calc_loss(self, y, y_):
        # Считаем MSE
        return np.mean((y_ - y) ** 2)

    def __calc_grad(self, X, y, y_):
        # Считаем градиент для MSE
        return (2 / X.shape[0]) * np.dot((y_ - y), X)

    def __step(self, grad):
        # Корректируем веса в сторону антиградиента с учетом скорости обучения learning_rate
        self.__weights -= self.learning_rate * grad

    def __predict(self, X):
        # Считаем предсказания: X * w
        return np.dot(X, self.__weights)

    def __get_metric_function(self, metric):
        if metric == "mse":
            return mse
        elif metric == "mae":
            return mae
        elif metric == "rmse":
            return rmse
        elif metric == "r2":
            return r2
        elif metric == "mape":
            return mape

    def predict(self, X):
        ones_column = pd.Series([1] * X.shape[0])
        X = pd.concat([ones_column, X], axis=1)
        return self.__predict(X)

    def get_coef(self):
        # Возвращает вектор весов со 2-го элемента,
        # потому что во время обучения добавили на место 1-го элемента
        # свободный коэффициент равный единице
        return self.__weights.values[1:]

    def get_best_score(self):
        # Возвращает значение метрики
        return self.__score


X, y = make_regression_sample()
a = LinearRegression(metric="mae")
a.fit(X, y, True)
print(a.get_best_score())

