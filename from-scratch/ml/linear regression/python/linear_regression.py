import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return f"LinearRegression class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        # Добавляем к матрице объектов-признаков фиктивный единичный признак перед первым столбцом
        ones_column = pd.Series([1] * X.shape[0])
        X = pd.concat([ones_column, X], axis=1)

        # Создаем вектор весов, состоящий из единиц
        self.weights = pd.Series([1] * X.shape[1])

        # Делаем n_iter итераций
        for i in range(self.n_iter):
            # Матрица объектов-признаков умножается на вектор весов
            # Матрица остается в исходном виде, а вектор весов рассматривается как вектор-столбец
            # (numpy делает это автоматически)
            y_ = self.predict(X)
            # Считаем ошибку
            loss = self.__calc_error(y, y_)
            # Считаем вектор-градиент
            grad = self.__calc_grad(X, y, y_)
            # Обновляем веса
            self.__forward(grad)

            # Выводим лог, если указан verbose
            if verbose and (i % verbose) == 0:
                print(f"{i} | loss: {loss}")

    def __calc_error(self, y, y_):
        # Считаем MSE
        return np.mean((y_ - y) ** 2)

    def __calc_grad(self, X, y, y_):
        # Считаем градиент для MSE
        return (2 / X.shape[0]) * np.dot((y_ - y), X)

    def __forward(self, grad):
        # Корректируем веса в сторону антиградиента с учетом скорости обучения learning_rate
        self.weights -= self.learning_rate * grad

    def predict(self, X):
        # Считаем предсказания: X * w
        return np.dot(X, self.weights)

    def get_coef(self):
        # Возвращает вектор весов
        return self.weights.values[1:]
