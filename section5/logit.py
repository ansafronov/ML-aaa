import numpy as np


class LogisticRegression:

    def __init__(self, max_iter=5e3, lr=0.04, tol=0.001, l1_coef=0.1):


        '''
        max_iter – максимальное количеств
        '''

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.l1_coef = l1_coef

        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):

        '''
        Обучение модели.

        X_train – матрица объектов для обучения
        y_train – ответы на объектах для обучения

        '''

        n, m = X_train.shape

        self.weights = np.zeros((m, 1))
        self.bias = 0

        n_iter = 0
        gradient_norm = np.inf

        while n_iter < self.max_iter and gradient_norm > self.tol:

            dJdw, dJdb = self.grads(X_train, y_train)
            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            self.weights = self.weights - self.lr * dJdw
            self.bias = self.bias - self.lr * dJdb

            n_iter += 1

        return self

    def predict(self, X):

        '''
        Метод возвращает предсказанную метку класса на объектах X
        '''

        return np.where(self.predict_proba(X) > 0.5, 1, 0)


    def predict_proba(self, X):

        '''
        Метод возвращает вероятность класса 1 на объектах X
        '''
        return self.sigmoid(X @ self.weights + self.bias)

    def grads(self, X, y):

        '''
        Рассчёт градиентов
        '''
        y_hat = self.predict_proba(X)

        dJdw = ((y_hat - y) * X).mean(axis=0, keepdims=True).T + self.l1_coef * np.sign(self.weights)
        dJdb = (y_hat - y).mean()

        return dJdw, dJdb

    @staticmethod
    def sigmoid(x):
        '''
        Сигмоида от x
        '''
        return 1 / (1 + np.exp(-x))


def read_input():
    n, m = map(int, input().split())
    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    return x_train, y_train


def solution():
    x_train, y_train = read_input()

    model = LogisticRegression(max_iter=5e3, lr=0.04, l1_coef=0.1)
    model.fit(x_train, y_train)

    all_weights = [model.bias] + list(model.weights.flatten())
    result = ' '.join(map(lambda x: str(float(x)), all_weights))
    print(result)

solution()