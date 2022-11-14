import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z: np.array):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


def logloss(y, y_hat):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).sum()


def cross_entropy(y, y_hat):
    return ( - y * np.log(y_hat)).sum()


def solution():
    n, k = map(int, input().split())
    y = np.array([list(map(int, input().split())) for _ in range(n)])
    z = np.array([list(map(float, input().split())) for _ in range(n)])

    y_hat_multilabel = sigmoid(z)
    y_hat_multiclass = softmax(z)

    logloss_value = logloss(y, y_hat_multilabel)
    crossentropy_value = cross_entropy(y, y_hat_multiclass)

    logloss_value = str(np.round(logloss_value, 3))
    crossentropy_value = str(np.round(crossentropy_value, 3))
    print(logloss_value + ' ' + crossentropy_value)


solution()