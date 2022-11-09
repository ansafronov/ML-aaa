import numpy as np

x = np.array([1, 2, 3])
y = np.array([1, 0, 1])

w = -1
b = 1

y_hat = w * x + b
sigm = lambda x: 1 / (1 + np.exp(-x))
s = sigm(y_hat)

logloss = np.sum(y * np.log(s) + (1 - y) * np.log(1 - s))
print(logloss)