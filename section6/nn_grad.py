import torch
import numpy as np


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    return x / torch.sum(torch.exp(x))


X = torch.tensor(
    [
        [0.333, 0.5, 5.],
        [0.666, 0.25, 10.],
        [0.999, 0.1, 1.],
        [0.1, 0.1, 0.2]
    ],
    requires_grad=False
)

w_1 = torch.tensor(
    [
        [-1.5, -0.5],
        [0., 1.],
        [1.5, 3.0]
    ],
    requires_grad=True
)

w_2 = torch.tensor(
    [
        [2.5, 0.5, -1.5],
        [3.5, 1.5, 1.5]
    ],
    requires_grad=True
)

b_1 = torch.tensor([[0.0, 0.0]], requires_grad=True)
b_2 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

a_1 = sigmoid(X @ w_1 + b_1)
z_2 = a_1 @ w_2 + b_2

target = torch.tensor(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ],
    dtype=torch.float
)

loss = torch.nn.CrossEntropyLoss()
loss_value = loss(z_2, target)

loss_value.backward()

res = loss_value + b_1.grad[0][0]

print(res)

