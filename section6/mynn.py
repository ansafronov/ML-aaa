import numpy as np
import torch
from torch import nn


class MLPClassifier():
    def __init__(self, input_size, output_size=1):
        # super(MLPClassifier, self).__init__()

        self.flatten = nn.Flatten()

        self.input_layer = nn.Linear(input_size, 16)
        self.input_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(16, 16)
        self.hidden_activation = nn.ReLU()

        self.output = nn.Linear(16, output_size)
        self.output_activation = nn.ReLU()

    def forward(self, x):

        x = self.flatten(x)

        x = self.input_layer(x)
        x = self.input_activation(x)

        x = self.hidden_layer(x)
        x = self.hidden_activation(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


if __name__ == '__main__':
    model = MLPClassifier(input_size=128, output_size=10)
    input_data = torch.tensor(np.random.rand(1, 128)).float()
    print(model(input_data))
