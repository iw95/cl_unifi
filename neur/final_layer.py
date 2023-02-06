import torch


class Final:
    def __init__(self, activation):
        # defining activation function
        self.activation = activation

    def forward(self, x):
        return self.activation(x)