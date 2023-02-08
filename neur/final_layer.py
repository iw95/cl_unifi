
class Final:
    """
    Class for final layer using only an activation function and no other parameters.
    """
    def __init__(self, activation):
        # defining activation function
        self.activation = activation

    def forward(self, x):
        return self.activation(x)

    def parameters(self):
        return []
