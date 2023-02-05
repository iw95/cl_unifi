import numpy as np
import torch


class LSTM:
    def __init__(self, dim, h, learning_rate=0.01):
        self.d = dim
        self.h = h
        self.lr = learning_rate
        # MODEL PARAMs TO BE LEARNED:
        # forget gate's activation vector
        self.Wf = torch.eye(h, dim, requires_grad=True)
        self.Uf = torch.eye(h, h, requires_grad=True)
        self.bf = torch.rand(h, requires_grad=True)
        # input/update gate's activation vector
        self.Wi = torch.eye(h, dim, requires_grad=True)
        self.Ui = torch.eye(h, h, requires_grad=True)
        self.bi = torch.rand(h, requires_grad=True)
        # output gate's activation vector
        self.Wo = torch.eye(h, dim, requires_grad=True)
        self.Uo = torch.eye(h, h, requires_grad=True)
        self.bo = torch.rand(h, requires_grad=True)
        # cell input activation vector
        self.Wc = torch.eye(h, dim, requires_grad=True)
        self.Uc = torch.eye(h, h, requires_grad=True)
        self.bc = torch.rand(h, requires_grad=True)

        pass

    def forward(self, x):
        # start with h and c as 0
        h = torch.zeros((x.shape[0], self.h))
        c = torch.zeros(self.d)
        # compute forward pass through lstm cells
        for t, xt in enumerate(x):
            h[t], c = self.forward_cell(xt, h[t-1 % h.shape[0]], c) # save hidden state vectors and use 0 as first
        return h

    def forward_cell(self, x, h, c):
        f = torch.nn.Sigmoid(self.Wf @ x + self.Uf @ h + self.bf)  # forget gate's activation vector
        i = torch.nn.Sigmoid(self.Wi @ x + self.Ui @ h + self.bi)  # input gate's activation vector
        o = torch.nn.Sigmoid(self.Wo @ x + self.Uo @ h + self.bo)  # output gate's activation vector
        s = torch.nn.Tanh(self.Wc @ x + self.Uc @ h + self.bc)  # cell input activation vector
        c_new = f * c + i * s  # TODO check dimensions
        h_new = o * torch.nn.Tanh(c_new)
        return h_new, c_new

    def predict(self, x):  # return final hidden vector to further use TODO necessary?
        h = self.forward(x)
        return h[-1]

    def backward(self): # TODO understand autograd
        gradient = torch.autograd()
        return gradient

    def grad_descend(self, gradient): # TODO understand autograd
        self.Wf += self.lr * gradient['Wf']
