import torch


class Dense:
    def __init__(self, dim_feat, dim_out):
        self.W = torch.eye((dim_feat, dim_out), requires_grad=True)
        # data (sz: batch_size,feature_size) @ W (sz: feature_size,output_size) = output (sz: batch_size,output_size)

    def forward(self, x):
        return x @ self.W

    def backward(self):  # TODO understand autograd
        torch.autograd()
