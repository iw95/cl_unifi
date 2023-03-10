import torch


class LSTM:
    """
    Long Short Term Memory Unit as type of Recurrent Neural Network to perform recurrent learning.
    LSTM uses gate mechanisms to resolve the vanishing gradient problem and remember information from a lot of steps ago.
    Computing gradient while learning using pytorch's autograd.
    """
    def __init__(self, dim, h, learning_rate=0.01):
        """
        Initialising model parameters such as weight matrices and biases.
        :param dim: Feature size of input
        :param h: Size of hidden vector
        :param learning_rate: learning rate
        """
        torch.manual_seed(12345)
        self.d = dim
        self.h = h
        self.lr = learning_rate
        # MODEL PARAMs TO BE LEARNED:
        # indicate to autograd using 'requires_grad=True'
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
        """
        Perform forward propagation through recurrent graph of LSTM cells.
        Pass forward h and c every time step out dimension 'time_steps' in x.
        Compute pass for full batch at once instead of iteratively.
        :param x: data batch of shape (time_steps, batch_size, feat_size)
        :return: hidden cell of last time step
        """
        # start with h and c as 0
        h = torch.zeros((x.shape[1], self.h))
        c = torch.zeros(x.shape[1], self.h)
        # compute forward pass through lstm cells
        for t, xt in enumerate(x):
            # xt of size (batch_size, feat_size)
            old_idx = t-1 % h.shape[0]
            h, c = self.forward_cell(xt, h, c)
        return h

    def forward_cell(self, x, h, c):
        """
        LSTM cell with gate mechanics
        :param x: input of batch at this time step, shape (batch_size, feat_size, 1)
        :param h: hidden state of last step, shape (hidden_size)
        :param c: cell state of last time step, shape (hidden_size)
        :return: new hidden state and cell state, shape (batch_size, hidden_size)
        """
        sig = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()
        # x of shape (batch_size, feat_size, 1), h & c & b of shape (hidden_size)
        # W of shape (hidden_size, feat_size), U of shape (hidden_size, hidden_size)
        # Expand and reduce dimensions to compute outputs for full batch at once
        x_ = x[:,:,None]
        h_ = h[:,:,None]
        prod_f = (self.Wf @ x_ + self.Uf @ h_)[:,:,0]
        prod_i = (self.Wi @ x_ + self.Ui @ h_)[:,:,0]
        prod_o = (self.Wo @ x_ + self.Uo @ h_)[:,:,0]
        prod_c = (self.Wc @ x_ + self.Uc @ h_)[:,:,0]
        f = sig(prod_f + self.bf)  # forget gate's activation vector
        i = sig(prod_i + self.bi)  # input gate's activation vector
        o = sig(prod_o + self.bo)  # output gate's activation vector
        s = tanh(prod_c + self.bc)  # cell input activation vector
        c_new = f * c + i * s
        h_new = o * tanh(c_new)
        return h_new, c_new  # of shape (batch_size, hidden_size)

    def parameters(self):
        """
        :return: Training parameters of LSTM layer
        """
        params = [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf, self.Wc, self.Uc, self.bc, self.Wo, self.Uo, self.bo]
        return params

