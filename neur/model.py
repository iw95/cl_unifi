from LSTM import LSTM
from dense import Dense
from final_layer import Final
import torch
import matplotlib.pyplot as plt
import numpy as np
# use layers lstm, dense, softmax
# give prediction
# train


def plot_training_loss(losses, ylabel=''):
    plt.figure()
    plt.plot(np.arange(losses.shape[0]), losses.detach().numpy(), 'k')
    plt.title('Learning curve')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.show()
    t = '_mse' if ylabel[0] in 'mM' else '_cel'
    plt.savefig(f'logs/learn_curve{t}.png')
    plt.close()


# class creates a model with <lstm> LSTM layers and one dense layer to classify sequences
class Model(torch.nn.Module):
    def __init__(self, data, labels, lstm=1, lr=0.01):
        super().__init__()
        self.layers = []
        self.optimizer = None
        self.lstm = lstm
        self.lr = lr  # learning rate
        # in case of different size per sequence: data must be padded with 0s in end
        # data and labels have to be tuples of torch.tensor TODO assert
        self.train_data = data[0]  # shape train_set_size, time_steps, feat_size
        self.valid_data = data[1]  # shape valid_set_size, time_steps, feat_size
        self.test_data = data[2]  # shape test_set_size, time_steps, feat_size
        self.train_lab = labels[0]  # shape train_set_size, class_number
        self.valid_lab = labels[1]  # shape valid_set_size, class_number
        self.test_lab = labels[2]  # shape test_set_size, class_number
        self.feat_size = data[0].shape[2]
        self.output_size = labels[0].shape[1]
        # check sizes
        assert self.feat_size == data[1].shape[2] == data[2].shape[2], f'Inconsistent feature size: train {data[0].shape[3]}, valid {data[1].shape[2]}, test {data[2].shape[2]}'
        assert self.output_size == labels[1].shape[1] == labels[2].shape[1], f'Inconsistent label size: train {labels[0].shape[2]}, valid {labels[1].shape[1]}, test {labels[2].shape[1]}'
        assert data[1].shape[0] == labels[1].shape[0], f'Inconsistent validation set size: data {data[1].shape[0]}, labels {labels[1].shape[0]}'
        assert data[2].shape[0] == labels[2].shape[0], f'Inconsistent test set size: data {data[2].shape[0]}, labels {labels[2].shape[0]}'

        # model validation: sizes to try:
        self.size_hidden = [8, 16, 32]

        self.final_epochs = 0
        self.final_hidden = 0
        self.trained = False

    def build(self, hidden):
        # lstm layers
        for i in range(self.lstm):
            self.layers.append(LSTM(self.feat_size, hidden))
        # dense layer
        self.layers.append(Dense(hidden, self.output_size))
        # final layer with softmax activation function
        self.layers.append(Final(torch.nn.Softmax(dim=1)))
        # optimizer SGD using pytorch
        self.optimizer = torch.optim.SGD(self.get_parameters(), lr=self.lr)

    def train_net(self, epochs=100):
        mse_per_epoch = torch.zeros(epochs)
        loss_per_epoch = torch.zeros(epochs)
        for epoch in range(epochs):
            # TODO create batches
            # right now: full batch training
            batch = self.train_data.permute(1, 0, 2)  # new shape: time_steps, batch_size, feat_length
            labels = self.train_lab  # shape: batch_size, feat_length

            # forward pass
            prediction = self.forward(batch)
            loss_b = self.loss(prediction, labels)
            # backward pass using pytorch's autograd
            self.zero_grad()
            loss_b.backward()
            # optimize - do gradient descent
            self.optimizer.step()
            # self.optimize()
            loss_per_epoch[epoch] = loss_b
            mse_per_epoch[epoch] = self.mse(prediction, labels)
            print(f'Epoch {epoch}: loss {loss_b}')
        plot_training_loss(loss_per_epoch, ylabel='Cross-entropy loss')
        plot_training_loss(mse_per_epoch, ylabel='means squared error')
        pass

    def forward(self, batch):
        data = batch
        for lay in self.layers:
            data = lay.forward(data)
        return data

    def loss(self, prediction, labels):
        # cross entropy loss: L = log p_model(y | x)
        assert prediction.shape == labels.shape
        prob = prediction[labels.to(torch.bool)]
        return torch.mean(-1 * torch.log(prob))  # scalar value

    def mse(self, prediction, labels):
        assert prediction.shape == labels.shape
        pred_class = torch.argmax(prediction,dim=1)
        corr_class = torch.argmax(labels, dim=1)
        diff = (pred_class - corr_class).to(torch.bool)
        return torch.mean(diff.to(torch.float))

    def get_parameters(self):
        params = []
        for lay in self.layers:
            params += lay.parameters()
        return params  #torch.nn.ParameterList(params)

    def optimize(self):
        params = self.get_parameters()
        for p in params:
            update = p.grad.data * self.lr
            p.data.sub_(update)
            print(torch.max(update))

