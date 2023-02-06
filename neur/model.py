from LSTM import LSTM
from dense import Dense
from final_layer import Final
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
# use layers lstm, dense, softmax
# give prediction
# train


def plot_training_loss(losses, ylabel='', plt_title='', folder='.'):
    plt.figure()
    plt.plot(np.arange(losses.shape[0]), losses.detach().numpy(), 'k')
    plt.title('Learning curve')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.show()
    t = '_mse' if ylabel[0] in 'mM' else '_cel'
    if len(plt_title) > 0:
        plt_title = '_h' + plt_title
    plt.savefig(f'logs/{folder}/learn_curve{t}{plt_title}.png')
    plt.close()


# class creates a model with <lstm> LSTM layers and one dense layer to classify sequences
def loss(prediction, labels):
    # cross entropy loss: L = log p_model(y | x)
    assert prediction.shape == labels.shape
    prob = prediction[labels.to(torch.bool)]
    return torch.mean(-1 * torch.log(prob))  # scalar value


def mse(prediction, labels):
    assert prediction.shape == labels.shape
    pred_class = torch.argmax(prediction,dim=1)
    corr_class = torch.argmax(labels, dim=1)
    diff = (pred_class - corr_class).to(torch.bool)
    return torch.mean(diff.to(torch.float))


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
        self.size_hidden = [4, 8, 16, 32]

        self.final_epochs = 0
        self.final_hidden = 0
        self.trained = False

    def build(self, hidden):
        # lstm layers
        self.layers = []
        for i in range(self.lstm):
            self.layers.append(LSTM(self.feat_size, hidden))
        # dense layer
        self.layers.append(Dense(hidden, self.output_size))
        # final layer with softmax activation function
        self.layers.append(Final(torch.nn.Softmax(dim=1)))
        # optimizer SGD using pytorch
        self.optimizer = torch.optim.SGD(self.get_parameters(), lr=self.lr)

    def train_net(self, epochs=100, plt_title='', validate=False):
        mse_per_epoch = torch.zeros(epochs)
        loss_per_epoch = torch.zeros(epochs)
        validation_error = torch.zeros(epochs) if validate else None
        for epoch in range(epochs):
            # TODO create batches
            # right now: full batch training
            batch = self.train_data
            labels = self.train_lab  # shape: batch_size, feat_length

            # forward pass
            prediction = self.forward(batch)
            loss_b = loss(prediction, labels)
            # backward pass using pytorch's autograd
            self.zero_grad()
            loss_b.backward()
            # optimize - do gradient descent
            self.optimizer.step()
            loss_per_epoch[epoch] = loss_b
            mse_per_epoch[epoch] = mse(prediction, labels)

            if validate:
                validation_error[epoch] = loss(self.forward(self.valid_data), self.valid_lab)

        plot_training_loss(loss_per_epoch, ylabel='Cross-entropy loss', plt_title=plt_title, folder='training')
        plot_training_loss(mse_per_epoch, ylabel='means squared error', plt_title=plt_title, folder='training')
        return loss_per_epoch, validation_error

    def model_validation(self):
        # parameters to validate: size of hidden state in lstm
        # number of epochs
        epochs = 100
        min_loss = torch.zeros(len(self.size_hidden),2)
        with open('logs/training/loss.csv', 'w') as training_f, open('logs/validation/loss.csv', 'w') as validation_f:
            tr_wr = csv.writer(training_f)
            va_wr = csv.writer(validation_f)
            tr_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])
            va_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])
            for i, sz in enumerate(self.size_hidden):
                print('validation')
                self.build(sz)
                loss_tr, loss_va = self.train_net(epochs=epochs, plt_title=str(sz), validate=True)
                tr_wr.writerow([sz] + list(loss_tr))
                va_wr.writerow([sz] + list(loss_va))
                plot_training_loss(loss_va, 'Cross-entropy loss', plt_title=str(sz), folder='validation')
                min_loss[i,0] = torch.min(loss_va)
                min_loss[i,1] = torch.argmin(loss_va)
        print(f'Minimum loss per parameter setting:')
        for i, mins in enumerate(min_loss):
            print(f'Size {self.size_hidden[i]}: minimum loss {mins[0]} in epoch {mins[1]}')
        print(f'Minimum {torch.min(min_loss[:,0])} for size {self.size_hidden[torch.argmin(min_loss[:,0])]}')

    def forward(self, batch):
        data = batch.permute(1, 0, 2)  # new shape: time_steps, batch_size, feat_length
        for lay in self.layers:
            data = lay.forward(data)
        return data

    def get_parameters(self):
        params = []
        for lay in self.layers:
            params += lay.parameters()
        return params
