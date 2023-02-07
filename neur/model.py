from LSTM import LSTM
from dense import Dense
from final_layer import Final
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm


def plot_training_loss(losses, ylabel='', plt_title='', folder='.'):
    plt.figure()
    plt.plot(np.arange(losses.shape[0]), losses.detach().numpy(), 'k')
    plt.title('Learning curve')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    # plt.show()
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
        self.hyp_set = False
        self.trained = False

    def build(self, hidden):
        if hidden is None:
            hidden = self.final_hidden

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

    def train_net(self, epochs, plt_title='', validate=False, batches=4, shuffle=False):
        if validate:
            tr_data = self.train_data
            tr_lab = self.train_lab
        else:
            tr_data = torch.stack(self.train_data, self.valid_data, dim=0)
            tr_lab = torch.stack(self.train_lab, self.valid_lab, dim=0)
        if epochs is None:
            epochs = self.final_epochs
        # todo random batches
        np.random.seed(500)
        mse_per_epoch = torch.zeros(epochs)
        loss_per_epoch = torch.zeros(epochs)
        validation_error = torch.zeros(epochs) if validate else None
        for epoch in tqdm(range(epochs)):
            # train in batches
            batch_size = int(tr_data.shape[0] / batches)
            batch_loss = 0
            batch_loss_mse = 0
            for b in range(batches):
                batch = tr_data[b*batch_size:(b+1)*batch_size]  # shape: batch_size, time_steps, feat_length
                labels = tr_lab[b*batch_size:(b+1)*batch_size]  # shape: batch_size, class_n

                # forward pass
                prediction = self.forward(batch)
                loss_b = loss(prediction, labels)
                # backward pass using pytorch's autograd
                self.zero_grad()
                loss_b.backward()
                # optimize - do gradient descent
                self.optimizer.step()
                batch_loss += loss_b
                batch_loss_mse += mse(prediction, labels)
            loss_per_epoch[epoch] = batch_loss/batches
            mse_per_epoch[epoch] = batch_loss_mse/batches

            if validate:
                validation_error[epoch] = loss(self.forward(self.valid_data), self.valid_lab)

        plot_training_loss(loss_per_epoch, ylabel='Cross-entropy loss', plt_title=plt_title, folder='training')
        plot_training_loss(mse_per_epoch, ylabel='means squared error', plt_title=plt_title, folder='training')
        return loss_per_epoch, (validation_error if validate else None)

    def mode_validation(self, size_hidden):
        # parameters to validate: size of hidden state in lstm and epochs
        epochs = 100
        # opening and initialising files for logging
        min_loss = torch.zeros(len(self.size_hidden),2)
        with open('logs/training/loss.csv', 'w') as training_f, open('logs/validation/loss.csv', 'w') as validation_f:
            tr_wr = csv.writer(training_f)
            va_wr = csv.writer(validation_f)
            tr_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])
            va_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])
            # initialising of logging done
            # starting model validation:
            # iterate over possibilities possible values for hyper parameters
            for i, sz in enumerate(self.size_hidden):
                print('validation')
                # Building model with size of hidden vector=sz
                self.build(sz)
                # training model and computing training and validation loss across epochs
                loss_tr, loss_va = self.train_net(epochs=epochs, plt_title=str(sz), validate=True)
                # logging loss
                tr_wr.writerow([sz] + list(loss_tr))
                va_wr.writerow([sz] + list(loss_va))
                # plotting loss
                plot_training_loss(loss_va, 'Cross-entropy loss', plt_title=str(sz), folder='validation')
                # saving minimum loss
                min_loss[i,0] = torch.min(loss_va)
                min_loss[i,1] = torch.argmin(loss_va)
        with open('logs/validation.txt', 'w') as valid_f:
            print(f'Minimum loss per parameter setting:', file=valid_f)
            for i, mins in enumerate(min_loss):
                print(f'Size {self.size_hidden[i]}: minimum loss {mins[0]} in epoch {mins[1]}', file=valid_f)
            print(f'Minimum {torch.min(min_loss[:,0])} for size {self.size_hidden[int(torch.argmin(min_loss[:,0]))]}', file=valid_f)
        # todo rerun training with best hidden size and write function to compute best ammount of epochs
        self.set_hyper_params(hidden=torch.min(min_loss[:,0]), epochs=min_loss[:,1]) # todo right now: overfitting

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

    def old_model(self):
        self.load_state_dict(torch.load('logs/model_params'))

    def save_model(self):
        torch.save(self.state_dict(), 'logs/model_params')

    def set_hyper_params(self, hidden, epochs):
        self.final_hidden = hidden
        self.final_epochs = epochs
        self.build()
        self.hyp_set = True

    def assess(self):
        if not self.trained:
            raise Exception("Model is not trained yet. Run 'model.train()'first.")
        test_error = loss(self.forward(self.test_data), self.test_lab)
        return test_error

    def train_model(self):
        if not self.hyp_set:
            raise Exception("Model validation has not yet been performed. Run 'model.model_validation()' first.")
        tr_loss = self.train_net()
        self.trained = True
        return tr_loss
