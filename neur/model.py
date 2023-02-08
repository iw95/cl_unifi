from neur.LSTM import LSTM
from neur.dense import Dense
from neur.final_layer import Final
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm


def plot_training_loss(losses, ylabel='', plt_title='', folder='.'):
    """Plots training loss per epoch"""
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


def plot_loss_hidden(losses, hidden):
    """Plots los for model validation."""
    plt.figure()
    plt.plot(hidden, losses.detach().numpy(), 'k')
    plt.xscale('log')
    plt.title('Loss model validation')
    plt.ylabel('Cross-entropy loss')
    plt.xlabel('Size hidden parameter')
    plt.xticks(hidden)
    plt.savefig(f'logs/validation/hidden_loss.png')
    plt.close()


def loss(prediction, labels):
    """Compute cross entropy loss L = log p_model(y | x)
    :param prediction: probability for each class assigned by model
    :param labels: true labels
    :return: cross entropy loss
    """
    assert prediction.shape == labels.shape
    prob = prediction[labels.to(torch.bool)]
    return torch.mean(-1 * torch.log(prob))  # scalar value


def mse(prediction, labels):
    """
    Compute mean squared error of hard classification
    :param prediction: probability for each class assigned by model
    :param labels: true labels
    :return: mean squared error
    """
    assert prediction.shape == labels.shape
    pred_class = torch.argmax(prediction,dim=1)
    corr_class = torch.argmax(labels, dim=1)
    diff = (pred_class - corr_class).to(torch.bool)
    return torch.mean(diff.to(torch.float))


class Model(torch.nn.Module):
    """
    Class creates a model with <lstm> LSTM layers and one dense layer to classify sequences
    """
    def __init__(self, data, labels, lstm=1, lr=0.005):
        """
        Initialising data sets and checking dimensions
        :param data: Tuple of training, validation and test data
        :param labels: Tuple of training, validation and test labels
        :param lstm: Number of LSTM layers to use
        :param lr: Learning rate
        """
        super().__init__()
        self.layers = []
        self.optimizer = None
        self.lstm = lstm
        self.lr = lr  # learning rate
        # in case of different size per sequence: data must be padded with 0s in end
        # Check type
        # data and labels have to be tuples of torch.tensor
        assert type(data) is tuple, 'data has to be of type tuple'
        assert type(labels) is tuple, 'labels has to be of type tuple'
        self.train_data = data[0]  # shape train_set_size, time_steps, feat_size
        self.valid_data = data[1]  # shape valid_set_size, time_steps, feat_size
        self.test_data = data[2]  # shape test_set_size, time_steps, feat_size
        self.train_lab = labels[0]  # shape train_set_size, class_number
        self.valid_lab = labels[1]  # shape valid_set_size, class_number
        self.test_lab = labels[2]  # shape test_set_size, class_number
        self.feat_size = data[0].shape[2]
        self.output_size = labels[0].shape[1]
        # check type
        for i in range(3):
            assert type(data[i]) is torch.Tensor, 'data[0] must be of type torch.Tensor'
            assert type(labels[i]) is torch.Tensor, 'labels[0] must be of type torch.Tensor'
        # check sizes
        assert self.feat_size == data[1].shape[2] == data[2].shape[2], f'Inconsistent feature size: train {data[0].shape[3]}, valid {data[1].shape[2]}, test {data[2].shape[2]}'
        assert self.output_size == labels[1].shape[1] == labels[2].shape[1], f'Inconsistent label size: train {labels[0].shape[2]}, valid {labels[1].shape[1]}, test {labels[2].shape[1]}'
        assert data[1].shape[0] == labels[1].shape[0], f'Inconsistent validation set size: data {data[1].shape[0]}, labels {labels[1].shape[0]}'
        assert data[2].shape[0] == labels[2].shape[0], f'Inconsistent test set size: data {data[2].shape[0]}, labels {labels[2].shape[0]}'

        # model validation: sizes to try:
        self.size_hidden = [4, 8, 16]

        # Parameters will be set after model validation
        self.final_epochs = 0
        self.final_hidden = 0
        self.hyp_set = False
        self.trained = False

    def build(self, hidden=None, use_saved=False):
        """
        Building model with 3 layers: LSTM, Dense and Final
        :param hidden: Size of hidden state in LSTM layer
        :param use_saved: Use parameters from previously trained model
        """
        if use_saved: # todo to be tested
            self.old_model()
        else:
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

    def train_net(self, epochs=None, plt_title='', validate=False, batches=50, shuffle=True):
        """Training model with <epoch> epochs and <batches> batches which are split randomly if <shuffle>==True.
        Model computes validation error on the go if <validate>==True.
        Uses either only test set or test and validation set for training depending on <validate>.
        Training should be done only after building model.
        :param epochs: Full iterations over the test set
        :param plt_title: To name plots created while validation
        :param validate: Whether to perform side tasks for model validation (eg error logging)
        :param batches: Number of batches per epoch
        :param shuffle: Whether to shuffle data for each epoch
        :return: training loss and optionally validation loss per epoch
        """
        # setting seed if shuffling
        if shuffle:
            np.random.seed(500)
        # using only test data or test and validation data respectively when performing model validation or assessment
        if validate:
            tr_data = self.train_data
            tr_lab = self.train_lab
        else:
            tr_data = torch.cat((self.train_data, self.valid_data), dim=0)
            tr_lab = torch.cat((self.train_lab, self.valid_lab), dim=0)
        # if hyper parameters are fixed, no epoch has to be given
        if epochs is None:
            epochs = self.final_epochs
        # Training in batches:
        batch_size = int(tr_data.shape[0] / batches)
        # initialise loss vectors
        mse_per_epoch = torch.zeros(epochs)
        loss_per_epoch = torch.zeros(epochs)
        validation_error = torch.zeros(epochs) if validate else None
        # iteration over epochs
        for epoch in tqdm(range(epochs)):
            # if shuffle option is True samples are shuffled each epoch - resulting in different batches
            if shuffle:
                idxs = np.arange(tr_data.shape[0])
                np.random.shuffle(idxs)
                idxs = torch.from_numpy(idxs)
                tr_data = tr_data[idxs]
                tr_lab = tr_lab[idxs]
            # Resetting batch loss each epoch
            batch_loss = 0
            batch_loss_mse = 0
            # iteration over batches
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
                # accumulating losses
                batch_loss += loss_b
                batch_loss_mse += mse(prediction, labels)
            # computing average loss per batch
            loss_per_epoch[epoch] = batch_loss/batches
            mse_per_epoch[epoch] = batch_loss_mse/batches
            # compute validation loss per epoch
            if validate:
                validation_error[epoch] = loss(self.forward(self.valid_data), self.valid_lab)

        # plot losses over epochs
        plot_training_loss(loss_per_epoch, ylabel='Cross-entropy loss', plt_title=plt_title, folder='training')
        plot_training_loss(mse_per_epoch, ylabel='means squared error', plt_title=plt_title, folder='training')
        return loss_per_epoch, (validation_error if validate else None)

    def model_validation(self, epochs=100):
        """
        Coordinate model validation over parameters epochs and size of hidden state in LSTM
        Logging and plotting validation error
        :param epochs: epochs
        """
        # opening and initialising files for logging
        min_loss = torch.zeros(len(self.size_hidden),2)
        with open('logs/training/loss.csv', 'w') as training_f,\
                open('logs/validation/loss.csv', 'w') as validation_f:
            tr_wr = csv.writer(training_f)
            va_wr = csv.writer(validation_f)
            tr_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])
            va_wr.writerow(['size_hidden'] + epochs * ['loss_per_epoch'])

            # starting model validation:
            # iterate over possible values for hyper parameters
            for i, sz in enumerate(self.size_hidden):
                # Building model with size of hidden vector=sz
                self.build(sz)
                # training model and computing training and validation loss across epochs
                loss_tr, loss_va = self.train_net(epochs=epochs, plt_title=str(sz), validate=True)
                # logging loss
                tr_wr.writerow([sz] + list(loss_tr))
                va_wr.writerow([sz] + list(loss_va))
                # plotting validation loss
                plot_training_loss(loss_va, 'Cross-entropy loss', plt_title=str(sz), folder='validation')
                # saving minimum loss
                min_loss[i,0] = torch.min(loss_va)
                min_loss[i,1] = torch.argmin(loss_va)
        self.print_validatation_results(min_loss)

    def print_validatation_results(self, min_loss):
        """
        Printing and saving results of model validation
        :param min_loss: Loss and epochs for each choice of parameter value for size hidden state
        """
        # print to file
        with open('logs/validation.txt', 'w') as valid_f:
            print(f'Minimum loss per parameter setting:', file=valid_f)
            for i, mins in enumerate(min_loss):
                print(f'Size {self.size_hidden[i]}: minimum loss {mins[0]} in epoch {mins[1]}', file=valid_f)
            print(f'Minimum {torch.min(min_loss[:,0])} for size {self.size_hidden[int(torch.argmin(min_loss[:,0]))]}', file=valid_f)
        # print to console
        print(f'Minimum loss per parameter setting:')
        for i, mins in enumerate(min_loss):
            print(f'Size {self.size_hidden[i]}: minimum loss {mins[0]} in epoch {mins[1]}')
        print(f'Minimum {torch.min(min_loss[:, 0])} for size {self.size_hidden[int(torch.argmin(min_loss[:, 0]))]}')
        # plot
        plot_loss_hidden(losses=min_loss[:,0], hidden=self.size_hidden)
        return

    def forward(self, batch):
        """
        Computing forward pass of batch data through network
        :param batch: batch data
        :return: prediction
        """
        # Switch dimensions of data to iterate over time_steps as first dimension
        data = batch.permute(1, 0, 2)  # new shape: time_steps, batch_size, feat_length
        # compute forward pass through layers
        for lay in self.layers:
            data = lay.forward(data)
        return data

    def get_parameters(self):
        """
        Return all model parameters for training to initialise torch optimiser
        :return: Model parameters
        """
        params = []
        for lay in self.layers:
            params += lay.parameters()
        return params

    def old_model(self):
        self.load_state_dict(torch.load('logs/model_params'))

    def save_model(self):
        torch.save(self.state_dict(), 'logs/model_params')

    def set_hyper_params(self, hidden, epochs):
        """
        Setting hyper parameters after model validation and building again with new setup.
        Indicating in model that validation is done.
        :param hidden: size of hidden state in LSTM
        :param epochs: number of epochs
        """
        self.final_hidden = hidden
        self.final_epochs = epochs
        self.build()
        self.hyp_set = True

    def assess(self):
        """
        Model assessment only after model is trained
        :return: test error
        """
        if not self.trained:
            raise Exception("Model is not trained yet. Run 'model.train()'first.")
        test_error = loss(self.forward(self.test_data), self.test_lab)
        return test_error

    def train_model(self):
        """
        Training final model after validation and hyper parameter setting
        Indicating in model that training is done.
        :return: training error
        """
        if not self.hyp_set:
            raise Exception("Model validation has not yet been performed. Run 'model.model_validation()' first.")
        tr_loss = self.train_net()
        self.trained = True
        return tr_loss[-1]
