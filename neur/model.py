from LSTM import LSTM
from dense import Dense
from final_layer import Final
import torch
# use layers lstm, dense, softmax
# give prediction
# train


# class creates a model with <lstm> LSTM layers and one dense layer to classify sequences
class Model:
    def __init__(self, data, labels, lstm=1,):
        self.layers = []
        self.lstm = lstm
        # in case of different size per sequence: data must be padded with 0s in end
        self.train_data = torch.tensor(data[0]) # shape batch_num, batch_size, time_steps, feat_size
        self.valid_data = torch.tensor(data[1]) # shape valid_set_size, time_steps, feat_size
        self.test_data = torch.tensor(data[2]) # shape test_set_size, time_steps, feat_size
        self.train_lab = torch.tensor(labels[0]) # shape batch_num, batch_size, class_number
        self.valid_lab = torch.tensor(labels[1]) # shape valid_set_size, class_number
        self.test_lab = torch.tensor(labels[2]) # shape test_set_size, class_number
        self.feat_size = data[0].shape[3]
        self.output_size = labels[0].shape[2]
        # check sizes
        assert self.feat_size == data[1].shape[2] == data[2].shape[2], f'Inconsistent feature size: train {data[0].shape[3]}, valid {data[1].shape[2]}, test {data[2].shape[2]}'
        assert self.output_size == labels[1].shape[1] == labels[2].shape[1], f'Inconsistent label size: train {labels[0].shape[2]}, valid {labels[1].shape[1]}, test {labels[2].shape[1]}'
        # TODO create batches in each epoch?
        assert data[1].shape[0] == labels[1].shape[0], f'Inconsistent validation set size: data {data[1].shape[0]}, labels {labels[1].shape[0]}'
        assert data[2].shape[0] == labels[2].shape[0], f'Inconsistent test set size: data {data[2].shape[0]}, labels {labels[2].shape[0]}'

        # model validation: sizes to try:
        self.size_hidden = [8,16,32,64]
        self.epochs = torch.arange(5,15,2)

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
        self.layers.append(Final(torch.nn.Softmax))

    def train(self, epochs):
        for e in range(epochs):
            # TODO create batches
            batch = self.train_data
            labels = self.train_lab
            # TODO manage input dimensions

            pred = self.forward(batch)
            loss_b = self.loss(pred, labels)
            # TODO start backward pass
            loss_b.backward()
            # TODO do gradient descent
        pass

    def forward(self, batch):
        proc = batch
        for lay in self.layers:
            proc = lay.forward(proc) # TODO possible to do full batch at once or iterative?
        return proc

    def loss(self, prediction, labels):
        # cross entropy loss: L = log p_model(y | x)
        idx = torch.outer(torch.arange(0,self.output_size), torch.ones(labels.shape[0])) == labels
        prob = torch.max(prediction[idx], dim=1)
        return torch.log(prob)



