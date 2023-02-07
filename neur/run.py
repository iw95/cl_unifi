import torch
from model import Model
from load_data import load_data
from preprocess import train_test_split


def main():
    # running now on real data!
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)
    model = Model(data=(X_train, X_validate, X_test), labels=(y_train, y_validate, y_test))
    model.model_validation()


def gen_data(samples=50, classes=4):
    torch.manual_seed(0)
    x_underly = torch.arange(0, 8 * torch.pi, 0.25 * torch.pi)
    period = torch.rand(samples) * 2 * torch.pi
    label = torch.zeros(samples, classes)
    for c in range(classes):
        label[:, c] = (period >= (torch.pi * c * 2 / classes)) & (period < (torch.pi * (c + 1) * 2 / classes))

    x_ = torch.sin(torch.outer(period, x_underly))
    x = torch.zeros(list(x_.shape) + [2])
    x[:, :, 0] = x_
    x[:, :, 1] = x_ * 2

    return x, label


def test_training():
    x, label = gen_data(samples=50, classes=4)

    print(x.shape)
    print(label.shape)

    print('Initialising...')
    model = Model(data=(x[:40], x[40:45], x[45:50]), labels=(label[:40], label[40:45], label[45:50]))
    print('Initialisation successful!')

    print('Building model...')
    model.build(8)
    print('Build successful!')

    print('Training model...')
    model.train_net(epochs=100)
    print('Training successful! Wow!')


def validate_dummy():
    x, label = gen_data(samples=50, classes=4)

    model = Model(data=(x[:40], x[40:45], x[45:50]), labels=(label[:40], label[40:45], label[45:50]))
    model.model_validation()
    model.save_model()


if __name__ == '__main__':
    main()
