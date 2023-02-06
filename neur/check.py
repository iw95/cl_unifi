import torch
from model import Model


def main():
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


if __name__ == '__main__':
    main()
