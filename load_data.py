import numpy as np
import torch
from tqdm import tqdm


def load_data(samples=600, time=3666, feat_s=128, class_num=9, X_path='data/specs.npy', y_path='data/labels.npy'):
    """Load data from numpy files

    Requires dimensions to initialize tensors.
    Padding samples with trailing zeros if shorter than time*.
    :param samples: number of samples
    :param time: time steps
    :param feat_s: feature size
    :param class_num: number of classes
    :param X_path: path to .npy file containing data points
    :param y_path: path to .npy file containing labels
    :return: Data and labels as tensors
    """

    print(f'{samples} samples, {time} time steps, {feat_s} features, {class_num} classes')
    X = torch.zeros(samples, time, feat_s)
    y = torch.zeros(samples,class_num)
    with open(y_path, 'rb') as y_data:
        y = torch.from_numpy(np.load(y_data))
    y = y[:samples]
    with open(X_path, 'rb') as X_data:
        for i in tqdm(range(samples)):
            xi = np.load(X_data)
            X[i,:xi.shape[1]] = torch.from_numpy(np.transpose(xi))
    return X, y
