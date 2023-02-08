import numpy as np
import torch


def train_test_split(X, y, test_size=0.25, random_state=150):
    """Splits data into train and test sets while maintaining the class distribution.
    Uses function 'compute_indices_dist' to compute the indices for each set
    :param X: data tensor
    :param y: labels tensor
    :param test_size: between 0 and 1, size of test set in relation to complete set
    :param random_state: seed
    :return: Training and test data and labels
    """
    # calculate class distribution
    dist = torch.sum(y, dim=0)
    y_flat = torch.argmax(y, dim=1)
    train_idx, test_idx = compute_indices_dist(class_num=dist.shape[0], dist=dist, test_size=test_size,
                                               y_flat=y_flat, random_state=random_state)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]  # X_train, X_test, y_train, y_test


def compute_indices_dist(class_num, dist, test_size, y_flat, random_state):
    """
    Randomly generating indexes for training set and test set in complete data set using numpy.
    Maintaining same distribution of classes.
    :param class_num: number of classes
    :param dist: tensor with number of samples per class
    :param test_size: between 0 and 1, size of test set in relation to complete set
    :param y_flat: 1-dimensional vector indicating class of each sample with integers
    :param random_state: seed
    :return: Training set indices, test set indices
    """
    assert 0 < test_size < 1
    y_flat = y_flat.numpy()
    # Set seed
    np.random.seed(random_state)
    # initializing test set and training set
    test_set_size = int(torch.sum(torch.floor(dist * test_size)))
    test_idx = np.zeros(test_set_size, dtype=np.int64)
    train_idx = np.zeros(y_flat.shape[0] - test_set_size, dtype=np.int64)
    running_test_idx = 0
    running_train_idx = 0

    # Iterating through classes
    for c in range(class_num):
        if dist[c] == 0:
            continue
        # number of samples for test set and training set out of class c
        c_test_size = int(test_size*dist[c])
        c_train_size = int(dist[c]-c_test_size)
        # Indexes for class c
        samples_idx = (y_flat == c).nonzero()[0]  # indexes of class c (nested in one more dimension)
        # Shuffling
        np.random.shuffle(samples_idx)
        # Assigning test and training indices
        test_idx[running_test_idx:running_test_idx+c_test_size] = samples_idx[:c_test_size]
        train_idx[running_train_idx:running_train_idx+c_train_size] = samples_idx[c_test_size:]
        # Increase running indices
        running_test_idx += c_test_size
        running_train_idx += c_train_size
    # Final shuffling to not keep the classes in order
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    return torch.from_numpy(train_idx), torch.from_numpy(test_idx)
