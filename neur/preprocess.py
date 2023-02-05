import librosa
import csv
import numpy as np

def preprocess():
    with open('speech-accent-archive/speakers_use.csv') as classes:
        data = list(csv.reader(classes))
        data = np.array(data)
        header = data[0]
        data = data[1:]
    # TODO
    # create mel spectrogram out of all audiofiles
    # save spectrograms


def split():
    # TODO
    # balance data?
    # split data
    # create batches TODO understand batches -> minibatch?
    # save splits
    pass
