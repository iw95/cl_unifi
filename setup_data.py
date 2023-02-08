from filter import filter_data
from preprocess import preprocess
import os
from dagshub.streaming import install_hooks
import numpy as np
import matplotlib.pyplot as plt


def set_up_data(setup_dagshub=False):
    """
    Filtering and preprocessing data.
    Set setup_dagshub=True in first run as dagshub hooks have to be installed.
    :param setup_dagshub: Set True in first run
    """
    if setup_dagshub:
        # install dagshub hooks to be able to access data from repo
        os.chdir('speech-accent-archive/')
        install_hooks()
        os.chdir('../')

    # Filter data to use for training based on class sizes
    # creates speech-accent-archive/speakers_use.csv and speech-accent-archive/overview.csv
    filter_data()
    # preprocess chosen audio files from wave form to spectrogram, save as .npy
    preprocess(max_len=1396)


# Details of data:
# saved audio data as spectrograms
# Mean seqenence length is 1272.2083333333333
# Mimimum seqenence length is 722
# Maximum seqenence length is 3666
# 0.7-quantile is 1396.3 -> use max_length = 1396

if __name__ == '__main__':
    set_up_data(setup_dagshub=False)
