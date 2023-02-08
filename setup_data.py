from filter import filter_data
from preprocess import preprocess
import os
from dagshub.streaming import install_hooks


def main():
    # install dagshub hooks to be able to access data from repo
    os.chdir('speech-accent-archive/')
    install_hooks()
    os.chdir('../')

    # Filter data to use for training based on class sizes
    # creates speech-accent-archive/speakers_use.csv and speech-accent-archive/overview.csv
    filter_data()
    # preprocess chosen audio files from wave form to spectrogram, save as .npy
    preprocess()


if __name__ == '__main__':
    main()
