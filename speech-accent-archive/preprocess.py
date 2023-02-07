import librosa
import librosa.display
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dagshub.streaming import DagsHubFilesystem


def preprocess():
    print('preprocessing...')
    with open('speakers_use.csv') as classes:
        data = list(csv.reader(classes))
        data = np.array(data)
        header = list(data[0])
        # finding labels
        l_idx = header.index('native_language')
        labels = data[1:,l_idx]
        array_from_labels(labels)
        print('found and saved labels')
        # finding file names
        f_idx = header.index('filename')
        filenames = list(data[1:, f_idx])
        print('collected file names')
        # preprocessing and saving audio data
        #   create mel spectrogram out of all audiofiles
        #   save spectrograms
        save_spectrograms(filenames)
        print('saved audio data as spectrograms')
        return


# change wave data to mel-stft
def calculate_melsp(x, sr, n_fft=512):
    # n_fft=512 suitable window size for speech data according to librosa
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, n_mels=128)
    S_log = librosa.power_to_db(S, ref=np.max)
    return S_log


# visualise audio file
def plot_audio(file):
    fig, axs = plt.subplots(2)
    y, sr = librosa.load(file)
    # wave form
    librosa.display.waveshow(y, ax=axs[0])
    # mel spectrogram
    S = calculate_melsp(y, sr=sr)
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', ax=axs[1])
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
    return


def save_spectrograms(files):
    fs = DagsHubFilesystem()
    with open('../data/specs.npy', 'wb') as data_file:
        for filename in tqdm(files):
            f = 'recordings/' + filename + '.mp3'
            if not os.path.exists(f):
                fs.open(f)
            y, sr = librosa.load(f)
            seq = calculate_melsp(y, sr=sr)
            np.save(data_file, seq)
            # open again with
            # with open('test.npy', 'rb') as f:
            #     a = np.load(f)
            #     t = torch.from_numpy(a)
    return


def array_from_labels(labels):
    lang = np.unique(labels)
    np_labels = np.zeros((labels.shape[0], lang.shape[0]))
    for i, l in enumerate(lang):
        np_labels[:,i] = (labels == l)
    with open('../data/labels.npy', 'wb') as data_file:
        np.save(data_file, np_labels)
    return


def main():
    preprocess()


if __name__ == '__main__':
    main()
