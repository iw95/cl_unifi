import librosa
import librosa.display
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dagshub.streaming import DagsHubFilesystem


def prepro(max_len):
    """Preprocess audio data
    Save for each sample to use numpy array of mel-spectrogram.
    Save all labels in numpy array.
    :param max_len: number of time steps per file to save at most to save memory
    """
    print('preprocessing...')
    with open('../data/speakers_use.csv') as samples:
        data = list(csv.reader(samples))
        data = np.array(data)
        header = list(data[0])
        # Finding labels and saving
        l_idx = header.index('native_language')
        labels = data[1:,l_idx]
        array_from_labels(labels)
        # Finding file names
        f_idx = header.index('filename')
        filenames = list(data[1:, f_idx])
        # preprocessing and saving audio data
        #   create mel spectrogram out of all audiofiles
        #   save spectrograms
        save_spectrograms(filenames, max_len=max_len)
        return


def calculate_melsp(x, sr, n_fft=512):
    """
    Calculate mel spectrogram from wave data of audio file using librosa.
    librosa performs short time fourier transform to extract amplitudes per frequency per time.\
    Then frequency scale is changed to mel scale and amplitude scale to decibel.
    :param x: wave form sequence
    :param sr: sampling rate
    :param n_fft: window size for fast fourier transform. 512 as recommended value for speech
    :return: spectrogram as ndarray
    """
    # n_fft=512 suitable window size for speech data according to librosa
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, n_mels=128)
    S_log = librosa.power_to_db(S, ref=np.max)
    return S_log


def save_spectrograms(files, max_len):
    """
    Calculate mel-spectrogram using librosa for each sample and save as ndarray in 'data/specs.npy'
    :param files: Array of file names to save
    :param max_len: number of time steps per file to save at most to save memory
    """
    # Using dagshub file system if not all audio samples are physically present yet
    fs = DagsHubFilesystem()

    # analyse sequence length
    with open('../data/specs.npy', 'wb') as data_file:
        for filename in tqdm(files):
            f = 'recordings/' + filename + '.mp3'
            # If file does not physically exist: first open (and download) via dagshub client
            if not os.path.exists(f):
                with fs.open(f):
                    pass
            # Load audio using librosa
            y, sr = librosa.load(f)
            # Calculate spectrogram as better representation of audio
            seq = calculate_melsp(y, sr=sr)
            # cutting sequence at max_len if necessary
            seq = seq[:,:min(max_len, seq.shape[1])]
            # Save
            np.save(data_file, seq)
    return


def array_from_labels(labels):
    """
    Save labels as numpy array in file 'data/labels.npy'
    :param labels: ndarray of `str` labels
    """
    lang = np.unique(labels)
    np_labels = np.zeros((labels.shape[0], lang.shape[0]))
    # Create a 1-dim array for each class
    for i, l in enumerate(lang):
        np_labels[:,i] = (labels == l)
    with open('../data/labels.npy', 'wb') as data_file:
        np.save(data_file, np_labels)
    return


def plot_audio(file):
    """
    Visualise audio file as wave plot and mel spectrogram
    :param file: path to file
    """
    fig, axs = plt.subplots(2)
    y, sr = librosa.load(file)
    # wave form
    librosa.display.waveshow(y, ax=axs[0])
    # mel spectrogram
    S = calculate_melsp(y, sr=sr)
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel', ax=axs[1])
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
    plt.show()
    fig.savefig('logs/example_audio.png')
    return


def preprocess(max_len=1396):
    """
    Calling preprocessing of audio data operating in subfolder 'speech-accent-archive' to download files if necessary
    :param max_len: define max length of a sequence
    """
    os.chdir('speech-accent-archive/')
    # using 0.7 quantile of length
    prepro(max_len=max_len)
    os.chdir('../')
