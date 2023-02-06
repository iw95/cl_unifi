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


def load_wave_data(audio_dir, file_name):
    file_path = audio_dir + '/' + file_name
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs


# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp


# listen to Example
def listen_example(file):
    return display.Audio(file)


# plot waveform
def wav_plot(file, ax):
    y, sr = librosa.load(file)
    return librosa.display.waveplot(y, ax=ax)


# plot mel-spectrogram
def mel_spectrogram_plot(file, ax):
    y, sr = librosa.load(file)
    S = librosa.feature.melspectrogram(y, sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), ax=ax)


def split():
    # TODO
    # balance data?
    # split data
    # create batches TODO understand batches -> minibatch?
    # save splits
    pass
