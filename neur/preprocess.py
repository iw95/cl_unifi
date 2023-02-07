import librosa
import librosa.display
import csv
import numpy as np
import matplotlib.pyplot as plt


def preprocess():
    with open('speech-accent-archive/speakers_use.csv') as classes:
        data = list(csv.reader(classes))
        data = np.array(data)
        header = data[0]
        data = data[1:]
    # TODO
    # create mel spectrogram out of all audiofiles
    # save spectrograms


# change wave data to mel-stft
def calculate_melsp(x, n_fft=512, hop_length=64):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=32)
    return melsp
    # M = librosa.feature.melspectrogram(y=y, sr=sr)
    # M_db = librosa.power_to_db(M, ref=np.max)


# plot waveform
def wav_plot(file, ax):
    y, sr = librosa.load(file)
    return librosa.display.waveshow(y, ax=ax)


# plot mel-spectrogram
def mel_spectrogram_plot(file, ax):
    y, sr = librosa.load(file)
    S = librosa.feature.melspectrogram(y, sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax)


def split():
    # TODO
    # balance data?
    # split data
    # create batches TODO understand batches -> minibatch?
    # save splits
    pass


def main():
    fig, axs = plt.subplots(2)
    wav_plot('../speech-accent-archive/recordings/amharic1.mp3', ax=axs[0])
    mel_spectrogram_plot('../speech-accent-archive/recordings/amharic1.mp3', ax=axs[1])


if __name__ == '__main__':
    main()
