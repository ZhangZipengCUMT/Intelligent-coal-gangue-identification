import numpy as np
import scipy.signal
import scipy.io as scio
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt


path = r"gan1-01-001.mat"
data = scio.loadmat(path)["smoothed_data"].reshape(-1)

sr = 22500
n_fft = 63
win_length = 63

def calcuate_energy_spectrogram(data, n_fft, win_length, chorma=False, sr=None):
    S = np.abs(librosa.stft(data, n_fft=n_fft, win_length=win_length)) ** 2
    r = librosa.amplitude_to_db(S, np.max)
    if chorma and sr is not None:
        r = librosa.feature.chroma_stft(S=S, sr=sr)
    return r

def PSD(x, fs, nperseg=None):
    if nperseg is not None:
        (f, S) = scipy.signal.welch(x, fs, nperseg=nperseg)
    else:
        (f, S) = scipy.signal.periodogram(x, fs, scaling="density")
    return S

S = np.abs(librosa.stft(data, n_fft=n_fft, win_length=win_length))**2
chorma = librosa.feature.chroma_stft(S=S, sr=sr)




# sns.heatmap(librosa.amplitude_to_db(S, np.max))
# plt.show()

librosa.display.specshow(librosa.amplitude_to_db(S, np.max), x_axis="time")
# psd = PSD(x=data, fs=sr)

