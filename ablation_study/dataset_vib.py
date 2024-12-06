import json
import os

import librosa
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class CustomDataset(Dataset):
    def __init__(self, root_dir, split):
        self.file_paths = []
        self.labels = []
        self.label_to_id = {}  # 标签到数字的映射
        split_dir = os.path.join(root_dir, split)
        label_names = os.listdir(split_dir)

        for label_id, label_name in enumerate(label_names):
            label_dir = os.path.join(split_dir, label_name)
            # a_dir = os.path.join(label_dir, '声音特征')
            a_dir = os.path.join(label_dir)


            if not os.path.isdir(a_dir) or not os.path.isdir(a_dir):
                continue
            a_files = os.listdir(a_dir)
            for a_file in a_files:
                a_path = os.path.join(a_dir, a_file)
                if not os.path.isfile(a_path):
                    continue
                self.file_paths.append(( a_path))
                self.labels.append(label_id)
                self.label_to_id[label_name] = label_id
    def __len__(self):
        return len(self.file_paths)
    def calcuate_energy_spectrogram(self, data, n_fft, win_length, chorma=False, sr=None):
        S = np.abs(librosa.stft(data, n_fft=n_fft, win_length=win_length)) ** 2
        r = librosa.amplitude_to_db(S)#, np.max)
        if chorma and sr is not None:
            r = librosa.feature.chroma_stft(S=S, sr=sr)
        r = (r - np.min(r))/(np.max(r) - np.min(r))
        return r

    def __getitem__(self, idx):
        a_path = self.file_paths[idx]
        label = self.labels[idx]
        # a_data = sio.loadmat(a_path)['ngs']
        # a_data = sio.loadmat(a_path)['new_matrix'][:8,:7]
        matfile = sio.loadmat(a_path)['croppedSignal'].reshape(-1)
        label = torch.tensor(label)
        spec = self.calcuate_energy_spectrogram(data=matfile, n_fft=63, win_length=63, chorma=False, sr=25600)

        return spec, label



