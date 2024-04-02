import os
import random

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram

from constants import SAMPLE_RATE, NUM_SAMPLES, TRAIN_DIR


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform, target_sample_rate, num_samples, augment=True, augment_rate=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transform.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.augmentation = [T.FrequencyMasking(freq_mask_param=5),
                             T.TimeMasking(time_mask_param=5),
                             T.Vol(0.5)] if augment else None
        self.augment_rate = augment_rate
        self.samples = self._load_samples(root_dir)

    def _load_samples(self, root_dir):
        samples = []
        for label, subdir in enumerate(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.wav') or filename.endswith('.mp3'):
                        sample_path = os.path.join(subdir_path, filename)
                        waveform, sr = torchaudio.load(sample_path)
                        waveform = waveform.to(self.device)
                        waveform = self._resample_if_necessary(waveform, sr)
                        waveform = self._mix_down_if_necessary(waveform)
                        waveform = self._cut_if_necessary(waveform)
                        waveform = self._right_pad_if_necessary(waveform)
                        waveform = self.transform(waveform)
                        samples.append((waveform, label))
        return samples

    def __getitem__(self, index):
        waveform, label = self.samples[index]
        if self.augmentation:
            waveform = self._augment(waveform)
        return waveform, label

    def plot_spectrogram(self, index):
        mel_spec_db, _ = self[index]
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(waveform)
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db[0].cpu().detach().numpy(), aspect='auto', origin='lower', cmap='inferno')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Filterbank Index')
        plt.colorbar(label='dB')
        plt.show()

    def _augment(self, waveform):
        for augmentation in self.augmentation:
            if random.random() < self.augment_rate:
                waveform = augmentation(waveform)
        return waveform

    def __len__(self):
        return len(self.samples)

    def _resample_if_necessary(self, waveform, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            waveform = resampler(waveform)
        return waveform

    def _mix_down_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut_if_necessary(self, waveform):
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        return waveform

    def _right_pad_if_necessary(self, waveform):
        length = self.num_samples - waveform.shape[1]
        if length > 0:
            waveform = torch.nn.functional.pad(waveform, (0, length))
        return waveform


if __name__ == '__main__':
    mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    dataset = AudioDataset(TRAIN_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES)
    print(f'Length of dataset: {len(dataset)}')
    waveform, label = dataset[400]
    print(f'Waveform shape: {waveform.shape}')
    print(f'Label: {label}')
    dataset.plot_spectrogram(400)
