from torch.utils.data import DataLoader

from dataset import AudioDataset


def create_loader(data_dir, mel_spec, sample_rate, num_samples, batch_size, augment=True):
    dataset = AudioDataset(data_dir, mel_spec, sample_rate, num_samples, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=augment)
