import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio.transforms import MelSpectrogram

from constants import NUM_CLASSES, DEVICE, SAMPLE_RATE, TRAIN_DIR, VAL_DIR, NUM_SAMPLES, BATCH_SIZE
from model import DualPathNet
from util import create_loader
from train import train, validate


def tune(model, loss_fn, train_loader, val_loader, epochs=20, iterations=10):
    learning_rates = torch.logspace(-4, -2, iterations)
    best_hyperparameters = {}
    best_f1 = 0

    for i, lr in enumerate(learning_rates):
        print(f'\nIteration {i + 1}/{iterations}: Tuning with lr={lr}, epochs={epochs}\n')

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=epochs // 5)

        train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, save=False)
        metrics = validate(model, val_loader, loss_fn)
        f1 = metrics['F1']
        if f1 > best_f1:
            best_f1 = f1
            best_hyperparameters = {'lr': lr}
            best_hyperparameters.update(metrics)

    return best_hyperparameters


def main():
    mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64).to(DEVICE)
    model = DualPathNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = create_loader(TRAIN_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, BATCH_SIZE)
    val_loader = create_loader(VAL_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, BATCH_SIZE, augment=False)

    best_hyperparameters = tune(model, loss_fn, train_loader, val_loader, epochs=20, iterations=10)
    print(f'Best hyperparameters: {best_hyperparameters}')


if __name__ == '__main__':
    main()
