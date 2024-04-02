import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from util import create_loader
from model import DualPathNet
from constants import (DEVICE, EPOCHS, SAMPLE_RATE, TRAIN_DIR,
                       VAL_DIR, NUM_SAMPLES, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, TEST_DIR, SAVED_NAME)
import torcheval.metrics.functional as M
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set()


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc='Training'):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, val_loader, loss_fn):
    model.eval()
    metrics = {'Accuracy': 0, 'Recall': 0, 'F1': 0, 'AUC-ROC': 0, 'Val_loss': 0}
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            classes = torch.argmax(predictions, dim=1)

            batch_metrics = {
                'Accuracy': M.multiclass_accuracy(classes, targets).to(DEVICE),
                'Recall': M.multiclass_recall(classes, targets).to(DEVICE),
                'F1': M.multiclass_f1_score(classes, targets).to(DEVICE),
                'AUC-ROC': M.multiclass_auroc(predictions, targets, num_classes=NUM_CLASSES).to(DEVICE)
            }

            size = inputs.size(0)
            for name, value in batch_metrics.items():
                metrics[name] += value.item() * size
            metrics['Val_loss'] += loss.item() * size
            num_samples += size

    avg_metrics = {name: metric / num_samples for name, metric in metrics.items()}
    return avg_metrics


def print_metrics(metrics):
    names = '\t'.join(metrics.keys())
    values = '\t'.join(f'{value:.3f}' for value in metrics.values())
    print(f'{names}\n{values}')


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, save=True):
    best_metrics = {}
    best_f1 = 0
    metrics_history = {'Train_loss': [], 'Val_loss': [], 'Accuracy': [], 'Recall': [], 'F1': [], 'AUC-ROC': []}
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        metrics = validate(model, val_loader, loss_fn)
        metrics['Train_loss'] = train_loss
        f1, val_loss = metrics['F1'], metrics['Val_loss']
        if save and f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
            print('Saving the best weights...')
            torch.save(model.state_dict(), SAVED_NAME)
        scheduler.step(val_loss)
        print_metrics(metrics)
        print('-' * 50)
        for name, value in metrics.items():
            metrics_history[name].append(value)
    print('Training is done\n')
    print('Validation metrics:')
    print_metrics(best_metrics)
    return metrics_history


def test(model, test_loader, loss_fn):
    model.load_state_dict(torch.load(SAVED_NAME))
    metrics = validate(model, test_loader, loss_fn)
    print('Test metrics:')
    print_metrics(metrics)


def plot_metrics(metrics_history):
    num_metrics = len(metrics_history.keys())
    num_rows = np.ceil(num_metrics / 3).astype(int)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    for ax, (metric, values) in zip(axs.flatten(), metrics_history.items()):
        sns.lineplot(data=values, ax=ax)
        sns.scatterplot(data=values, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)

    if num_metrics % 3 != 0:
        for ax in axs.flatten()[num_metrics:]:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig('metrics.png')
    print('The plot is saved as "metrics.png"')


def main():
    mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64).to(DEVICE)
    train_loader = create_loader(TRAIN_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, BATCH_SIZE)
    val_loader = create_loader(VAL_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, BATCH_SIZE, augment=False)
    test_loader = create_loader(TEST_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, BATCH_SIZE, augment=False)

    model = DualPathNet(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=10)

    metrics_history = train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, EPOCHS)
    plot_metrics(metrics_history)
    test(model, test_loader, loss_fn)


if __name__ == '__main__':
    main()
