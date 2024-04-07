import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from util import create_loader
from model import DualPathNet
from constants import (DEVICE, EPOCHS, SAMPLE_RATE, TRAIN_DIR,
                       VAL_DIR, NUM_SAMPLES, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, TEST_DIR, WEIGHTS_PATH,
                       class_mapping, SAVE_PATH)
import torcheval.metrics.functional as M
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme()


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


def validate(model, val_loader, loss_fn, testing=False):
    model.eval()
    total_loss = 0.0
    true_labels, pred_labels, pred_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            classes = torch.argmax(predictions, dim=1)

            true_labels.append(targets.cpu())
            pred_labels.append(classes.cpu())
            pred_probs.append(predictions.cpu())
            total_loss += loss.item() * inputs.size(0)

    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    pred_probs = torch.cat(pred_probs)
    loss = total_loss / len(val_loader.dataset)

    metrics = {
        'Recall': M.multiclass_recall(pred_labels, true_labels).item(),
        'F1': M.multiclass_f1_score(pred_labels, true_labels).item(),
        'AP': M.multiclass_auprc(pred_probs, true_labels).item(),
        'AUC-ROC': M.multiclass_auroc(pred_probs, true_labels, num_classes=NUM_CLASSES).item(),
        'Val_loss': loss
    }

    if testing:
        metrics['Confusion'] = M.multiclass_confusion_matrix(pred_labels, true_labels, num_classes=NUM_CLASSES)
        metrics['PR'] = M.multiclass_precision_recall_curve(pred_probs, true_labels, num_classes=NUM_CLASSES)

    return metrics


def print_metrics(metrics):
    names = '\t'.join(key for key in metrics.keys() if key not in ['Confusion', 'PR'])
    values = '\t'.join(f'{value:.3f}' for key, value in metrics.items() if key not in ['Confusion', 'PR'])
    print(f'{names}\n{values}')


def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, save=True):
    best_metrics = {}
    best_f1 = 0
    metrics_history = {
        'Train_loss': [], 'Val_loss': [],
        'Recall': [], 'F1': [], 'AP': [], 'AUC-ROC': []
    }

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
            torch.save(model.state_dict(), WEIGHTS_PATH)
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
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    metrics = validate(model, test_loader, loss_fn, testing=True)
    print('Test metrics:')
    print_metrics(metrics)
    return metrics


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
    plt.savefig(os.path.join(SAVE_PATH, 'metrics.png'))


def plot_confusion_matrix(test_metrics):
    confusion = test_metrics['Confusion']
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion = confusion / row_sums
    fig, ax = plt.subplots()

    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.2f', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Normalized Confusion Matrix')
    ax.xaxis.set_ticklabels(class_mapping)
    ax.yaxis.set_ticklabels(class_mapping)
    plt.savefig(os.path.join(SAVE_PATH, 'confusion-matrix.png'))


def plot_pr(test_metrics):
    pr = test_metrics['PR']
    fig, ax = plt.subplots()

    for i in range(NUM_CLASSES):
        precision, recall = pr[i]
        ax.plot(recall, precision, label=class_mapping[i])

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    plt.savefig(os.path.join(SAVE_PATH, 'PR.png'))


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

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
    test_metrics = test(model, test_loader, loss_fn)
    plot_confusion_matrix(test_metrics)
    plot_pr(test_metrics)


if __name__ == '__main__':
    main()
