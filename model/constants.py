import torch

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

SAMPLE_RATE = 44100
NUM_SAMPLES = SAMPLE_RATE * 1
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'
NUM_CLASSES = 2

SAVED_NAME = 'best.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'