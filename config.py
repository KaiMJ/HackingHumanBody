# Default Experiment Values
from torch.cuda import is_available

EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
SPLIT = 0.8
IMG_SIZE = 256 
PLOT = True
PLOT_EVERY = 5


CSV_FILE = 'data/train.csv'
ROOT_DIR = 'data/train_images'

DEVICE = "cuda" if is_available() else "cpu"

LOG_PATH = 'saved/logs.txt'
TENSORBOARD_PATH_DIR = "saved/tensorboard"
MODEL_PATH = 'saved/models'
