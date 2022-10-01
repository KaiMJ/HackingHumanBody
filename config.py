# File Directories
PRETRAIN_DIR = 'data_mri_pretrain'
IMAGES_DIR = 'data/train_images'
DATA_CSV = 'data/train.csv'
MOD_IMAGES_DIR = 'data_modified/images'
MOD_MASKS_DIR = 'data_modified/masks'

MRI_TENSORBOARD_DIR = "saved/mri/tensorboard"
HPA_TENSORBOARD_DIR = "saved/hpa/tensorboard"
MRI_MODEL_PATH = 'saved/mri/models'
HPA_MODEL_PATH = 'saved/hpa/models'

# Dataset
IMG_SIZE = 256
BATCH_SIZE = 16
SPLIT = [0.8, 0.1, 0.1]

# Training
EPOCHS = 500
LEARNING_RATE = 0.01
OPTIMIZER = "adam"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

# Inference
PLOT = True
PLOT_EVERY = 10