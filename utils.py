# Loss functions
import torch
import torch.nn.functional as F
# Process Dataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Data Augmentation and Loader
import albumentations as A
from albumentations import Resize, Normalize, RandomRotate90, HorizontalFlip, VerticalFlip
from dataset import HPADataset, HPATestset, MRIDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Configure training and inference
import argparse
from config import *

def dice_loss(output, target):
    num = torch.sum(output * target)
    den = torch.sum(output) + torch.sum(target)
    dice = (2 * num) / (den + 1)
    return 1 - dice

def focal_loss(output, target, gamma=2):
    bce_loss = F.binary_cross_entropy(output, target, reduction='none')
    pt = torch.exp(-bce_loss) # high loss = low prob, Low loss = high prob
    loss = (1. - pt)**gamma * bce_loss # penalize false predictions more.
    return loss.mean()

# Loss
criterion_name = "FOCAL+DICE"
def criterion(output, target):
    return focal_loss(output, target) + dice_loss(output, target)

score_fn_name = "DICE"
def score_fn(output, target):
    return 1 - dice_loss(output, target)

def bce(output, target):
    return F.binary_cross_entropy(output, target.to(torch.float32))

def f1_score(output, target):
    tp = (output * target).sum(dim=(1, 2))
    fp = (output * (1 - target)).sum(dim=(1, 2))
    fn = ((1 - output) * (target)).sum(dim=(1, 2))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.mean()

# Convert Input and Outputs
def threshold_tensor(array, t=0.5):
    out = (array - torch.min(array)) / (torch.max(array) - torch.min(array))
    out = array.clone()
    idx = out > t
    out[idx] = 1
    out[~idx] = 0
    return out

# Return image with mask highlighted
def get_mask_highlight(img, mask):
    '''
    Get Mask Highlights for better visualization.
        img - numpy array
        mask - numpy array
    '''
    img_mask_highlight = img.copy()
    # Draw Mask Highlight
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_mask_highlight, [c], -1, (255, 0, 0), thickness=7)
    return img_mask_highlight

def plot_images(writer, inputs, outputs, losses, score_name, n_iter, e, name, n=9):
    '''
    Plot Input and Output numpy arrays on Tensorboard.
        writer - SummaryWriter
        inputs - Batch of Numpy [Batch, H, W, C]
        outputs - Batch of Mask Numpy [Batch, H, W]
    '''
    plt.tight_layout()
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flatten()):
        highlight = get_mask_highlight(inputs[i], outputs[i])
        ax.imshow((highlight * 255).astype(np.uint8))
        if losses is not None:
            ax.set_title(f"{score_name} : {losses[i]:.4f}")
    plt.suptitle(f"Epoch {e}")
    writer.add_figure(name, fig, n_iter)


def mri_plot(image, output, mask, mean, std, score_fn, score_fn_name, writer, n_iter, e, args, name, n=9):
    '''
    Plot MRI Scans.
    '''
    losses = [score_fn(output[i], mask[i]) for i in range(n)]
    plt_imgs = [(image[i].permute(1, 2, 0).cpu() * std[i] + mean[i]).numpy() / 255.0 for i in range(n)]
    plt_outs = threshold_tensor(output[:n]).detach().cpu().numpy().astype(np.uint8) # [9, 256, 256]
    plt_masks = mask[:n].cpu().numpy().astype(np.uint8) # [9, 256, 256]
    plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, n_iter, e, name=name + " Original")
    plot_images(writer, plt_imgs, plt_outs, losses, score_fn_name, n_iter, e, name=name + " Predicted")

def mri_plot_test(image, output, mask, mean, std, score_fn, score_fn_name, writer, n=9):
    losses = [score_fn(output[i], mask[i]) for i in range(n)]
    plt_imgs = [(image[i].permute(1, 2, 0).cpu() * std[i] + mean[i]).numpy() / 255.0 for i in range(n)]
    plt_outs = threshold_tensor(output[:n]).detach().cpu().numpy().astype(np.uint8) # [9, 256, 256]
    plt_masks = mask[:n].cpu().numpy().astype(np.uint8) # [9, 256, 256]

    plot_images(writer, plt_imgs, plt_masks, None, score_fn_name, 1, "Final", "Test Original", n)
    plot_images(writer, plt_imgs, plt_outs, losses, score_fn_name, 1, "Final", "Test Predicted", n)

def format_tensors_plt(inp, out, transform, n=9, mask=None):
    '''
        Format tensors and then output plot ready arrays.
    '''
    plt_imgs = transform(image=inp[:n].permute(0, 2, 3, 1).cpu().numpy())['image'] # [9, 256, 256, 3]
    plt_outs = threshold_tensor(out[:n]).detach().cpu().numpy().astype(np.uint8) # [9, 256, 256]
    if mask is not None:
        plt_masks = mask[:n].cpu().numpy().astype(np.uint8) # [9, 256, 256]
        return plt_imgs, plt_outs, plt_masks
    else:
        return plt_imgs, plt_outs

def combine_tensors(arrays: torch.Tensor, shape) -> np.ndarray:
    '''
    Combines tiled tensors and returns one big image.

    Parameters:
        arrays:
            images: [9, 4, 3, 256, 256] --> [9, 256, 256, 3]
            output: Tensor [9, 4, 256, 256] --> [9, 256, 256]
            masks: [9, 4, 256, 256] --> [9, 256, 256]
    '''
    h = shape[0]
    w = shape[1]
    if len(arrays.shape) == 4:
        result = np.zeros((arrays.shape[0], h*2, w*2))
        result[:, :h, :w] = arrays[:, 0, ...]
        result[:, :h, w:] = arrays[:, 1, ...]
        result[:, h:, :w] = arrays[:, 2, ...]
        result[:, h:, w:] = arrays[:, 3, ...]
    else:
        result = np.zeros((arrays.shape[0], 3, h*2, w*2))
        result[:, :, :h, :w] = arrays[:, 0, ...]
        result[:, :, :h, w:] = arrays[:, 1, ...]
        result[:, :, h:, :w] = arrays[:, 2, ...]
        result[:, :, h:, w:] = arrays[:, 3, ...]
        result = result.transpose(0, 2, 3, 1)
    return result


def hpa_transforms(mean, std, size):
    '''
    Get transformation functions.
    
    Parameters:
        mean: mean of the dataset
        std: standard deviation of the dataset
        size: size of resized image.
    Returns:
        transform: transform function for training
        test_transform: resize and normalize for testing
        normalize: normalize function for input images
        inv_normalize: inverse normalize function for normalized images
    '''
    # mean, std = np.zeros(3), np.ones(3)
    transform = A.Compose([
        Resize(size, size),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90()
    ])

    normalize = Normalize(mean=mean, std=std, max_pixel_value=1)

    test_transform = A.Compose([
        Resize(size, size)
    ])

    inv_normalize = A.Compose([
        Normalize(mean = (0., 0., 0.), std=1/std, max_pixel_value=1),
        Normalize(mean = -mean, std = (1.0, 1.0, 1.0), max_pixel_value=1),
    ])

    return transform, test_transform, normalize, inv_normalize

def hpa_get_mean_std(full_dataset=None, size=None, calculate=False):
    '''
    Calculate the mean and standard deviation across the dataset.

    Parameters:
        full_dataset: PyTorch Dataset class
        size: image size
        calculate: if False, return pre-calculated values with train indexes
    Returns:
        mean: array of size 3 of the mean
        std: array of size 3 of the standard deviation
    '''
    if not calculate:
        mean = torch.Tensor([0.8316, 0.7917, 0.8253])
        std = torch.Tensor([0.1756, 0.1866, 0.1794])
        return mean, std

    print("Calulating mean, std...")
    # Find mean
    total = torch.zeros(3)
    for image, _ in full_dataset:
        mean = torch.mean(image.view(3, -1), dim=1)
        total += mean
    mean = total / len(full_dataset)

    # Find STD
    total_var = torch.zeros(3)
    for image, _ in full_dataset:
        total_var += ((image.view(3, -1) - mean.unsqueeze(1))**2).sum(1)
    std = torch.sqrt(total_var / (len(full_dataset)*size*size))

    return mean, std

def hpa_prepare(args):
    '''
    Prepares HPA data loaders and tensorboard writers.
        
    Returns:
        train, val, test data loaders
        train, val, test tensorboard writers
    '''
    # full_dataset = HPADataset(args.MOD_IMAGES_DIR, args.MOD_MASKS_DIR, )
    mean, std = hpa_get_mean_std(calculate=False)
    transform, test_transform, normalize, inv_normalize = hpa_transforms(mean, std, args.IMG_SIZE)

    n = len(os.listdir(args.MOD_IMAGES_DIR))
    idxs = np.random.permutation(n)
    if args.image_tiling:
        train_size = int(n * (args.SPLIT[0]+args.SPLIT[1]))
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size: ]
    else:
        train_size = int(n * args.SPLIT[0])
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size: train_size+int(n*args.SPLIT[1])]
        test_idxs = idxs[train_size+int(n*args.SPLIT[1]):]

    train_dataset = HPADataset(args.MOD_IMAGES_DIR, args.MOD_MASKS_DIR, transform, normalize, train_idxs)
    val_dataset = HPADataset(args.MOD_IMAGES_DIR, args.MOD_MASKS_DIR, test_transform, normalize, val_idxs)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, 
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE,
                            shuffle=True, drop_last=True)

    if args.image_tiling:
        test_dataset = HPATestset(args.TEST_IMAGES_DIR, args.TEST_MASKS_DIR, test_transform, normalize)
        test_loader = DataLoader(test_dataset, batch_size=9, shuffle=True)
    else:
        test_dataset = HPADataset(args.MOD_IMAGES_DIR, args.MOD_MASKS_DIR, test_transform, normalize, test_idxs)
        test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True)

    train_writer = SummaryWriter(args.HPA_TENSORBOARD_DIR + "/train")
    val_writer = SummaryWriter(args.HPA_TENSORBOARD_DIR + "/val")
    test_writer = SummaryWriter(args.HPA_TENSORBOARD_DIR + "/test")

    return train_loader, train_writer, val_loader, val_writer, test_loader, test_writer, inv_normalize

def mri_transform(size):
    '''
    Get transformation functions for MRI dataset.

    Parameter:
        size: size of the image
    '''
    transform = A.Compose([
        Resize(size, size),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90()
    ])
    
    test_transform = A.Compose([
        Resize(size, size)
    ])
    
    return transform, test_transform

def mri_prepare(args):
    '''
    Prepares MRI data loaders and tensorboard writers.
        
    Returns:
        train, val, test data loaders
        train, val, test tensorboard writers
    '''
    transform, test_transform = mri_transform(args.IMG_SIZE)
    
    n = len(os.listdir(args.PRETRAIN_DIR)) - 2 # Train.csv and README.md
    print(f"Scanning a total of {n} patients...")

    idxs = np.random.permutation(n)
    train_size = int(n * args.SPLIT[0])
    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size: train_size+int(n*args.SPLIT[1])]
    test_idxs = idxs[train_size+int(n*args.SPLIT[1]):]

    train_dataset = MRIDataset(args.PRETRAIN_DIR, transform, train_idxs)
    val_dataset = MRIDataset(args.PRETRAIN_DIR, test_transform, val_idxs)
    test_dataset = MRIDataset(args.PRETRAIN_DIR, test_transform, test_idxs)

    print(f"Image counts -- Train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, 
                            shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE,
                            shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True)

    train_writer = SummaryWriter(args.MRI_TENSORBOARD_DIR + "/train")
    val_writer = SummaryWriter(args.MRI_TENSORBOARD_DIR + "/val")
    test_writer = SummaryWriter(args.MRI_TENSORBOARD_DIR + "/test")

    return train_loader, train_writer, val_loader, val_writer, test_loader, test_writer

def get_args():
    '''
    Argument parser for training.
    '''
    parser = argparse.ArgumentParser(description="Training U-Net model for segmentation of brain MRI")

    parser.add_argument(
        "--description",
        '-d',
        default = DESCRIPTION,
        type=str,
        help="Description of training."
    )
    # Train
    parser.add_argument(
        "--mri-pretrain",
        default = True,
        type=bool,
        help="With MRI pretraining training."
    )
    parser.add_argument(
        "--image-tiling",
        default=True,
        type=bool,
        help="With image tiling for training."
    )
    # Dataset
    parser.add_argument(
        "--IMG-SIZE",
        type=int,
        default=IMG_SIZE,
        help="Image size during training and inference."
    )
    parser.add_argument(
        "--BATCH-SIZE",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--SPLIT",
        type=list,
        default=SPLIT,
        help="Split of train, validation datasets."
    )

    # Training
    parser.add_argument(
        "--EPOCHS",
        type=int,
        default=EPOCHS,
        help="Epochs of training."
    )
    parser.add_argument(
        "--MRI-LEARNING-RATE",
        type=float,
        default=MRI_LEARNING_RATE,
        help="Learning rate for MRI"
    )
    parser.add_argument(
        "--HPA-LEARNING-RATE",
        type=float,
        default=HPA_LEARNING_RATE,
        help="Learning rate for MRI"
    )
    parser.add_argument(
        "--OPTIMIZER",
        type=str,
        default=OPTIMIZER,
        help="Default optimizer: Adam"
    )
    # File directories
    parser.add_argument(
        "--PRETRAIN-DIR",
        type=str,
        default=PRETRAIN_DIR,
        help="Path to MRI pretraining dataset directory"
    )
    parser.add_argument(
        "--IMAGES-DIR",
        type=str,
        default=IMAGES_DIR,
        help="Path to HPA images training images directory"
    )
    parser.add_argument(
        "--DATA-CSV",
        type=str,
        default=DATA_CSV,
        help="Path to HPA training csv file"
    )
    parser.add_argument(
        "--MOD-IMAGES-DIR",
        type=str,
        default=MOD_IMAGES_DIR,
        help="Path to HPA training images directory"
    )
    parser.add_argument(
        "--MOD-MASKS-DIR",
        type=str,
        default=MOD_MASKS_DIR,
        help="Path to HPA training masks directory"
    )
    parser.add_argument(
        "--TEST-IMAGES-DIR",
        type=str,
        default=TEST_IMAGES_DIR,
        help="Path to HPA testing images directory"
    )
    parser.add_argument(
        "--TEST-MASKS-DIR",
        type=str,
        default=TEST_MASKS_DIR,
        help="Path to HPA testing masks directory"
    )
    parser.add_argument(
        "--MRI-TENSORBOARD-DIR",
        type=str,
        default=MRI_TENSORBOARD_DIR,
        help="Path to MRI Tensorboard directory"
    )
    parser.add_argument(
        "--HPA-TENSORBOARD-DIR",
        type=str,
        default=HPA_TENSORBOARD_DIR,
        help="Path to HPA Tensorboard directory"
    )
    parser.add_argument(
        "--MRI-MODEL-PATH",
        type=str,
        default=MRI_MODEL_PATH,
        help="Path to MRI model directory"
    )
    parser.add_argument(
        "--HPA-MODEL-PATH",
        type=str,
        default=HPA_MODEL_PATH,
        help="Path to HPA model directory"
    )

    args = parser.parse_args()
    return args
