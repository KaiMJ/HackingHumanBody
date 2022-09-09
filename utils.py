import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Resize, Normalize, RandomRotate90, \
                        HorizontalFlip, VerticalFlip, Transpose, \
                        ElasticTransform, GridDistortion, OpticalDistortion, \
                        RandomSizedCrop, CLAHE, RandomBrightnessContrast, RandomGamma
import argparse
import os
import shutil
from datetime import datetime
from config import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loss
def dice_loss(output, target, dim=None):
    if dim:
        num = torch.sum(output * target, dim=dim)
        den = torch.sum(output, dim=dim) + torch.sum(target, dim=dim)
    else:
        num = torch.sum(output * target)
        den = torch.sum(output) + torch.sum(target)
    loss = 2 * num / den
    return -loss

binary_crossentropy = nn.BCELoss()

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
        cv2.drawContours(img_mask_highlight, [c], -1, (255, 0, 0), thickness=10)
    return img_mask_highlight

def plot_images(writer, inputs, outputs, losses, n_iter, e, name, n=9):
    '''
    Plot Input and Output numpy arrays on Tensorboard.
        writer - SummaryWriter
        inputs - Batch of Numpy [Batch, H, W, C]
        outputs - Batch of Mask Numpy [Batch, H, W]
    '''
    plt.tight_layout()
    fig, axs = plt.subplots(int(n**0.5), int(n**0.5))
    for i, ax in enumerate(axs.flatten()):
        highlight = get_mask_highlight(inputs[i], outputs[i])
        ax.imshow((highlight * 255).astype(np.uint8))
        if losses is not None:
            ax.set_title(f"Dice: {losses[i]:.4f}")
    plt.suptitle(f"Epoch {e}")
    writer.add_figure(name, fig, n_iter)

def format_tensors_plt(inp, out, transform, n=9, mask=None):
    '''
        Format tensors and then plot.
    '''
    plt_imgs = transform(image=inp[:n].permute(0, 2, 3, 1).cpu().numpy())['image'] # [9, 256, 256, 3]
    plt_outs = threshold_tensor(out[:n]).detach().cpu().numpy().astype(np.uint8) # [9, 256, 256]
    if mask is not None:
        plt_masks = mask[:n].cpu().numpy().astype(np.uint8) # [9, 256, 256]
        return plt_imgs, plt_outs, plt_masks
    else:
        return plt_imgs, plt_outs

def get_transforms(means, stds, size):
    # TODO: Fix transformation. Get cropped and rotated mirroring.
    transform = A.Compose([
        Resize(size, size),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        Transpose(),
        # A.OneOf([
        #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #     A.GridDistortion(p=0.5),
        #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        # ], p=0.8),
        # A.RandomBrightnessContrast(p=0.8),
        ToTensorV2()
    ])

    transform_img = Normalize(mean=means, std=stds)

    test_transform = A.Compose([
        Resize(size, size),
        Normalize(mean=means, std=stds),
        ToTensorV2()
    ])

    inv_normalize = A.Compose([
        Normalize(mean = (0., 0., 0.), std=1/stds, max_pixel_value=1),
        Normalize(mean = -means, std = (1.0, 1.0, 1.0), max_pixel_value=1),
    ])

    transform = {"transform":transform, "transform_img": transform_img}
    return transform, test_transform, inv_normalize

def find_stats(full_dataset, size):
    # Find mean
    total = torch.zeros(3)
    for i, data in enumerate(full_dataset):
        image, mask = data['image'], data['mask']
        mean = torch.mean(image.view(3, -1), dim=1)
        total += mean

    mean = total / len(full_dataset)

    # Find STD
    total_var = torch.zeros(3)
    for i, data in enumerate(full_dataset):
        image, mask = data['image'], data['mask']
        total_var += ((image.view(3, -1) - mean.unsqueeze(1))**2).sum(1)

    std = torch.sqrt(total_var / (len(full_dataset)*size*size))
    # means = np.array([212.1089, 205.7680, 210.4203]) / 255
    # stds = np.array([41.9276, 48.7806, 45.6515]) / 255
    return mean, std

# Logging Experiments
def get_args():
    '''
    Prase arguments. Initialize log files. Return parser and log function.
    '''
    parser = argparse.ArgumentParser(description='Train UNet')
    with open(LOG_PATH, 'r+') as f:
        lines = f.readlines()
        numbers = [0]
        for line in lines:
            # --number=1
            numbers.append(int(line.split(' ')[0][9:]))
        
        for count in range(1, max(numbers)):
            if count not in numbers:
                break
        else:
            count = max(numbers) + 1

    for model_name in os.listdir(MODEL_PATH):
        # 'model_1'
        if int(model_name[6:]) not in numbers:
            file_path = os.path.join(MODEL_PATH, model_name)
            os.remove(file_path)
    for tensorboard_name in os.listdir(TENSORBOARD_PATH_DIR):
        # runs_1
        if int(tensorboard_name[5:]) not in numbers:
            file_path = os.path.join(TENSORBOARD_PATH_DIR, tensorboard_name)
            shutil.rmtree(file_path)

    parser.add_argument('--number', '-n', metavar='N', type=str, default=str(count), help='Model Number')

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='learning_rate', metavar='L', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--optimizer', '-o', metavar='O', type=str, default=OPTIMIZER, help='Optimizer')
    if OPTIMIZER == 'sgd':
        parser.add_argument('--momentum', '-m', metavar='M', type=float, default=MOMENTUM, help='Momentum')

    parser.add_argument('--split', '-s', metavar='S', type=float, default=SPLIT, help='Training Validation Split')
    parser.add_argument('--img-size', '-i', metavar='I', type=int, default=IMG_SIZE, help='Encoding Image Size')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--description', '-d', metavar='D', type=str, help='Description of Model')
    parser.add_argument('--time', '-t', type=str, default=now, help='Start of Experiment')

    return parser.parse_args()

def log_experiment(args, add):
    vars(args).update(add)
    with open(LOG_PATH, 'a') as f:
        line = ""
        for k, v in vars(args).items():
            line += f"--{k}={v} "
        line = line[:-1]
        f.write(line)
        f.write('\n')
