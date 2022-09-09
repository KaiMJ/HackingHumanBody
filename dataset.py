from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import os
from config import *

# Dataset
class HPADataset(Dataset):
    '''
        Dataset Class.
    '''
    def __init__(self, csv_file, root_dir, transform, val_file='saved/messed_up_ids.npy', test=False):
        csv = pd.read_csv(csv_file)
        val_ids = np.load(val_file)
        if test:
            self.annotations = csv[csv['id'].isin(val_ids)]
        else:
            self.annotations = csv[~csv['id'].isin(val_ids)]
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, str(img_row.id) + '.tiff')
        img = plt.imread(img_path)

        if not self.test:
            # Don't Normalize Mask
            mask = self.rle2mask(img_row.rle, shape=img.shape[:2])
            img = self.transform["transform_img"](image=img)["image"]
            sample = self.transform["transform"](image=img, mask=mask)
        else:
            sample = self.transform(image=img)

        # TODO: Data augmentation: Try splitting into smaller frames
        return sample

    def rle2mask(self, mask_rle: str, label=1, shape=(3000, 3000)):
        """
        mask_rle: run-length as string formatted (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = label
        return img.reshape(shape).T


def prepare(transform, test_transform, TENSORBOARD_PATH):
    # full_dataset = HPADataset(CSV_FILE, ROOT_DIR, transform=None)
    # means, stds = find_stats(full_dataset, 256)
    means = np.array([212.1089, 205.7680, 210.4203]) / 255
    stds = np.array([41.9276, 48.7806, 45.6515]) / 255

    # Prepare Dataset
    full_dataset = HPADataset(CSV_FILE, ROOT_DIR, transform=transform)
    # Split Dataset
    train_size = int(SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 325 * 0.8 Training Data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, drop_last=True)
    train_writer = SummaryWriter(TENSORBOARD_PATH + "/train")
    # 325 * 0.2 Validation Data
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=True)
    val_writer = SummaryWriter(TENSORBOARD_PATH + "/val")
    # 26 Dev-Test Data (messed up data)
    test_dataset = HPADataset(CSV_FILE, ROOT_DIR, transform=test_transform, test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_writer = SummaryWriter(TENSORBOARD_PATH + "/test")

    submission_writer = SummaryWriter(TENSORBOARD_PATH + "/submission")

    return train_loader, train_writer, val_loader, val_writer, test_loader, test_writer, submission_writer