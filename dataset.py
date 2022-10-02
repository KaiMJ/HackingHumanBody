import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PrepareData import make_tiles
from config import *

# Dataset
class HPADataset(Dataset):
    '''
    Dataset Class for the HPA dataset. 
    
    Parameters:
        IMG_DIR: path to images directory
        MASK_DIR: path to masks directory
        transform: transform function to augment the images
        normalize: normalize function to normalize the dataset
        idxs: train or validation indexes
    Returns:
        img, mask: tiled, resized image and mask.
    '''
    def __init__(self, IMG_DIR, MASK_DIR, transform=None, normalize=None, idxs=None):
        self.images = [os.path.join(IMG_DIR, f) for f in sorted(os.listdir(IMG_DIR))]
        self.masks = [os.path.join(MASK_DIR, f) for f in sorted(os.listdir(MASK_DIR))]

        if idxs is not None:
            self.images = [self.images[i] for i in idxs]
            self.masks = [self.masks[i] for i in idxs]
        
        self.transform = transform
        self.normalize = normalize
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = plt.imread(self.images[idx])
        # TODO: Figure out how plt reads and cv2 writes.
        # Sometimes mask reads 1/255 instead of 1.
        mask = plt.imread(self.masks[idx])
        if mask.max() < 0.5:
            mask = mask *255
        if self.normalize:
            img = self.normalize(image=img)['image']
        if self.transform:
            data = self.transform(image=img, mask=mask)
            img, mask = data['image'].transpose(2, 0, 1), data['mask']

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return img, mask
    
class HPATestset(Dataset):
    '''
    Dataset class for HPA test dataset.
    '''
    def __init__(self, TEST_IMAGES_DIR, TEST_MASKS_DIR, transform, normalize):
        self.images = [os.path.join(TEST_IMAGES_DIR, f) for f in sorted(os.listdir(TEST_IMAGES_DIR))]
        self.masks = [os.path.join(TEST_MASKS_DIR, f) for f in sorted(os.listdir(TEST_MASKS_DIR))]
        self.images = [plt.imread(f) for f in self.images]
        self.masks = [plt.imread(f)*255 for f in self.masks]
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        '''
        Returns a [4, 3, 256, 256] tiled image
        '''
        img, mask = self.images[idx], self.masks[idx]
        
        tiled_images, tiled_masks = make_tiles(img, mask, IMG_SIZE, exclude=False)
        tiled_images = np.array([self.normalize(image=t)['image'].transpose(2, 0, 1) for t in tiled_images])
        
        tiled_images = torch.from_numpy(tiled_images)
        tiled_masks = torch.from_numpy(np.array(tiled_masks))
        return tiled_images, tiled_masks
        

class MRIDataset(Dataset):
    '''
    Datataset class for MRI dataset with lower-grade giloma tumors labeled.
    '''
    def __init__(self, PRETRAIN_DIR, transform, idxs):
        '''
        Parameters:
            PRETRAIN_DIR: directory of the MRI data
            transform: transform function for data augmentation
            idxs: indexes of patients
        Returns:
            images_path: list of (patient_name, image_path)
            masks_path: list of (patient_name, mask_path)
            patients: dict of {patient_name: tuple of mean, and std}
        '''
        images_path = []
        masks_path = []
        patients = defaultdict(tuple)

        sort_order = lambda x: int(x.split('_')[4].split('.')[0])

        patient_folders = sorted(os.listdir(PRETRAIN_DIR))
        patient_folders = [patient_folders[i] for i in idxs \
                if patient_folders[i] not in ['data.csv', 'README.md']]

        for patient_name in patient_folders:
            patient_path = os.path.join(PRETRAIN_DIR, patient_name)
            images = []
            image_paths = []
            mask_paths = []
            for file_name in sorted(os.listdir(patient_path), key=sort_order):
                file_path = os.path.join(patient_path, file_name)
                if "mask" in file_path:
                    mask_paths.append((patient_name, file_path))
                else:
                    image_paths.append((patient_name, file_path))
                    img = plt.imread(file_path)
                    images.append(img)

            # Calculate mean and std of patient
            images = np.array(images)
            mean, std = self.get_mean_std(images)
            patients[patient_name] = (mean, std)

            images_path += image_paths
            masks_path += mask_paths

        self.patients = patients
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform

    def __len__(self): 
        '''
        Total number of images
        '''
        return len(self.images_path)

    def __getitem__(self, idx):
        '''
        Get a single image from random patient.

        Returns:
            img: Tensor [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
            mask: Tensor [BATCH_SIZE, IMG_SIZE, IMG_SIZE]
        '''
        # Get image and patient pair
        patient_name, image_path = self.images_path[idx]
        patient_name, mask_path = self.masks_path[idx]
        mean, std = self.patients[patient_name]

        img = plt.imread(image_path)
        img = ((img - mean) / std).astype(np.float32)
        mask = np.array(plt.imread(mask_path))

        data = self.transform(image=img, mask=mask)
        img, mask = data['image'], data['mask']
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask) / 255.0

        return img, mask, (torch.tensor(mean), torch.tensor(std))

    def get_mean_std(self, volume):
        '''
        Get mean and std of patient scans.

        Parameter:
            volume: 4D scan of a patient
        Returns:
            mean, std: mean and standard deviation with 3 channels
        '''
        mean = np.mean(volume, axis=(0, 1, 2))
        std = np.std(volume, axis=(0, 1, 2))
        return mean.astype(np.float32), std.astype(np.float32)
