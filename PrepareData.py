# Download dataset and create new data directories


# Create a modified dataset with tiled, reshaped images and masks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio
import cv2


def make_tiles(img, mask, IMG_SIZE, exclude=True):
    '''
    Tile an image into 4 smaller images, resize them, 
        and return if mask is not empty.
        
    Parameters:
        img: numpy array image
        mask: numpy array with same shape as img
    Returns:
        tiled_imgs: tiled images with batch size 0~4
        tiled_masks: tiled masks with batch size 0~4
    '''
    h, w = int(img.shape[0]/2), int(img.shape[1]/2)
    
    tiled_masks = np.array([mask[:h, :w], mask[:h, w:], mask[h:, :w], mask[h:, w:]])
    
    idxs = []
    for i in range(len(tiled_masks)):
        if tiled_masks[i].sum() != 0:
            idxs.append(i)
            
    tiled_imgs = np.array([img[:h, :w], img[:h, w:], img[h:, :w], img[h:, w:]])

    if not exclude:
        idxs = np.arange(4)
    # Exclude if mask does not exist in the image
    tiled_imgs = [cv2.resize(tiled_imgs[i], (IMG_SIZE, IMG_SIZE)) for i in idxs]
    tiled_masks = [cv2.resize(tiled_masks[i], (IMG_SIZE, IMG_SIZE)) for i in idxs]
    return tiled_imgs, tiled_masks

def rle2mask(mask_rle: str, label=1, shape=(3000, 3000)):
    """
    Conver rle encoding to numpy mask image.
    
    Parameters:
        mask_rle: run-length as string formatted (start length)
        shape: (height,width) of array to return
    Returns:
        numpy array: 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape).T

def create_modified_dataset(DATA_CSV, IMAGES_DIR, MOD_IMAGES_DIR, MOD_MASKS_DIR, TEST_IMAGES_DIR, TEST_MASKS_DIR, IMG_SIZE):
    '''
    Create a modified dataset into MOD_IMAGES_DIR and MOD_MASKS_DIR.
    
    Parameters:
        csv_path: csv path to mask encoding and meta data
        IMAGES_DIR: input directory
        MOD_IMAGES_DIR: output images directory
        MOD_MASKS_DIR: output masks directory
        IMG_SIZE: size of reshaped images
    '''
    csv = pd.read_csv(DATA_CSV)
    invalid_id = 31800 # Top quarter of image is ruined.
    
    images_names = sorted(os.listdir(IMAGES_DIR))
    np.random.shuffle(images_names)
    
    # Create 9 test images
    test_n = 9
    for img_name in (images_names):
        id = int(img_name[:-5])
        if id == invalid_id:
            continue
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = plt.imread(img_path)
        rle = csv[csv.id == id].iloc[0].rle
        mask = rle2mask(rle, shape=img.shape[:2])
        
        if test_n > 0:
            cv2.imwrite(TEST_IMAGES_DIR+f"/{id}.png", img)
            cv2.imwrite(TEST_MASKS_DIR+f"/{id}.png", mask)
            test_n -= 1
        else:
            imgs, masks = make_tiles(img, mask, IMG_SIZE)
            for i in range(len(imgs)):
                # plt.imread reads scales ints to 0-1
                cv2.imwrite(MOD_IMAGES_DIR+f"/{id}_{i}.png", imgs[i])
                cv2.imwrite(MOD_MASKS_DIR+f"/{id}_{i}.png", masks[i])

def create_gif(PRETRAIN_DIR, n=6):
    '''
    Create n random image-mask pair gifs of MRI scans.
    
    Parameters:
        PRETRAIN_DIR: MRI scans directory
        n: number of gifs to create
    '''

    sort_order = lambda x: int(x.split('_')[4].split('.')[0])
    
    pretrain_folders = [f for f in sorted(os.listdir(PRETRAIN_DIR)) if f not in ['data.csv', 'README.md']]
    rand_folders = np.random.choice(pretrain_folders, n, replace=False)
    for folder in rand_folders:
        scan_path = os.path.join(PRETRAIN_DIR, folder)
        scan_folder = sorted(os.listdir(scan_path), key=sort_order)

        scans = []
        for file in scan_folder:
            if file[-8:] == 'mask.tif': # Skip masks to avoid duplicates
                continue
            else: # Get the Image, mask Pair
                img_path = os.path.join(scan_path, file)
                mask_path = img_path[:-4] +'_mask.tif'
                img = plt.imread(img_path)
                mask = plt.imread(mask_path)
                # Expand channels to 3 and keep only green channel
                mask = np.expand_dims(mask, axis=-1)
                mask = (np.concatenate((mask, mask,mask), axis=-1) * np.array([0, 1, 0])).astype('uint8')
                combined = cv2.addWeighted(img, 1, mask, 0.2, 0)
                scans.append(combined)

        gif_path = img_path.split('/')[1]
        imageio.mimsave(f'images/{gif_path}.gif', scans, duration=0.1)
