import os
import glob
import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nibabel as nib
from monai import transforms
from monai.data import Dataset, CacheDataset
from tqdm import tqdm



def split_data(data_dir, test_size=0.15):
    """
    Split a list of files into train/val/test subsets based on provided fractions.
    """
    pattern = f"{data_dir}/**/*.nii.gz"
    nii_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(nii_files)} .nii.gz files in {data_dir}.")
    train_files, temp_df = train_test_split(nii_files, test_size=test_size*2, random_state=42)
    val_files, test_files = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Training set length: ", len(train_files))
    print("Validation set length: ", len(val_files))
    print("Test set length: ", len(test_files))

    train_data = [{"image": f, "file_path": f} for f in train_files]
    val_data   = [{"image": f, "file_path": f} for f in val_files]
    test_data  = [{"image": f, "file_path": f} for f in test_files]

    return train_data, val_data, test_data

        

def compute_global_stats(train_data, num_coeff, min_i, max_i, min_j, max_j, min_k, max_k, single_coeff=True):
    """
    Compute the global mean, standard deviation, minimum, and maximum intensity values 
    for each channel across all files in train_data.
    
    Assumes that each file is a NIfTI file containing a 4D volume, where the last 
    dimension corresponds to channels. The function selects the channels up to num_channels.
    """
    total_sum = 0.0       # Will store per-channel sum
    total_sq_sum = 0.0    # Will store per-channel sum of squares
    total_voxels = 0       # Total spatial voxels per channel (same for every channel)
    global_min = float('inf')      # Per-channel minimum values
    global_max = float('-inf')      # Per-channel maximum values


    print('Computing global stats')

    for batch in tqdm(train_data):

        image_path = batch["image"]
        data = nib.load(image_path).get_fdata()

        ## ======================================================

        # Add a small buffer around bounding box

        if single_coeff:
            data = data[min_i-2:max_i+2, min_j-2:max_j+2, min_k-2:max_k+2, num_coeff:num_coeff+1]
        else:
            data = data[min_i-2:max_i+2, min_j-2:max_j+2, min_k-2:max_k+2, :num_coeff]


        ## ======================================================

        # Sum over spatial dimensions (axis 0, 1, and 2) to get per-channel sums.
        file_sum = data.sum(axis=(0, 1, 2))
        file_sq_sum = np.sum(data**2, axis=(0, 1, 2))
        
        # Compute per-channel minimum and maximum
        file_min = data.min(axis=(0, 1, 2))
        file_max = data.max(axis=(0, 1, 2))
            
        total_sum += file_sum
        total_sq_sum += file_sq_sum
        global_min = np.minimum(global_min, file_min)
        global_max = np.maximum(global_max, file_max)
        
        # For each file, the number of spatial voxels is the product of the first three dimensions.
        total_voxels += np.prod(data.shape[:-1])

    # Calculate per-channel global mean.
    global_mean = total_sum / total_voxels
    # Calculate variance using E[x^2] - (E[x])^2 and then derive the standard deviation.
    variance = (total_sq_sum / total_voxels) - global_mean**2
    global_std = np.sqrt(variance)
    
    return global_mean, global_std, global_min, global_max
    

def prepare_coeff_data(
    data_dir,
    model_type: str,
    num_unet_layers: int, 
    resolution: float, 
    cache: bool = False,
    augment_train: bool = True,
    coeff = 0,
    bbox_csv: str = '',
    stats_csv: str = ''
):
    """
    Prepare the dataset by:
      1. Finding all .nii.gz files.
      2. Splitting into train/validation sets.
      3. Creating MONAI datasets with proper transforms.
    """
    

    ## ======================================================
    # Load and use pre computed bounding box vals for dataset

    
    df = pd.read_csv(bbox_csv, index_col='Dimension')
    min_i, max_i = df.loc['i', 'Min Index'], df.loc['i', 'Max Index']
    min_j, max_j = df.loc['j', 'Min Index'], df.loc['j', 'Max Index']
    min_k, max_k = df.loc['k', 'Min Index'], df.loc['k', 'Max Index']


    print(f'i inds: {min_i} -> {max_i}')
    print(f'j inds: {min_j} -> {max_j}')
    print(f'k inds: {min_k} -> {max_k}')

    crop_transform = transforms.SpatialCropd(
        keys=["image"],
        roi_start=(min_i-1, min_j-1, min_k-1),
        roi_end=(max_i+1, max_j+1, max_k+1)
    )

    ## ======================================================

    # Load the dataset and the splits.

    train_data, valid_data, test_data = split_data(data_dir)

     # Get the file path from the first item in the training list
    first_train_file_path = train_data[0]['image']
    
    # Load the NIfTI image using nibabel
    nifti_image = nib.load(first_train_file_path)

    # Get the data from the image as a NumPy array
    image_data = nifti_image.get_fdata()
    # Print the shape
    print("\n--- DEBUGGING INFO ---")
    print(f"Shape of the first training image ('{first_train_file_path}'): {image_data.shape}")
    print("----------------------\n")

    # Model specific coeficient transforms

    stats_df = pd.read_csv(stats_csv)
    train_mean = stats_df["means"][coeff]
    train_std = stats_df["stds"][coeff]

    print(f'Computed mean: {train_mean}, std: {train_std}')
    normalise_transform = transforms.NormalizeIntensityd(keys=["image"],subtrahend=train_mean, divisor=train_std)


    loading_transforms = [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        crop_transform,
        transforms.Spacingd(keys=['image'], pixdim=resolution),
        transforms.DivisiblePadd(keys=['image'], k=2**(num_unet_layers-1)),
        normalise_transform
    ]
    
    augmentation = [
        transforms.RandGaussianNoised(
            keys=["image"], 
            prob=0.2,
            mean=0,
            std=0.1
        )
    ] if augment_train else []
        
        
    train_transforms = transforms.Compose(loading_transforms + augmentation)
    valid_transforms = transforms.Compose(loading_transforms)

    trainset = CacheDataset(train_data, train_transforms, cache_rate=1, num_workers=8) \
        if cache else Dataset(train_data, train_transforms)

    validset = CacheDataset(valid_data, valid_transforms, cache_rate=1, num_workers=8) \
        if cache else Dataset(valid_data, valid_transforms)

    testset  = CacheDataset(test_data,  valid_transforms, cache_rate=1, num_workers=8) \
        if cache else Dataset(test_data, valid_transforms)

    return trainset, validset, testset




