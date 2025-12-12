import os
import glob
import csv
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    Lambda,
    MapTransform,
    RandomizableTransform,
)
from monai.config import KeysCollection
from monai.data import Dataset, CacheDataset
from tqdm import tqdm
from pathlib import Path

    

def split_data(csv_dir, cond=False, test=False):
    """
    Split a list of files into train/val/test subsets based on provided fractions.
    """
    df = pd.read_csv(csv_dir)
    train_df = df[df['split']=='train']
    val_df = df[df['split']=='val']
    test_df = df[df['split']=='test']

    print("Training set length: ", train_df.shape[0])
    print("Validation set length: ", val_df.shape[0])
    print("Test set length: ", test_df.shape[0])


    if not test:

        if cond:
            train_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in train_df.itertuples()]
            val_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in val_df.itertuples()]
            test_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in test_df.itertuples() if r.deg == 0.0]
        else:
            
            train_data = [{"tracts": r.tract_path} for r in train_df.itertuples()]
            val_data = [{"tracts": r.tract_path} for r in val_df.itertuples()]
            test_data = [{"tracts": r.tract_path} for r in test_df.itertuples()]

    else:   
            train_data = None
            val_data = None
            test_data = [{"latent": r.latent_path, "odf": r.odf_path} for r in test_df.itertuples() if r.deg == 0.0]


    print(f"Filtered test set size (non-rotated only): {len(test_data)}")

    return train_data, val_data, test_data



def split_data_temp(csv_dir, temp_dir, cond=False):
    """
    Creates a temporary manifest CSV in temp_dir with updated file paths,
    then splits the data into train/val/test subsets.
    """
    # 1. Load the original CSV file
    original_df = pd.read_csv(csv_dir)
    df = original_df.copy() # Work on a copy

    # 2. Update path columns to point to the temporary directory
    print(f"Updating file paths to point to temporary directory: {temp_dir}")
    
    # Use os.path.basename to get just the filename from the old path
    df['tract_path'] = df['tract_path'].apply(
        lambda p: os.path.join(temp_dir, os.path.basename(p))
    )
    
    # If conditioning is on, update the latent path as well
    if cond and 'latent_path' in df.columns:
        df['latent_path'] = df['latent_path'].apply(
            lambda p: os.path.join(temp_dir, os.path.basename(p))
        )

    # 3. Save the new, temporary manifest CSV inside temp_dir
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    temp_csv_path = os.path.join(temp_dir, 'temp_manifest.csv')
    df.to_csv(temp_csv_path, index=False)
    print(f"Temporary manifest file saved to: {temp_csv_path}")

    # --- NEW: Print the head of the DataFrame for verification ---
    print("\n--- Verifying top 5 rows of the new manifest ---")
    print(df.head().to_string()) # .to_string() ensures nice formatting
    print("--------------------------------------------------\n")
    # --- END NEW ---

    # 4. Proceed with splitting using the updated DataFrame
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print("Training set length: ", train_df.shape[0])
    print("Validation set length: ", val_df.shape[0])
    print("Test set length: ", test_df.shape[0])

    # 5. Create the final data lists from the DataFrames with corrected paths
    if cond:
        train_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} 
                      for r in train_df.itertuples()]
        val_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} 
                    for r in val_df.itertuples()]
        test_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} 
                     for r in test_df.itertuples() if r.deg == 0.0]
        print(f"Filtered test set size (non-rotated only): {len(test_data)}")
    else:
        train_data = [{"tracts": r.tract_path} for r in train_df.itertuples()]
        val_data = [{"tracts": r.tract_path} for r in val_df.itertuples()]
        test_data = [{"tracts": r.tract_path} for r in test_df.itertuples()]

    return train_data, val_data, test_data



class StreamlineNormalize(MapTransform):
    """
    A MONAI dictionary transform that normalizes streamline coordinates.

    The input is expected to be a dictionary with key "image".
    The "image" value should be a NumPy array of shape (num, 256, 3), where 3 corresponds to the x, y, and z coordinates.
    
    The normalization is computed as:
    
        normalized = (image - min_coords) / (max_coords - min_coords)
    
    and the result is converted to a torch.Tensor before being stored back into the dict.

    Args:
        min_coords (tuple, list, or np.ndarray): Minimum values for each coordinate (x, y, z).
        max_coords (tuple, list, or np.ndarray): Maximum values for each coordinate (x, y, z).
    """
    def __init__(self, min_coords, max_coords):
        self.min_coords = np.array(min_coords)
        self.max_coords = np.array(max_coords)

    def __call__(self, data):
        # Expect data to be a dict with the key "image"
        image = data["tracts"]
        # Ensure the image is a NumPy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        # Normalize the coordinates:
        # For each coordinate: (value - min) / (max - min)
        normalized = (((image - self.min_coords) / (self.max_coords - self.min_coords)) * 2) - 1
        # Optionally, you might want to clip values to [0, 1]
        
        # Convert the normalized NumPy array to a torch tensor
        data["tracts"] = normalized

        return data

class StreamlineSample(RandomizableTransform, MapTransform):


    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 1.0,
        allow_missing_keys: bool = False,
        num_samples: int = 128
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.num_samples = num_samples


    def __call__(self, data):

        tracts = data["tracts"]

        if tracts.shape[0] < self.num_samples:
            raise ValueError(f"Cannot sample {self.num_samples} tracts from data")

        selected_inds = np.sort(np.random.choice(tracts.shape[0], size=self.num_samples))
        data['tracts'] = tracts[selected_inds]

        return data


def prepare_diff_data(
    csv_file, 
    temp_dir,
    cache: bool = False,
    num_coeffs=6,
    num_samples=128,
    cond=False,
    return_std=False,
    test=False,
    streamline_csv=''
    ):
    """
    Prepare the dataset by:
      1. Finding all .nii.gz files.
      2. Splitting into train/validation sets.
      3. Creating MONAI datasets with proper transforms.
    """
    
    if temp_dir is None:
        
        train_data, valid_data, test_data = split_data(csv_file, cond=cond, test=test)
        
    else:

        train_data, valid_data, test_data = split_data_temp(csv_file, temp_dir, cond=cond)

    ## ======================================================
    # Model specific coeficient transforms

    if cond:
        select_coeffs = Lambda(func=lambda data: {**data, "latent": data["latent"][:num_coeffs,...]})
    

    ## ======================================================
    
    # Streamline specific transforms

    df = pd.read_csv(streamline_csv)

    # 2. Get the first row (assuming the values are in the first row)
    row = df.iloc[0]

    # 3. Assign the specific columns to your lists
    mins = [row['min_x'], row['min_y'], row['min_z']]
    maxs = [row['max_x'], row['max_y'], row['max_z']]

    print(f'Computed streamline mins: {mins}, maxs: {maxs}')

    if not test:
        normalise_streamlines = StreamlineNormalize(mins, maxs)
        sample_streamlines = StreamlineSample(keys=["tracts"], num_samples=num_samples)

    ## ======================================================

    # Ensure float32 datatype

    convert_tracts = Lambda(func=lambda data: {**data, "tracts": data["tracts"].astype(np.float32)})

    if cond:
        
        convert_latent = Lambda(func=lambda data: {**data, "latent": data["latent"].astype(np.float32)})

    ## ======================================================

    if not test:

        if cond:

            loading_transforms = [
                LoadImaged(keys=["latent"]),
                convert_latent,
                LoadImaged(keys=["tracts"]),
                select_coeffs,
                normalise_streamlines,
                convert_tracts,
                sample_streamlines
            ]


        else:

            loading_transforms = [
                LoadImaged(keys=["tracts"]),
                normalise_streamlines,
                convert_tracts,
                sample_streamlines,
            ]

            
    else:
            loading_transforms = [
                LoadImaged(keys=["latent"]),
                convert_latent,
                select_coeffs
            ]
        

        
    train_transforms = Compose(loading_transforms)    
    valid_transforms = Compose(loading_transforms)


    testset  = CacheDataset(test_data,  valid_transforms, cache_rate=1, num_workers=8) \
        if cache else Dataset(test_data, valid_transforms)
    

    if not test:

        trainset = CacheDataset(train_data, train_transforms, cache_rate=1, num_workers=8) \
            if cache else Dataset(train_data, train_transforms)

        validset = CacheDataset(valid_data, valid_transforms, cache_rate=1, num_workers=8) \
            if cache else Dataset(valid_data, valid_transforms)
        
        output = [trainset, validset, testset]
        

    else:

        output = testset

    return output

