import albumentations as A
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms
from transformers import AutoProcessor

class PARDataset(Dataset):
    """ Custom Dataset class for loading attribute images and annotations. """

    #--- Set the attribute keys ---#
    attribute_keys = ['upper_color', 'lower_color', 'gender', 'bag', 'hat']
        
    def __init__(self, data_folder : str, annotation_path : str, augment : bool = True):
        """Initialize the dataset with the path to the images and the annotations.

        Parameters
        ----------
        data_folder : str
            Path to the folder containing the images.
        annotation_path : str
            Path to the CSV file containing the annotations.
        augment : bool
            Whether to apply data augmentation.
        """
        #--- Set the attributes ---#
        self.data_folder = data_folder
        
        #--- Load annotations into a DataFrame ---#
        annotations = pd.read_csv(annotation_path)

        # Filter out rows with missing images
        annotations['image_exists'] = annotations.apply(
            lambda row: os.path.exists(os.path.join(data_folder, row.iloc[0])), axis=1
        )
        annotations = annotations[annotations['image_exists']]

        # Filter out rows with unknown values (-1)
        annotations = annotations[~annotations.iloc[:, 1:].eq(-1).any(axis=1)]

        # Drop the 'image_exists' column as it's no longer needed
        annotations.drop(columns=['image_exists'], inplace=True)

        #--- Set the annotations ---#
        self.annotations = annotations                              # > 81737 samples

        #--- Initialize the preprocessor ---#
        self.processor = AutoProcessor.from_pretrained("./models/siglip2-so400m-patch16-naflex_VE", use_fast=True)

        #--- Apply data augmentation if specified ---#      
        if augment:
            self.transforms = A.Compose([
                A.CLAHE(clip_limit=(1, 3), tile_grid_size=(13, 13), p=0.2),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Affine(scale=1, translate_percent=0.05, rotate=3, p=0.15),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.2),
                A.Illumination(mode='gaussian', intensity_range=(0.05, 0.1), effect_type="both", angle_range=(0, 360), center_range=(0.5, 0.5), sigma_range=(0.5, 1), p=0.25),
                A.RingingOvershoot(blur_limit=(3, 7), cutoff=(np.pi/4, np.pi/2), p=0.2),
            ])
        else:
            self.transforms = None
        
    def __len__(self) -> int:
        """ Return the number of samples in the dataset. 
        
        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx : int) -> dict:
        """Retrieve an image and its annotations by index.

        Parameters
        ----------
        idx : int
            Index of the data sample.
        
        Returns
        -------
        dict
            A sample containing the image and its annotations.        
        """
        #--- Get the image name and attribute values ---#
        img_name = os.path.join(self.data_folder, self.annotations.iloc[idx, 0])
        attribute_values = self.annotations.iloc[idx, 1:]

        #--- Open the image with PIL ---#
        image = Image.open(img_name).convert("RGB")

        #--- Get the attributes associated with the image ---#
        attribute_tensors = [torch.tensor(val) for val in attribute_values]
        attributes = {
            key: (val - 1).long() if key in ['upper_color', 'lower_color'] else val.float()
            for key, val in zip(self.attribute_keys, attribute_tensors)
        }
        
        #--- Apply the transformations, if any ---#
        if self.transforms:
            image_np = np.array(image)
            image_np = self.transforms(image=image_np)["image"]
            image = Image.fromarray(image_np)

        #--- Now pass the correctly formatted image to the processor ---#
        inputs = self.processor(image, return_tensors="pt")
                
        #--- Return the sample ---#
        return {
            "inputs": inputs,
            "attributes": attributes,
        }