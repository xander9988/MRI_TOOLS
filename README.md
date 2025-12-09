
# `scan_augment.py` — Scan Augmentation Script

This Python script performs augmentation on medical imaging files (`.nii`) and their corresponding segmentation masks.  
It applies **random rotations (90°, 180°, or 270°)** and a **Gaussian blur** with a sigma value between **0.5 and 1.5**.

Augmented images are saved in the **same directory** with a prefix added to the filename.

##  How It Works
The script automatically scans a folder and detects valid image + segmentation pairs using the following naming format:

    IMAGE.nii <- Image
    awSeg_IMAGE.nii <- Segmentation Mask  

Augmented outputs will be generated as:

    aug_IMAGE.nii
    aug_awSeg_IMAGE.nii
  
 You can optionally augment only a portion of the dataset using the `--split` argument (default = augment everything).

## Usage

Run the script with:

    python scan_augment.py -i path/to/folder
    
|Arguement|Desription  |Defalt|
|--|--|--|
|`-i / --input`|Directory containing scans and segmentations|REQUIRED|
|`-p / --seg_prefix`|Prefix for segmentation masks|awSeg_
|`-s / --split| Percentage of dataset to augment | 1

Example:

    python scan_augment.py -i X:/scans_segs -s 0.5

