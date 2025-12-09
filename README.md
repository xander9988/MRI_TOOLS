## `scan_augment.py` Scan Augmentation Script

This script augments medical images (`.nii` only for now) along with their corresponding segmentation masks. It applies a random rotation (90°, 180°, or 270°) and Gaussian blur with a Sigma value ranging from `0.5-1.5`, then saves the augmented files in the same folder.

### How it Works

-   Looks for image + segmentation pairs using this naming format:
    
`IMAGE.nii awSeg_IMAGE.nii ← segmentation` 

-   Augmented files are saved with the prefix:

`aug_IMAGE.nii
aug_awSeg_IMAGE.nii` 

### Run Command

`python scan_augment.py -i path/to/folder` 


### Requirements

 - SimpleITK
 - Scipy
 - Numpy

 
