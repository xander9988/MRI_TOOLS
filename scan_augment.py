import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import os

class ScanAugmenter:
    def __init__(self):
        pass

    def random_rotation(self, image, segmentation=None, angle=None):
        transform = sitk.Euler3DTransform()
        transform.SetRotation(np.deg2rad(angle), 0, 0)
        return sitk.Resample(image, transform, sitk.sitkLinear, 0, image.GetPixelID())

    def gaussian_blur(self, image, sigma=None):
        array = sitk.GetArrayFromImage(image)
        blurred_array = gaussian_filter(array, sigma=sigma)
        return sitk.GetImageFromArray(blurred_array,isVector=image.GetNumberOfComponentsPerPixel() > 1)
    
    def augment(self, image, segmentation=None):
        angles = [90, 180, 270]
        angle = np.random.choice(angles)
        sigma = np.random.uniform(0.5, 1.5)
        image = self.random_rotation(image, angle=angle)
        image = self.gaussian_blur(image, sigma=sigma)

        segmentation = self.random_rotation(segmentation, angle=angle) if segmentation else None
        return image, segmentation

def process_file(image_path, augmenter):
    filename = os.path.basename(image_path)
    folder = os.path.dirname(image_path)
    if filename.startswith("aug_"):
        return

    print(f" Processing: {filename}")
    image = sitk.ReadImage(image_path)
    aug_image, _ = augmenter.augment(image)
    output_path = os.path.join(folder, f"aug_{filename}")
    sitk.WriteImage(aug_image, output_path)
    print(f"Saved: {output_path}")

def load_pairs(input_dir, seg_prefix="awSeg_"): #this assumes that the files will have awSeg_ prefix for segmentation masks (like in the shared drive folder)
    files = os.listdir(input_dir)
    pairs = {}
    for file in files:
        if os.path.isdir(os.path.join(input_dir, file)):
            continue
        if file.startswith(seg_prefix):
            base = file.replace(seg_prefix, "")
            pairs.setdefault(base, {})["seg"] = file
        else:
            pairs.setdefault(file, {})["img"] = file
    for k, v in pairs.items():
        if "img" in v and "seg" in v:
            print(f"Found pair: Image: {v['img']}, Segmentation: {v['seg']}")
    print(f"Total pairs found: {len([1 for v in pairs.values() if 'img' in v and 'seg' in v])}")
    return {k: v for k, v in pairs.items() if "img" in v and "seg" in v}

def process(input_dir):
    pairs = load_pairs(input_dir)
    for base, pair in pairs.items():
        img_path = os.path.join(input_dir, pair["img"])
        seg_path = os.path.join(input_dir, pair["seg"])
        img = sitk.ReadImage(img_path)
        seg = sitk.ReadImage(seg_path)
        augmenter = ScanAugmenter()
        aug_img, aug_seg = augmenter.augment(img, seg)
        sitk.WriteImage(aug_img, os.path.join(input_dir, f"aug_{pair['img']}"))
        sitk.WriteImage(aug_seg, os.path.join(input_dir, f"aug_{pair['seg']}"))
        print(f"Saved aug_{pair['img']} and aug_{pair['seg']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch augment paired images and masks.")
    parser.add_argument("-i", "--input", required=True, help="Directory containing image + awSeg_ masks")
    args = parser.parse_args()
    process(args.input)


#TO DO
# - Add more augmentation techniques?
# - Allow only a % of the dataset to be augmented?