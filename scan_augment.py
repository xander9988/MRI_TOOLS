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

def load_pairs(input_dir, seg_prefix="awSeg_"):
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
    return {k: v for k, v in pairs.items() if "img" in v and "seg" in v}

def process(input_dir, seg_prefix="awSeg_", split="1"):
    print(f"Loading pairs from {input_dir} with segmentation prefix '{seg_prefix}'")
    pairs = load_pairs(input_dir, seg_prefix)
    print(f"Starting augmentation on {len(pairs)} pairs with split {float(split)*100}%")
    for base, pair in pairs.items():
        if np.random.rand() > float(split):
            print(f"Skipping pair: Image: {pair['img']}, Segmentation: {pair['seg']}")
            continue
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
    parser.add_argument("-p", "--seg_prefix", default="awSeg_", help="Prefix for segmentation masks (default: awSeg_)")
    parser.add_argument("-s", "--split",default="1", help="percentage of dataset to augment. Default is all (1)")
    args = parser.parse_args()
    process(args.input, seg_prefix=args.seg_prefix, split=args.split)