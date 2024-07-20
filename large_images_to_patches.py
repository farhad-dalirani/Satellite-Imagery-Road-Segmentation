import math
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
import shutil
import cv2 as cv
import matplotlib.pyplot as plt


class LargeImagesToPatches:

    def __init__(
            self, in_image_dir: str, in_label_dir: str, 
            out_image_dir: str, out_label_dir: str,
            images_hw: tuple, patch_size: int,
            overlap_between_patches: int):
        
        self.in_image_dir = in_image_dir
        self.in_label_dir = in_label_dir
        self.out_image_dir = out_image_dir
        self.out_label_dir = out_label_dir
        self.images_hw = images_hw
        self.patch_size = patch_size
        self.overlap_between_patches = overlap_between_patches
        
        # Create folder for output
        self.create_folders()

        # Read file names of images and ground truth masks
        self.images_names = self.read_names_of_files(directory_path=self.in_image_dir)
        self.gt_masks_names = self.read_names_of_files(directory_path=self.in_label_dir)
        assert(len(self.images_names) == len(self.gt_masks_names)) 

    def create_folders(self):
        """Create new folders"""
        for folder_path in [self.out_image_dir, self.out_label_dir]:
            # Check if the folder exists
            if os.path.exists(folder_path):
                # Remove the folder and its contents
                shutil.rmtree(folder_path)

            # Create the folder
            os.makedirs(folder_path)

    def read_names_of_files(self, directory_path):
        """
        Read name of files in a directory and sort them to have a unique order each time.
        """
        files_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files_names.sort()
        return files_names

    def convert_image_to_patches(self, image_path, mask_path):
        """Extract patches from one image"""
        
        # Read image from file 
        image = cv.imread(filename=os.path.join(self.in_image_dir, image_path))
        # Read corresponding gt mask from file
        mask = cv.imread(filename=os.path.join(self.in_label_dir, mask_path))
        
        assert(image.shape[0]==self.images_hw[0])
        assert(image.shape[1]==self.images_hw[1])
        assert(mask.shape[0]==self.images_hw[0])
        assert(mask.shape[1]==self.images_hw[1])
        
        # Count number of white pixels
        num_white_pxs = np.sum(np.all(image == [255, 255, 255], axis=-1))
        
        # If number of empty pixels is larger than a value, reject image
        if (num_white_pxs / np.prod(self.images_hw)) > 0.1:
            return [] 
        
        # pixel value to rage [0,1]
        image = image.astype(float) / 255.0
        mask = mask.astype(float) / 255.0

        # Extract patches
        list_patches = []
        patch_id = 0
        for i in range(0, self.images_hw[0], self.patch_size):
            for j in range(0, self.images_hw[1], self.patch_size):
                
                # ij-patch of image
                crop_img = np.ones(shape=(self.patch_size, self.patch_size, 3))
                tmp_img = image[i:min(i+self.patch_size, self.images_hw[0]), j:min(j+self.patch_size, self.images_hw[1]), :]
                crop_img[0:tmp_img.shape[0], 0:tmp_img.shape[1], :] = tmp_img
                
                # If most of patch will be outside image, ignore it
                if ((self.patch_size - tmp_img.shape[0]) > (0.2 * self.patch_size)) \
                    or ((self.patch_size - tmp_img.shape[1]) > (0.2 * self.patch_size)):
                    continue  

                # ij-patch of mask
                crop_mask = np.zeros(shape=(self.patch_size, self.patch_size, 3))
                tmp_img = mask[i:min(i+self.patch_size, self.images_hw[0]), j:min(j+self.patch_size, self.images_hw[1]), :]
                crop_mask[0:tmp_img.shape[0], 0:tmp_img.shape[1], :] = tmp_img

                img_name_parts = image_path.split('.')
                patch_name = "{}-{}.png".format(img_name_parts[0], patch_id)
                list_patches.append((crop_img, crop_mask, patch_name))

                # Increment number of patches                
                patch_id += 1 

        return list_patches

    def convert_dataset_from_images_to_patches(self):
        """Convert all images and their corresponding gt mask to patches and save them in file"""

        num_processed_imgs = 0

        for path_image_i, path_mask_i in zip(self.images_names, self.gt_masks_names):
            
            # For the i-th image get patches
            list_patches = self.convert_image_to_patches(image_path=path_image_i, mask_path=path_mask_i)

            # Save patches in files
            for patch_image_i, patch_mask_i, patch_name_i in list_patches:
                wr1 = cv.imwrite(os.path.join(self.out_image_dir, patch_name_i), (patch_image_i * 255).astype(np.uint))
                wr2 = cv.imwrite(os.path.join(self.out_label_dir, patch_name_i), (patch_mask_i * 255).astype(np.uint))

                if not(wr1 and wr2):
                    raise ValueError('Can not write patches in file.')
            
            num_processed_imgs += 1 
            if num_processed_imgs % 10 == 0:
                print("Patchified {}/{}".format(num_processed_imgs, len(self.images_names)))


if __name__ == '__main__':
    
    # Convert train, validation and test set to smaller patches
    for set_i in ['train', 'val', 'test']:
        
        print("Processing set {}".format(set_i))
        
        data = LargeImagesToPatches(
            in_image_dir='data/massachusetts-roads-dataset/tiff/{}'.format(set_i),
            in_label_dir='data/massachusetts-roads-dataset/tiff/{}_labels'.format(set_i), 
            out_image_dir='data/massachusetts-roads-dataset/tiff/{}_patched'.format(set_i), 
            out_label_dir='data/massachusetts-roads-dataset/tiff/{}_labels_patched'.format(set_i),
            images_hw= (1500, 1500),
            patch_size=515,
            overlap_between_patches=15)
        
        data.convert_dataset_from_images_to_patches()