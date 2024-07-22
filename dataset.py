import json
import numpy as np
import math
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from utils import tensor_image_to_pil_image, tensor_mask_to_pil_image
from torch.utils.data import Dataset, DataLoader


class MRDDataset(Dataset):
    """
    Class for reading the Massachusetts Roads Dataset, which consists of satellite 
    images and their corresponding ground truth segmentation masks.
    """

    def __init__(self, image_dir: str, label_dir: str, images_wh: tuple, transformas):
        
        super(MRDDataset, self).__init__()

        # Directory of images
        self.image_dir = image_dir
        # Directory of mask ground truth
        self.label_dir = label_dir
        # Images size in dataset
        self.images_width, self.images_height = images_wh

        # Read files' name of images and ground truth masks
        self.images_names = self.read_names_of_files(directory_path=self.image_dir)
        self.gt_masks_names = self.read_names_of_files(directory_path=self.label_dir)
        assert(len(self.images_names) == len(self.gt_masks_names))    

        # Number of images in directory
        self.num_images = len(self.images_names)
        
        # Transformations to apply of images such as convert to tensor, etc
        self.transforms = transformas

    def read_names_of_files(self, directory_path):
        """
        Read name of files in a directory and sort them to have a unique order each time.
        """
        files_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files_names.sort()
        return files_names


    def __len__(self):
        """Length of dataset"""
        return self.num_images


    def __getitem__(self, index):
        """
        Get a (image, ground truth mask) from dataset by index
        """
        
        # Path of image of index'th sample in dataset
        img_path = self.images_names[index]
        # Path of image of index'th sample in dataset
        mask_path = self.gt_masks_names[index]

        # Read image
        image_i = Image.open(os.path.join(self.image_dir, img_path))
        # Read mask
        mask_i = Image.open(os.path.join(self.label_dir, mask_path))

        image_tensor, mask_tensor = self.transforms(image_i, mask_i)

        # Convert 3 channel to 1 since with have 2 class
        mask_tensor = torch.mean(mask_tensor, axis=0).to(torch.float32)
        
        return image_tensor, mask_tensor

        
class JointTransform:
    """Class to apply transformations both to the image and mask"""
    def __init__(self, joint_transform=None, image_transform=None):
        self.joint_transform = joint_transform
        self.image_transform = image_transform

    def __call__(self, img, mask):
        if self.joint_transform is not None:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            img = self.joint_transform(img)
            torch.manual_seed(seed)
            mask = self.joint_transform(mask)
        
        if self.image_transform is not None:
            img = self.image_transform(img)
            
        return img, mask


if __name__ == '__main__':
    
    # Define the path to the JSON configuration file
    config_file_path = 'config/config.json'

    # Open and read the JSON file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Define the joint transformations for both image and mask
    joint_transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=config['test_patch_size'], scale=(0.75, 1), interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    joint_transform_test = transforms.Compose([transforms.ToTensor()])
    
    # Define the image-specific transformations
    image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transformations = JointTransform(joint_transform=joint_transform_train, image_transform=image_transform)
    test_transformations = JointTransform(joint_transform=joint_transform_test, image_transform=image_transform)

    # train dataset
    set_i = 'train'
    train_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=train_transformations)
    
    img_i, mask_i = train_ds[38]
    pil_img_i = tensor_image_to_pil_image(img_i.clone())
    pil_mask_i = tensor_mask_to_pil_image(mask_i.clone())
    plt.figure()
    plt.imshow(pil_img_i)
    plt.figure()
    plt.imshow(pil_mask_i)
    plt.show()

    # Train dataloader
    dataloader_train = DataLoader(dataset=train_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_train)))

    set_i = 'test'
    test_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=test_transformations)
    
    img_i, mask_i = test_ds[28]
    pil_img_i = tensor_image_to_pil_image(img_i.clone())
    pil_mask_i = tensor_mask_to_pil_image(mask_i.clone())
    plt.figure()
    plt.imshow(pil_img_i)
    plt.figure()
    plt.imshow(pil_mask_i)
    plt.show()

    # Test dataloader
    dataloader_test = DataLoader(dataset=test_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_test)))