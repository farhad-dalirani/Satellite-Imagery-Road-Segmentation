import os
import numpy as np
from PIL import Image
import json
import torch
import torchvision.transforms as transforms
from model import UnetLikeSegmentatorModel
from dataset import JointTransform
import argparse
from utils import tensor_mask_to_pil_image
from segment_images import segment_image

def dice(y_true, y_pred):
    """
    Calculate the Dice coefficient for binary segmentation.

    Args:
    y_true (numpy array): Ground truth binary mask.
    y_pred (numpy array): Predicted binary mask.

    Returns:
    float: Dice coefficient.
    """
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy for binary segmentation.

    Args:
    y_true (numpy array): Ground truth binary mask.
    y_pred (numpy array): Predicted binary mask.

    Returns:
    float: Accuracy.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    correct = (y_true_f == y_pred_f).sum()
    total = y_true_f.shape[0]
    return correct / total

def iou(y_true, y_pred):
    """
    Calculate the Intersection over Union (IoU) for binary segmentation.

    Args:
    y_true (numpy array): Ground truth binary mask.
    y_pred (numpy array): Predicted binary mask.

    Returns:
    float: Intersection over Union.
    """
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


if __name__ == '__main__':

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the path to the JSON configuration file
    config_file_path = 'config/config.json'
    # Open and read the JSON file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Path to images directory
    images_dir = os.path.join(config['data_dir'], 'test') 
    # Path to labels directory
    label_dir_path = os.path.join(config['data_dir'], 'test_labels') 

    # Define the joint transformations for both image and mask
    joint_transform_test = transforms.Compose([transforms.ToTensor()])
    image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transformations = JointTransform(joint_transform=joint_transform_test, image_transform=image_transform)

    # Load segmentation model weights
    model = UnetLikeSegmentatorModel()
    model.to(device=device)
    model.load_state_dict(torch.load(config['train_save_dir'], map_location=device))
    model.eval()

    # Get a list of all files and directories in the specified directory
    images_file_names = os.listdir(images_dir)
    images_file_names.sort()
    labels_file_names = os.listdir(label_dir_path)
    labels_file_names.sort()
    assert(len(images_file_names) == len(labels_file_names))

    acc_scores = []
    iou_scores = []
    dice_scores = []

    # For each image in input perform segmentation
    for idx, paths in enumerate(zip(images_file_names, labels_file_names)):
        
        img_path_i, label_path_i = paths
        
        print("Processing ({}, {}), image {}/{}...".format(img_path_i, label_path_i, idx+1, len(images_file_names)))

        # Read i-th image and its label
        image_i = Image.open(os.path.join(images_dir, img_path_i))
        label_i = Image.open(os.path.join(label_dir_path, label_path_i))

        # Segment image and get segmentation mask in PIL image format
        segmentation_mask_i = segment_image(
                                config=config,
                                model=model, 
                                image=image_i, 
                                device=device, 
                                img_transformations=test_transformations)
        
        label_i = np.array(label_i) / 255
        
        segmentation_mask_i = np.array(segmentation_mask_i)[:,:,0]/255
        segmentation_mask_i[segmentation_mask_i >= 0.5] = 1.0
        segmentation_mask_i[segmentation_mask_i < 0.5] = 0.0

        acc_scores.append(accuracy(y_true=label_i, y_pred=segmentation_mask_i))
        iou_scores.append(iou(y_true=label_i, y_pred=segmentation_mask_i))
        dice_scores.append(dice(y_true=label_i, y_pred=segmentation_mask_i))

        print("Last acc {}, IoU {}, Dice {}".format(acc_scores[-1], iou_scores[-1], dice_scores[-1]))
    
    print('Test Mean Accuracy {}'.format(np.mean(acc_scores)))
    print('Test Mean IOU      {}'.format(np.mean(iou_scores)))
    print('Test Mean Dice     {}'.format(np.mean(dice_scores)))