import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def tensor_image_to_pil_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized tensor to a PIL image.
    
    Args:
    - tensor (torch.Tensor): The normalized tensor to convert.
    - mean (list): The mean values used for normalization.
    - std (list): The standard deviation values used for normalization.
    
    Returns:
    - PIL.Image: The resulting PIL image.
    """
    # Denormalize the tensor
    if (mean is not None) and (std is not None):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    
    # Convert the tensor to a numpy array
    numpy_image = tensor.permute(1, 2, 0).numpy()
    
    # Clip values to the valid range [0, 1] and convert to uint8
    numpy_image = np.clip(numpy_image, 0, 1)
    numpy_image = (numpy_image * 255).astype(np.uint8)
    
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(numpy_image)
    
    return pil_image


def tensor_mask_to_pil_image(tensor):
    """
    Convert a tensor of segmentation mask to a PIL image.
    
    Args:
    - tensor (torch.Tensor): The tensor to convert.
    Returns:
    - PIL.Image: The resulting PIL image.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert the tensor to a numpy array
    if len(tensor.shape) == 3:
        numpy_image = tensor.permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 2:
        numpy_image = tensor.numpy()

    # Clip values to the valid range [0, 1] and convert to uint8
    numpy_image = np.clip(numpy_image, 0, 1)
    numpy_image = (numpy_image * 255).astype(np.uint8)
    
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(numpy_image)
    
    return pil_image