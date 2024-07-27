import os
from PIL import Image
import json
import torch
import torchvision.transforms as transforms
from model import UnetLikeSegmentatorModel
from dataset import JointTransform
import argparse
from utils import tensor_mask_to_pil_image


def segment_image(config, model, image, device, img_transformations):
    """
    Segments an input image using a specified model and configuration by processing it in smaller patches.

    Args:
        config (dict): Configuration dictionary containing parameters for image patch size, overlap, and batch size.
        model (torch.nn.Module): PyTorch model for image segmentation.
        image (PIL.Image): Input image to be segmented.
        device (torch.device): Device (CPU or GPU) on which to perform the computations.
        img_transformations (callable): Function or transform to apply to image patches before feeding them to the model.

    Returns:
        PIL.Image: A PIL Image object containing the segmented output of the whole image.

    Description:
        The function processes the input image by cropping it into smaller patches based on the configuration provided.
        Each patch is then transformed and fed into the segmentation model in batches. The model's outputs are 
        then combined to reconstruct the segmented output for the entire image. The final result is a PIL image where
        the segmentation masks are assembled back into the original image dimensions.

    Notes:
        - The input image is divided into overlapping patches to manage the memory limitations of edge devices.
        - The function handles the reconstruction of the segmented image by positioning the patches back into their 
          original locations.
        - Ensure that the `img_transformations` function is capable of handling and transforming image patches correctly.
    """ 
    batches = []
    batch = []

    assert(config['deployment_overlap_between_patches'] % 2 == 0)
    assert((config['deployment_patch_size']//32) % 2 == 0)

    img_width, img_height = image.size

    # crop image to smaller patches  instead of feeding whole 
    # stellite image to deal with limitation of edge devices
    patches_top_left = []
    list_patches_left_over = []
    overlap_size = config['deployment_overlap_between_patches']
    stride = config['deployment_patch_size']-overlap_size
    half_of_patches_overlap = overlap_size//2
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            
            # Keep top left coordinate for sticking patches back
            patches_top_left.append((x, y))
            
            # Leftover of each side of the image during patching (W, N, E, S).
            # Since patches have overlap, we pick half of the overlap from the involved patches.            
            left_over_image_i = [0, 0, 0, 0]
            if 0 < x:
                left_over_image_i[0] = half_of_patches_overlap
            if x + config['deployment_patch_size'] < img_width - overlap_size:
                left_over_image_i[2] = -half_of_patches_overlap
            if 0 < y:
                left_over_image_i[1] = half_of_patches_overlap
            if y + config['deployment_patch_size'] < img_width - overlap_size:
                left_over_image_i[3] = -half_of_patches_overlap
            list_patches_left_over.append(left_over_image_i)

            # Crop image to get patch
            patch_k = image.crop((x, y, x+config['deployment_patch_size'], y+config['deployment_patch_size']))

            # Convert to tensor
            batch.append(img_transformations(patch_k, None)[0])
            if len(batch) == config['deployment_batch_size']:
                batches.append(batch)
                batch = []
    
    if len(batch) > 0:
        batches.append(batch)
        batch = []

    # Feed each batch to the model
    list_patches_mask = []
    for batch_i in batches:
        
        # Convert from list of tensors to a tensor (Batch size, Channel, Width, Height)
        tensor_batch = torch.stack(batch_i)
        tensor_batch.to(device)

        # Feed it to model
        with torch.no_grad():
            tensor_segmentation_out = model(tensor_batch)
        
        # Convert masks from tensor to images
        for mask_i in tensor_segmentation_out:
            pil_mask_i = tensor_mask_to_pil_image(mask_i.squeeze(0).clone())
            list_patches_mask.append(pil_mask_i)

    # Put patches together
    whole_mask = Image.new('RGB', (config['dataset_image_size'][0], config['dataset_image_size'][1]), (255, 255, 255))
    for top_left_i, left_over_i, output_patch_i in zip(patches_top_left, list_patches_left_over, list_patches_mask):
        x, y = top_left_i
        # put mask patch on its true position in while image
        whole_mask.paste(output_patch_i.crop((left_over_i[0], left_over_i[1], config['deployment_patch_size']+left_over_i[2], config['deployment_patch_size']+left_over_i[3])), 
                         (x + left_over_i[0], y + left_over_i[1]))

    return whole_mask


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Segment images using a trained model.')
    parser.add_argument('--images_paths', type=str, nargs='+', required=True, help='Paths to image files')
    parser.add_argument('--out_dir_path', type=str, required=True, help='Path to save segmentation outputs')

    # Parse arguments
    args = parser.parse_args()

    # Path to read image
    images_paths = args.images_paths
    # Path to write output of segmentation into
    out_dir_path = args.out_dir_path

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the path to the JSON configuration file
    config_file_path = 'config/config.json'
    # Open and read the JSON file
    with open(config_file_path, 'r') as file:
        config = json.load(file)

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

    # For each image in input perform segmentation
    for img_path_i in images_paths:

        print("Processing {} ...".format(img_path_i))

        # Read i-th image
        image_i = Image.open(img_path_i)

        # Segment image and get segmentation mask in PIL image format
        segmentation_mask = segment_image(
                                config=config,
                                model=model, 
                                image=image_i, 
                                device=device, 
                                img_transformations=test_transformations)
        
        # Save image in output directory
        segmentation_mask.save(os.path.join(out_dir_path, os.path.basename(img_path_i))) 