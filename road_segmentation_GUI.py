import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import UnetLikeSegmentatorModel
from dataset import JointTransform
from segment_images import segment_image
import streamlit as st

# Set page layout to wide
st.set_page_config(layout="wide")

# Main function
def main():
    """
    Streamlit application for Road Segmentation on Satellite Imagery.

    This application allows users to upload a satellite image and perform road segmentation
    using a U-Net-like deep neural network. The model utilizes a modified ResNet-50 as the encoder
    and a CNN with SCSE attention decoder head. The segmentation task addresses binary imbalance using a combined 
    Dice and Binary Cross-entropy loss function. Large satellite images are patchified during 
    training and deployment.

    The model is trained on the Massachusetts Roads Dataset to identify road structures in 
    high-resolution images, providing a segmented output that highlights the road network.

    Functions:
        main():
            The main function to run the Streamlit application. It contains the following components:
            
            - Title: Displays the title of the application.
            - Sidebar: Provides a project description and details about the model architecture and training.
            - File Uploader: Allows users to upload a satellite image for segmentation.
            - Buttons:
                - Segment: Segments the uploaded image and displays the segmentation mask.
                - Save Output: Saves the segmented image to the specified path.
            - Image Display: Displays the uploaded image and the segmentation mask.

    Usage:
        Run this script using Streamlit to start the application:
        $ streamlit run road_segmentation_GUI.py
    """

    st.title("Road Segmentation on Satellite Imagery")

    # Sidebar for project description
    st.sidebar.write("""
        ### Road Semantic Segmentation from Satellite Imagery

        This project involves:

        - **U-Net-like** deep neural network architecture for semantic segmentation.
        - **Encoder**: Pre-trained **ResNet-50**, modified to output each stage's hierarchical feature maps.
        - **Decoder**: Consists of interpolation, convolutional layers, and **SCSE attention**.
        - **Loss**: Combined **Dice and Binary Cross-entropy loss** for the binary imbalance segmentation task.
        - **Patchifying input images** during training and deployment to handle very large satellite images.     

        The model is trained on the **Massachusetts Roads Dataset** to identify road structures in high-resolution images, providing a segmented output that highlights the road network.
        """)

    # Select a file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff", "tif"])

    if uploaded_file is not None:
        # Store the uploaded image in session state
        st.session_state['input_image'] = Image.open(uploaded_file)

        # Container for buttons
        button_container = st.container()

        # Row for buttons
        with button_container:
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Segment', use_container_width=True):
                    # Segment image and get segmentation mask in PIL image format
                    st.session_state['segmentation_mask'] = segment_image(
                        config=st.session_state['config'],
                        model=st.session_state['segmentation_model'], 
                        image=st.session_state['input_image'], 
                        device=st.session_state['device'], 
                        img_transformations=st.session_state['test_transformations'])

            with col2:
                save_path = st.text_input("Enter save path:", value="segmented_image.png")
                if st.button('Save Output', use_container_width=True):
                    if save_path:
                        # Create the directory if it does not exist
                        directory = os.path.dirname(save_path)
                        if (len(directory) != 0) and (not os.path.exists(directory)):
                            os.makedirs(directory)

                        st.session_state['segmentation_mask'].save(save_path)
                        st.success(f"Image saved to {save_path}")

        # Container for images
        image_container = st.container()

        # Row for images
        with image_container:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state['input_image'], caption='Uploaded Image.', use_column_width=True)

            if 'segmentation_mask' in st.session_state:
                with col2:
                    st.image(st.session_state['segmentation_mask'], caption='Segmented Image.', use_column_width=True)


if __name__ == '__main__':

    # Define device
    if 'device' not in st.session_state.keys():    
        st.session_state['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration
    if 'config' not in st.session_state.keys():
        # Define the path to the JSON configuration file
        config_file_path = 'config/config.json'
        # Open and read the JSON file
        with open(config_file_path, 'r') as file:
            st.session_state['config'] = json.load(file)

    # Transformation for converting PIL input images to tensor suitable for deep network model
    if 'test_transformations' not in st.session_state.keys():
        joint_transform_test = transforms.Compose([transforms.ToTensor()])
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        st.session_state['test_transformations'] = JointTransform(joint_transform=joint_transform_test, image_transform=image_transform)

    if 'segmentation_model' not in st.session_state.keys():
        # Load segmentation model weights
        model = UnetLikeSegmentatorModel()
        model.to(device=st.session_state['device'])
        model.load_state_dict(torch.load(st.session_state['config']["train_save_dir"], map_location=st.session_state['device']))
        model.eval()
        st.session_state['segmentation_model'] = model

    # Open GUI
    main()