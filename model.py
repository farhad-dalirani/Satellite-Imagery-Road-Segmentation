import json
import numpy as np
import math
import os
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from utils import tensor_image_to_pil_image, tensor_mask_to_pil_image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

class ModifiedResNet50(nn.Module):
    """
    Custom ResNet-50 architecture tailored for semantic segmentation tasks.

    This modified version of the standard ResNet-50 model includes the following adjustments:
    
    (1) Removal of the global average pooling and fully connected layers to adapt the model
        for feature extraction rather than classification.
    (2) Addition of intermediate layer outputs to provide a hierarchical feature map,
        allowing for better handling of features at multiple scales, which enhances
        the quality of segmentation results.

    The model utilizes a pre-trained ResNet-50 backbone, which is fine-tuned for the
    specific needs of semantic segmentation.
    """
    
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the global average pooling and fully connected layers
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])
        
        # Extract the stages from the ResNet-50 architecture
        self.stage0_conv2d = self.features[0]                 # Initial Conv Layer (stride 2)
        self.stage0_bn = self.features[1]              # Batch normalization of Initial Conv Layer
        self.stage0_statge_act = self.features[2]             # Activation of Initial Conv Layer
        self.stage0_statge_pooling = self.features[3]         # Max Pooling of Initial Conv Layer (stride 2)

        # Residual blocks
        self.stage1 = self.features[4]                        # Residual Block 1
        self.stage2 = self.features[5]                        # Residual Block 2
        self.stage3 = self.features[6]                        # Residual Block 3
        self.stage4 = self.features[7]                        # Residual Block 4

        
    def forward(self, x):
        
        x = self.stage0_conv2d(x)
        x = self.stage0_bn(x)
        x1 = self.stage0_statge_act(x)          # Output of Stage 0
        x_tmp = self.stage0_statge_pooling(x1)  

        x2 = self.stage1(x_tmp)                 # Output of Stage 1
        x3 = self.stage2(x2)                    # Output of Stage 2
        x4 = self.stage3(x3)                    # Output of Stage 3
        x5 = self.stage4(x4)                    # Output of Stage 4

        return x1, x2, x3, x4, x5

    


if __name__ == '__main__':

    # # Load a pre-trained ResNet-50 model
    # resnet50 = models.resnet50(pretrained=True)

    # print()
    # for id, m_i in enumerate(resnet50.children()):
    #     print('=' * 50)
    #     print("child index: {}".format(id))
    #     print(m_i)
    #     input()
    #     print('=' * 50)

    # Create an instance of the modified model
    model = ModifiedResNet50()

    # Example input tensor
    input_tensor = torch.randn(1, 3, 512, 512)

    # Forward pass through the model
    outputs = model(input_tensor)

    # Outputs from each stage
    stage1_output, stage2_output, stage3_output, stage4_output, stage5_output = outputs

    print(f"Stage 1 Output: {stage1_output.shape}")
    print(f"Stage 2 Output: {stage2_output.shape}")
    print(f"Stage 3 Output: {stage3_output.shape}")
    print(f"Stage 4 Output: {stage4_output.shape}")
    print(f"Stage 4 Output: {stage5_output.shape}")