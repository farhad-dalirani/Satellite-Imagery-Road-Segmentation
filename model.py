import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

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
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
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


class SCSEModule(nn.Module):
    """
    Squeeze and Channel/Spatial Excitation (SCSE) Module.

    This module applies both channel and spatial attention mechanisms to the input tensor.
    It is designed to enhance feature representation by recalibrating feature responses 
    using squeeze-and-excitation blocks for both channels and spatial dimensions.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Reduction ratio for the channel squeeze. Default is 16.

    Attributes:
        cSE (nn.Sequential): Channel Squeeze-and-Excitation block.
        sSE (nn.Sequential): Spatial Squeeze-and-Excitation block.

    Methods:
        forward(x):
            Forward pass of the module.

            Args:
                x (torch.Tensor): Input tensor of shape (N, C, H, W).

            Returns:
                torch.Tensor: Output tensor after applying SCSE attention.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        # Calculate attention score for each channel
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Calculate attention score for each pixel in feature map (spatial)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        channel_score = self.cSE(x)
        pixel_score = self.sSE(x)
        return x * channel_score + x * pixel_score


class ConvDecoderBlock(nn.Module):
    """
    Building block of a Convolutional Decoder for image segmentation or similar tasks.
    
    This block consists of two convolutional layers, each followed by a Batch Normalization layer
    and a ReLU activation function. The first convolutional layer also handles the concatenation 
    of skip connections from the encoder. This block can optionally be the last block in the decoder,
    in which case it omits the second Batch Normalization layer. Moreover, each block also has two 
    SCSE attention layers (spatial and channel attention).
    
    Attributes:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - skip_channel (int): Number of channels from the skip connection. Default is 0.
    - last_block (bool): Flag to indicate if this is the last block in the decoder. Default is False.
    """
    def __init__(self, in_channels, out_channels, skip_channel=0, last_block=False):
        super(ConvDecoderBlock, self).__init__()

        self.last_block=last_block

        self.conv1 = nn.Conv2d(in_channels=in_channels+skip_channel, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if skip_channel != 0:
            self.atten1 = SCSEModule(in_channels=in_channels+skip_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if self.last_block == False:
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.atten2 = SCSEModule(in_channels=out_channels)
        
        self.act = nn.ReLU()

    def forward(self, x, skip=None):
        
        # To increase the spatial resolution, interpolate the input feature map.
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        # Concatenate feature map from encoder
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.atten1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)

        x = self.conv2(x)
        if self.last_block == False:
            x = self.act(x)
            x = self.bn2(x)
            x = self.atten2(x)

        return x


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder consisting of a sequence of ConvDecoderBlock modules.
    
    This class constructs a decoder with a specified number of blocks, where each block can optionally 
    receive skip connections from an encoder. The last block in the decoder does not include a Batch 
    Normalization layer after the second convolutional layer.
    
    Attributes:
    - list_in_channels (list of int): List of input channels for each decoder block.
    - list_out_channels (list of int): List of output channels for each decoder block.
    - list_skip_channels (list of int): List of skip connection channels for each decoder block.
    - num_blocks (int): Number of decoder blocks. Default is 5.
    """

    def __init__(self, list_in_channels, list_out_channels, list_skip_channels, num_blocks=5):
        super(ConvDecoder, self).__init__()

        self.num_blocks = num_blocks

        self.neck = nn.Identity()

        # Create blocks of decoder
        list_blocks = [ConvDecoderBlock(
                            in_channels=list_in_channels[i], 
                            out_channels=list_out_channels[i], 
                            skip_channel=list_skip_channels[i], 
                            last_block=(i == (num_blocks-1))) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(list_blocks)

    def forward(self, features):
        
        x = self.neck(features[0])
        for i in range(self.num_blocks):
            if i != (self.num_blocks-1):
                x = self.blocks[i](x, features[i+1])
            else:
                x = self.blocks[i](x)
        
        return x


class UnetLikeSegmentatorModel(nn.Module):
    """
    A class to implement a U-Net like segmentation model.
    """

    def __init__(self):
        super(UnetLikeSegmentatorModel, self).__init__()

        # Encoder to extract features from input image at multiple level
        self.encoder = ModifiedResNet50()
        
        # Number of output channel at each stage of encoder from last stage to the first 
        self.encoder_featuremap_out_channels = [2048, 1024, 512, 256, 64]
        # Number of output channel of each block of decoder
        self.ecoder_each_block_out_channels = self.encoder_featuremap_out_channels[1::] + [1]
        # Number of channel of skip connection that come from encoder to each decoder block 
        self.ecoder_each_block_skip_connection_in_channels = self.encoder_featuremap_out_channels[1::] + [0]

        # Decoder to convert hierarchical feature maps from the encoder to a segmentation mask
        self.decoder = ConvDecoder(
                        list_in_channels=self.encoder_featuremap_out_channels, 
                        list_out_channels=self.ecoder_each_block_out_channels, 
                        list_skip_channels=self.ecoder_each_block_skip_connection_in_channels,
                        num_blocks=len(self.encoder_featuremap_out_channels))

    def forward(self, x):
        # Feed input to encoder
        encoder_out = self.encoder(x)
        # reverse output of encoder since last one processed first
        encoder_out = encoder_out[::-1]
        # Feed output of encoder to decoder to create segmentation mask
        decoder_out = self.decoder(encoder_out)

        return decoder_out


if __name__ == '__main__':


    # Create an instance of the modified model
    model = ModifiedResNet50()

    # Example input tensor
    input_tensor = torch.randn(1, 3, 512, 512)

    # Forward pass through the model
    outputs = model(input_tensor)

    # Outputs from each stage
    encoder_output = outputs
    

    print(f"Stage 1 Output: {encoder_output[0].shape}")
    print(f"Stage 2 Output: {encoder_output[1].shape}")
    print(f"Stage 3 Output: {encoder_output[2].shape}")
    print(f"Stage 4 Output: {encoder_output[3].shape}")
    print(f"Stage 4 Output: {encoder_output[4].shape}")


    dec_block = ConvDecoderBlock(in_channels=20, out_channels=10, skip_channel=5, last_block=True)
    input_tensor = torch.randn(1, 20, 16, 16)
    skip_tensor = torch.randn(1, 5, 32, 32)
    output = dec_block(input_tensor, skip_tensor)
    print("")
    print(f"Decoder Block Output: {output.shape}")

    list_in_channels = [2048, 1024, 512, 256, 64]
    list_out_channels = list_in_channels[1::] + [1]
    list_skip_channels = list_in_channels[1::] + [0]
    dec_model = ConvDecoder(list_in_channels=list_in_channels, list_out_channels=list_out_channels, list_skip_channels=list_skip_channels, num_blocks=5)

    encoder_output = encoder_output[::-1]
    dec_output = dec_model(encoder_output)

    print(f"Decoder Output: {dec_output.shape}")
    
    with torch.no_grad():
        input_tensor = torch.randn(3, 3, 512, 512)
        seg_model = UnetLikeSegmentatorModel()
        seg_out = seg_model(input_tensor)

    print(f"Segmentator Output: {seg_out.shape}")