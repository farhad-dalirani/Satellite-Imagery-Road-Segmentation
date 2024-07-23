import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Combination of Dice Loss plus Binary Cross-entropy for binary segmentation tasks.
    
    Dice Loss is used to handle class imbalance and measures the overlap
    between the predicted and ground truth masks. It is particularly useful
    when the positive class is rare or when you have imbalanced classes.
    
    Args:
        smooth (float): A smoothing factor to avoid division by zero. Default is 1e-6.
    """

    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)  
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate dice liss
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice_score = 1.0 - ((2. * intersection + self.smooth) / (union + self.smooth))

        # Calculate Binary Cross Entropy Loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # BCE-Dice loss
        Dice_BCE_loss = BCE_loss + dice_score

        return Dice_BCE_loss


if __name__ == '__main__':
    
    # Example input tensor

    dicebce_loss = DiceBCELoss()

    outputs_tensor = torch.randn(1, 1, 512, 512)
    targets_tensor = torch.randint(low=0, high=1, size=(1, 1, 512, 512)).float()
    loss = dicebce_loss(outputs_tensor, targets_tensor)
    print(loss)

    outputs_tensor = torch.tensor([[[-100,-100,-100], [-100,100,-100], [-100,-100,-100]]])
    targets_tensor = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, .0]]])
    
    loss = dicebce_loss(outputs_tensor, targets_tensor)
    print(loss)
