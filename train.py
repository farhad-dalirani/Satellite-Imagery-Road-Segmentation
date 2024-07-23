import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from loss import DiceBCELoss
from model import UnetLikeSegmentatorModel
from dataset import MRDDataset, JointTransform


def train_model(model, train_loader, val_loader, test_loader, num_epochs=25, lr=1e-4, checkpoint_path='saved_model/best_model.pth'):
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = DiceBCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # TensorBoard writer
    writer = SummaryWriter()

    # Track the best validation score
    best_val_loss = float('inf')

    # Train for maximum number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        batch_i = 0
        total_batch = len(train_loader)

        # For each batch in dataset
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update learning rate
            # scheduler.step()  
            
            running_loss += loss.item() * inputs.size(0)

            print(">    Epoch {}, Batch {}/{}, Average loss in batch: {}".format(epoch, batch_i, total_batch, loss.item()))
            batch_i += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved with loss: {best_val_loss:.4f}")

    # Test phase
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    writer.add_scalar('Test Loss', test_loss, num_epochs)
    print(f"Test Loss: {test_loss:.4f}")

    writer.close()


if __name__ == '__main__':
    
    # Data

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

    # Train dataset
    set_i = 'train'
    train_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=train_transformations)
    # Train dataloader
    dataloader_train = DataLoader(dataset=train_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_train)))

    # Validation dataset
    set_i = 'val'
    val_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=test_transformations)
    # Validation dataloader
    dataloader_val = DataLoader(dataset=val_ds, batch_size=config["train_batch_size"], shuffle=False, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_val)))

    # Test dataset
    set_i = 'test'
    test_ds = MRDDataset(
                    image_dir=os.path.join(config['data_dir'], '{}_patched'.format(set_i)), 
                    label_dir=os.path.join(config['data_dir'], '{}_labels_patched'.format(set_i)),
                    images_wh=tuple(config['dataset_image_size']),
                    transformas=test_transformations)
    # Test dataloader
    dataloader_test = DataLoader(dataset=test_ds, batch_size=config["train_batch_size"], shuffle=False, num_workers=2)
    print("Number of batches: {}".format(len(dataloader_test)))

    # Initialize the model
    model = UnetLikeSegmentatorModel()

    # Train the model
    train_model(model, dataloader_train, dataloader_val, dataloader_test, num_epochs=config['train_max_epoch'], lr=config['train_init_lr'], checkpoint_path=config['train_save_dir'])
