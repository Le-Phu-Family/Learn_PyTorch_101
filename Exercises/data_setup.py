"""
Contains functions for downloading and loading data 
by DataLoaders for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str, test_dir: str,
 transform: transforms.Compose, batch_size: int = 32, num_workers: int = 0):
    """
    Creates training and test dataloaders.
    """
    # Load data from train and test directories
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn data into dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names
