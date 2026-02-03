"""Dataset loading and preprocessing utilities"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import os


class DatasetConfig:
    """Configuration for different datasets"""
    
    CIFAR10 = {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'input_size': 32,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    }
    
    CIFAR100 = {
        'name': 'CIFAR-100',
        'num_classes': 100,
        'input_size': 32,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    }
    
    IMAGENET = {
        'name': 'ImageNet',
        'num_classes': 1000,
        'input_size': 224,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }


def get_transforms(dataset_name: str, train: bool = True, input_size: Optional[int] = None):
    """
    Get appropriate transforms for dataset
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'imagenet')
        train: Whether training transforms or validation transforms
        input_size: Override default input size
    
    Returns:
        torchvision.transforms.Compose object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        config = DatasetConfig.CIFAR10
    elif dataset_name == 'cifar100':
        config = DatasetConfig.CIFAR100
    elif dataset_name == 'imagenet':
        config = DatasetConfig.IMAGENET
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    size = input_size or config['input_size']
    mean = config['mean']
    std = config['std']
    
    if train:
        if dataset_name in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            ])
        else:  # ImageNet
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        if dataset_name in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:  # ImageNet
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    return transform


def get_dataset(dataset_name: str, data_dir: str = './data', 
                train: bool = True, download: bool = True):
    """
    Load dataset
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store/load data
        train: Load training or test set
        download: Download if not available
    
    Returns:
        torch.utils.data.Dataset
    """
    dataset_name = dataset_name.lower()
    transform = get_transforms(dataset_name, train=train)
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=download, transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=train, download=download, transform=transform
        )
    elif dataset_name == 'imagenet':
        split = 'train' if train else 'val'
        dataset = torchvision.datasets.ImageNet(
            root=data_dir, split=split, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_dataloaders(dataset_name: str, data_dir: str = './data',
                   batch_size: int = 128, num_workers: int = 0,
                   download: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders
    
    Args:
        dataset_name: Name of dataset
        data_dir: Directory to store/load data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes (0 for Windows compatibility)
        download: Download if not available
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = get_dataset(dataset_name, data_dir, train=True, download=download)
    test_dataset = get_dataset(dataset_name, data_dir, train=False, download=download)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for dataset"""
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'imagenet':
        return 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing CIFAR-10 dataset loading...")
    train_loader, test_loader = get_dataloaders('cifar10', batch_size=4)
    
    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("Dataset loading successful!")
