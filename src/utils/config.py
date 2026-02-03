"""Configuration management"""

import yaml
import os
from easydict import EasyDict


def get_config(config_path=None):
    """
    Get configuration
    
    Args:
        config_path: Path to config file (yaml)
    
    Returns:
        Configuration dictionary
    """
    # Default configuration
    config = {
        'dataset': 'cifar10',
        'batch_size': 128,
        'num_workers': 0,
        'epochs': 100,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'cosine',
        'milestones': [30, 60, 90],
        'seed': 42,
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        config.update(file_config)
    
    return EasyDict(config)


def save_config(config, save_path):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(dict(config), f, default_flow_style=False)


# Default training configurations for different models
BASELINE_CONFIG = {
    'model': 'resnet50',
    'dataset': 'cifar10',
    'batch_size': 128,
    'num_workers': 0,
    'epochs': 100,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'scheduler': 'cosine',
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
}

SE_RESNET_CONFIG = {
    'model': 'se_resnet50',
    'dataset': 'cifar10',
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'reduction': 16,
}

SPATIAL_RESNET_CONFIG = {
    'model': 'spatial_resnet50',
    'dataset': 'cifar10',
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
}

HYBRID_RESNET_CONFIG = {
    'model': 'hybrid_resnet50',
    'dataset': 'cifar10',
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'reduction': 16,
}


if __name__ == "__main__":
    config = get_config()
    print("Default configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
