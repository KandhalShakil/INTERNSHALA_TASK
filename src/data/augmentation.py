"""Data augmentation strategies"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import numpy as np


class Cutout:
    """Randomly mask out one or more patches from an image."""
    
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MixUp:
    """
    MixUp data augmentation
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2017)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, labels):
        """
        Args:
            batch: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        
        Returns:
            Mixed batch and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)

        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_batch, labels_a, labels_b, lam


class CutMix:
    """
    CutMix data augmentation
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, labels):
        """
        Args:
            batch: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
        
        Returns:
            Mixed batch and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)

        _, _, h, w = batch.size()
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        batch[:, :, y1:y2, x1:x2] = batch[index, :, y1:y2, x1:x2]
        
        # Adjust lambda to match pixel ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        
        labels_a, labels_b = labels, labels[index]
        
        return batch, labels_a, labels_b, lam


def get_augmentation_policy(policy='standard'):
    """
    Get data augmentation policy
    
    Args:
        policy: 'standard', 'autoaugment', or 'cutout'
    
    Returns:
        Transform pipeline
    """
    if policy == 'standard':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif policy == 'cutout':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
        ])
    elif policy == 'autoaugment':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unknown policy: {policy}")


if __name__ == "__main__":
    # Test augmentation
    print("Testing data augmentation...")
    
    # Test Cutout
    cutout = Cutout(n_holes=1, length=16)
    img = torch.randn(3, 32, 32)
    augmented = cutout(img)
    print(f"Cutout shape: {augmented.shape}")
    
    # Test MixUp
    mixup = MixUp(alpha=1.0)
    batch = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    mixed_batch, labels_a, labels_b, lam = mixup(batch, labels)
    print(f"MixUp batch shape: {mixed_batch.shape}, lambda: {lam:.4f}")
    
    # Test CutMix
    cutmix = CutMix(alpha=1.0)
    mixed_batch, labels_a, labels_b, lam = cutmix(batch, labels)
    print(f"CutMix batch shape: {mixed_batch.shape}, lambda: {lam:.4f}")
    
    print("Augmentation tests passed!")
