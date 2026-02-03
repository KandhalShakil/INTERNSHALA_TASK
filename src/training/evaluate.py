"""Evaluation utilities and metrics"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, test_loader, device, num_classes=10):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        num_classes: Number of classes
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.topk(5, 1, True, True)
            
            # Top-1 accuracy
            correct_top1 += (predicted[:, 0] == targets).sum().item()
            
            # Top-5 accuracy
            correct_top5 += predicted.eq(targets.view(-1, 1).expand_as(predicted)).sum().item()
            
            total += targets.size(0)
            
            # Store for confusion matrix
            all_preds.extend(predicted[:, 0].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    results = {
        'loss': avg_loss,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'confusion_matrix': cm,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probs)
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(cm, class_names, save_path=None):
    """Plot per-class accuracy"""
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(class_names)), per_class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return per_class_acc


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def measure_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_runs=100):
    """Measure average inference time"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    
    import time
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_runs * 1000  # Convert to ms
    
    return avg_time


def print_evaluation_summary(results, model_name="Model"):
    """Print comprehensive evaluation summary"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    
    if 'parameters' in results:
        print(f"\nModel Parameters:")
        print(f"  Total: {results['parameters']['total']:,}")
        print(f"  Trainable: {results['parameters']['trainable']:,}")
    
    if 'inference_time' in results:
        print(f"\nInference Time: {results['inference_time']:.2f} ms")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities loaded successfully!")
    
    # Test confusion matrix plotting
    cm = np.random.randint(0, 100, (10, 10))
    class_names = [f'Class_{i}' for i in range(10)]
    plot_confusion_matrix(cm, class_names, 'test_cm.png')
    print("Test confusion matrix saved to test_cm.png")
