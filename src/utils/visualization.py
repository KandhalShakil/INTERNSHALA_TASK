"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_dict, save_path=None):
    """
    Plot comparison of different models
    
    Args:
        results_dict: Dict of {model_name: {'top1': acc, 'top5': acc, 'params': num}}
    """
    models = list(results_dict.keys())
    top1_accs = [results_dict[m]['top1'] for m in models]
    top5_accs = [results_dict[m]['top5'] for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, top5_accs, width, label='Top-5 Accuracy',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([85, 100])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_maps(image, attention_map, save_path=None):
    """Visualize attention maps"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:  # CHW to HWC
        image = np.transpose(image, (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min())
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention map
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    if len(attention_map.shape) == 3:
        attention_map = attention_map[0]
    ax2.imshow(attention_map, cmap='hot')
    ax2.set_title('Attention Map')
    ax2.axis('off')
    
    # Overlay
    ax3.imshow(image)
    ax3.imshow(attention_map, cmap='hot', alpha=0.5)
    ax3.set_title('Attention Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(results_dict, save_path=None):
    """Create and save results comparison table"""
    import pandas as pd
    
    df = pd.DataFrame(results_dict).T
    df = df.round(2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(results_dict) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     rowLabels=df.index, cellLoc='center', 
                     loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style row labels
    for i in range(len(df.index)):
        table[(i+1, -1)].set_facecolor('#ecf0f1')
        table[(i+1, -1)].set_text_props(weight='bold')
    
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    if save_path:
        csv_path = Path(save_path).with_suffix('.csv')
        df.to_csv(csv_path)


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    
    # Test plotting
    train_losses = [2.3, 1.8, 1.5, 1.2, 1.0]
    val_losses = [2.4, 1.9, 1.6, 1.4, 1.2]
    train_accs = [20, 40, 60, 75, 85]
    val_accs = [18, 38, 58, 70, 80]
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 'test_curves.png')
    print("Test training curves saved!")
