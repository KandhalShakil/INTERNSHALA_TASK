"""Training script for ResNet models"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from src.data.dataset import get_dataloaders, get_num_classes
from src.utils.metrics import AverageMeter, accuracy
from src.utils.config import get_config


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, writer=None):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc1': f'{top1.avg:.2f}',
            'acc5': f'{top5.avg:.2f}'
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc1', acc1.item(), global_step)
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
    
    return losses.avg, top1.avg, top5.avg


def train_model(model, train_loader, val_loader, config, device, save_dir):
    """
    Complete training loop
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    # Setup
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['milestones'], gamma=0.1
        )
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, writer
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        
        # Log to tensorboard
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/train_acc1', train_acc1, epoch)
        writer.add_scalar('epoch/val_loss', val_loss, epoch)
        writer.add_scalar('epoch/val_acc1', val_acc1, epoch)
        writer.add_scalar('epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc1:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc1:.2f}%")
        
        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'config': config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'last_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'best_checkpoint.pth'))
            print(f"✓ New best accuracy: {best_acc:.2f}%")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    return best_acc


if __name__ == "__main__":
    # Example usage
    from src.models.resnet_baseline import resnet50
    
    # Configuration
    config = {
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 100,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'cosine',
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        config['dataset'],
        batch_size=config['batch_size']
    )
    
    # Model
    num_classes = get_num_classes(config['dataset'])
    model = resnet50(num_classes=num_classes).to(device)
    
    # Train
    best_acc = train_model(
        model, train_loader, val_loader, config, device,
        save_dir='results/baseline_resnet50'
    )
