"""
Run baseline ResNet experiments
"""

import torch
import os
from src.models.resnet_baseline import resnet50
from src.data.dataset import get_dataloaders, get_num_classes
from src.training.train import train_model
from src.training.evaluate import evaluate_model, count_parameters, measure_inference_time
from src.utils.config import BASELINE_CONFIG


def main():
    print("="*80)
    print("BASELINE RESNET-50 EXPERIMENT")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    config = BASELINE_CONFIG.copy()
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Data
    print(f"\nLoading {config['dataset']} dataset...")
    train_loader, val_loader = get_dataloaders(
        config['dataset'],
        batch_size=config['batch_size']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    num_classes = get_num_classes(config['dataset'])
    model = resnet50(num_classes=num_classes).to(device)
    print(f"\nModel: ResNet-50")
    
    # Count parameters
    params = count_parameters(model)
    print(f"Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Train
    save_dir = 'results/baseline'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"Results will be saved to: {save_dir}")
    
    best_acc = train_model(
        model, train_loader, val_loader, config, device, save_dir
    )
    
    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    
    results = evaluate_model(model, val_loader, device, num_classes)
    
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    
    # Measure inference time
    input_size = (1, 3, 32, 32) if config['dataset'] in ['cifar10', 'cifar100'] else (1, 3, 224, 224)
    inf_time = measure_inference_time(model, input_size, device=str(device))
    print(f"Inference Time: {inf_time:.2f} ms")
    
    # Save final results
    final_results = {
        'model': 'ResNet-50 (Baseline)',
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'parameters': params['total'],
        'inference_time_ms': inf_time
    }
    
    import json
    with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\nResults saved to {save_dir}/final_results.json")
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
