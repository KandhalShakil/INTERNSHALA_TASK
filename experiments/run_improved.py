"""
Run improved ResNet with Hybrid Attention experiments
"""

import torch
import os
from src.models.resnet_hybrid import hybrid_resnet50
from src.models.resnet_se import se_resnet50
from src.models.resnet_spatial import spatial_resnet50
from src.data.dataset import get_dataloaders, get_num_classes
from src.training.train import train_model
from src.training.evaluate import evaluate_model, count_parameters, measure_inference_time
from src.utils.config import HYBRID_RESNET_CONFIG


def main():
    print("="*80)
    print("IMPROVED RESNET WITH HYBRID ATTENTION EXPERIMENT")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Set seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    config = HYBRID_RESNET_CONFIG.copy()
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Data
    print(f"\nLoading {config['dataset']} dataset...")
    train_loader, val_loader = get_dataloaders(
        config['dataset'],
        batch_size=config['batch_size']
    )
    
    # Models to test
    models_to_test = [
        ('SE-ResNet-50', se_resnet50),
        ('Spatial-ResNet-50', spatial_resnet50),
        ('Hybrid-ResNet-50 (PROPOSED)', hybrid_resnet50),
    ]
    
    all_results = {}
    
    for model_name, model_fn in models_to_test:
        print(f"\n{'='*80}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*80}")
        
        # Create model
        num_classes = get_num_classes(config['dataset'])
        model = model_fn(num_classes=num_classes, reduction=config.get('reduction', 16)).to(device)
        
        # Count parameters
        params = count_parameters(model)
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        
        # Train
        save_dir = f"results/{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nStarting training...")
        print(f"Results will be saved to: {save_dir}")
        
        best_acc = train_model(
            model, train_loader, val_loader, config, device, save_dir
        )
        
        # Load best model
        checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        print(f"\nFinal Evaluation...")
        results = evaluate_model(model, val_loader, device, num_classes)
        
        print(f"\nTest Loss: {results['loss']:.4f}")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        
        # Measure inference time
        input_size = (1, 3, 32, 32) if config['dataset'] in ['cifar10', 'cifar100'] else (1, 3, 224, 224)
        inf_time = measure_inference_time(model, input_size, device=str(device))
        print(f"Inference Time: {inf_time:.2f} ms")
        
        # Store results
        all_results[model_name] = {
            'top1': results['top1_accuracy'],
            'top5': results['top5_accuracy'],
            'params': params['total'],
            'inference_time': inf_time
        }
        
        # Save individual results
        import json
        with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
            json.dump(all_results[model_name], f, indent=4)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY - ALL MODELS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<35} {'Top-1':<10} {'Top-5':<10} {'Params':<12} {'Time (ms)'}")
    print("-" * 80)
    for model_name, res in all_results.items():
        print(f"{model_name:<35} {res['top1']:>6.2f}%  {res['top5']:>6.2f}%  {res['params']:>10,}  {res['inference_time']:>8.2f}")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    
    # Save summary
    import json
    with open('results/all_results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
