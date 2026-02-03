#!/usr/bin/env python3
"""
Launcher script for experiments - handles PYTHONPATH automatically
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the experiment
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['baseline', 'improved', 'compare'],
                       help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    
    args = parser.parse_args()
    
    if args.experiment == 'baseline':
        # Temporarily set epochs in config
        from src.utils.config import BASELINE_CONFIG
        BASELINE_CONFIG['epochs'] = args.epochs
        
        # Import after path is set
        from experiments import run_baseline
        run_baseline.main()
        
    elif args.experiment == 'improved':
        from src.utils.config import HYBRID_RESNET_CONFIG
        HYBRID_RESNET_CONFIG['epochs'] = args.epochs
        
        from experiments import run_improved
        run_improved.main()
        
    elif args.experiment == 'compare':
        from experiments import compare_results
        compare_results.main()
