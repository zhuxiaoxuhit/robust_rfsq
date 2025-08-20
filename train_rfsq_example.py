#!/usr/bin/env python
"""
Example training script for RFSQ-4×1024-LayerNorm configuration
Demonstrates the best-performing RFSQ variant from our paper
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Override command line arguments for this configuration
sys.argv = [
    'train_rfsq_example.py',
    
    # Model configuration
    '--quantizer', 'rfsq',
    '--levels', '4', '4', '4', '4', '4',  # 4-stage configuration
    '--rfsq-stages', '4',
    '--rfsq-strategy', 'layernorm',  # Best performing strategy
    '--channel', '256',  # Unified architecture
    
    # Training configuration  
    '--batch-size', '128',
    '--lr', '0.0008',
    '--max-train-epochs', '50',
    '--save-interval', '500',
    '--weight-decay', '5e-05',
    '--clip-grad', '1.0',
    '--warmup', '0.01',
    
    # Loss configuration
    '--l1-weight', '1.0',
    '--perceptual-weight', '1.0',
    '--codebook-weight', '1.0',
    
    # Data configuration (modify these paths according to your setup)
    '--train-data-path', './data/imagenet/train',
    '--val-data-path', './data/imagenet/val',
    '--img-size', '128',
    '--num-workers', '4',
    
    # Other settings
    '--seed', '1234',
    '--experiment-name', 'RFSQ-4x1024-LayerNorm-Example'
]

# Import and run training
import train

if __name__ == '__main__':
    print("🚀 Starting RFSQ-4×1024-LayerNorm training example...")
    print("📊 Configuration: 4-stage RFSQ with LayerNorm strategy")
    print("🎯 Target: 12.0 bits code rate, 4096 codebook size")
    print("=" * 60)
    
    train.main() 