# Robust Residual Finite Scalar Quantization for Neural Compression

This repository contains the official implementation of **Robust Residual Finite Scalar Quantization (RFSQ)**, a novel quantization framework that addresses the residual magnitude decay problem in naive residual FSQ implementations.

## Abstract

Finite Scalar Quantization (FSQ) has emerged as a promising alternative to Vector Quantization (VQ) in neural compression. However, naive application of FSQ in residual quantization frameworks suffers from the **residual magnitude decay problem**, where subsequent FSQ layers receive progressively weaker signals. We propose **Robust Residual Finite Scalar Quantization (RFSQ)** with two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our approach achieves up to 45% improvement in perceptual loss and 28.7% reduction in L1 reconstruction error compared to strong baselines.

## Key Features

- **Robust Framework**: Addresses residual magnitude decay through novel conditioning strategies
- **Superior Performance**: Significant improvements over VQ-EMA, FSQ, and LFQ baselines
- **Two Strategies**: Learnable scaling and invertible LayerNorm for different use cases
- **Plug-and-Play**: Compatible with any encoder-decoder architecture
- **Comprehensive Evaluation**: Extensive experiments with fair comparison protocols

### Conditioning Strategies

1. **Learnable Scaling**: Adaptive scaling factors that amplify residual signals
2. **Invertible LayerNorm**: Feature normalization while maintaining perfect reconstruction
3. **None**: Vanilla RFSQ baseline for comparison

## Installation

```bash
git clone https://github.com/zhuxiaoxuhit/Robust-Residual-Finite-Scalar-Quantization-for-Neural-Compression.git
cd Robust-Residual-Finite-Scalar-Quantization-for-Neural-Compression
pip install torch torchvision torchaudio
pip install lpips
```

## Quick Start

### Basic Usage

```python
from quantizers import RFSQ

# Initialize RFSQ with LayerNorm strategy
rfsq = RFSQ(
    levels=[8, 8, 8, 4],      # FSQ levels for each dimension
    stages=2,                  # Number of residual stages
    strategy='layernorm'       # Conditioning strategy
)

# Quantize features
quantized, indices = rfsq(features)
```

### Training Example

```python
from model import VQVAE
from arguments import get_args
import train

# Parse arguments
args = get_args()

# Create model with RFSQ
model = VQVAE(args)

# Train the model
train.main()
```

## Experimental Configurations

We provide nine pre-configured experimental setups with identical 12.0-bit code rates:

| Configuration | Strategy | Stages | Levels | Codebook Size |
|---------------|----------|--------|--------|---------------|
| VQ-EMA | - | - | - | 4096 |
| FSQ | - | 1 | [8,8,8,8] | 4096 |
| LFQ | - | 1 | - | 4096 |
| RFSQ-2×2048-Scale | Scale | 2 | [8,8,8,4] | 4096 |
| RFSQ-2×2048-LayerNorm | LayerNorm | 2 | [8,8,8,4] | 4096 |
| RFSQ-2×2048-None | None | 2 | [8,8,8,4] | 4096 |
| RFSQ-4×1024-Scale | Scale | 4 | [4,4,4,4,4] | 4096 |
| RFSQ-4×1024-LayerNorm | LayerNorm | 4 | [4,4,4,4,4] | 4096 |
| RFSQ-4×1024-None | None | 4 | [4,4,4,4,4] | 4096 |

## Results

### Performance Comparison (ImageNet 128×128)

| Method | L1 Loss | Perceptual Loss | PSNR (dB) |
|--------|---------|-----------------|-----------|
| **RFSQ-4×1024-LayerNorm** | **0.102** | **0.100** | **22.9** |
| RFSQ-4×1024-Scale | 0.103 | 0.101 | 22.9 |
| RFSQ-2×2048-Scale | 0.122 | 0.152 | 21.5 |
| FSQ (baseline) | 0.143 | 0.182 | 20.3 |
| LFQ | 0.241 | 0.361 | 16.0 |
| VQ-EMA | 0.355 | 0.489 | 12.7 |

### Key Improvements

- **45.1%** improvement in perceptual loss over FSQ
- **28.7%** reduction in L1 reconstruction error
- **12.8%** improvement in PSNR
- **Consistent performance** across all evaluation metrics

## Repository Structure

```
├── quantizers/           # Quantization implementations
│   ├── __init__.py
│   ├── fsq.py           # Finite Scalar Quantization
│   ├── rfsq.py          # Robust Residual FSQ (our method)
│   ├── lfq.py           # Lookup-Free Quantization
│   └── ema.py           # VQ with Exponential Moving Average
├── model.py             # Encoder-Decoder architecture
├── train.py             # Training loop
├── arguments.py         # Command-line arguments
├── dataset.py           # Data loading utilities
├── loss/                # Loss functions
├── util.py              # Utility functions
├── metric.py            # Evaluation metrics
├── scheduler.py         # Learning rate scheduling
└── lpips.py             # LPIPS perceptual loss
```

## Training

### Single Configuration

```bash
python train.py \
    --quantizer rfsq \
    --levels 8 8 8 4 \
    --rfsq-stages 2 \
    --rfsq-strategy layernorm \
    --batch-size 128 \
    --lr 0.0008 \
    --max-train-epochs 50
```

### Reproduce Paper Results

All experimental configurations from the paper can be reproduced using the provided training scripts and parameters.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhu2025rfsq,
  title={Robust Residual Finite Scalar Quantization for Neural Compression},
  author={Zhu, Xiaoxu},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

We acknowledge the open-source implementations that facilitated this research:
- FSQ and LFQ implementations adapted from [FSQ-pytorch](https://github.com/duchenzhuang/FSQ-pytorch)
- LPIPS implementation for perceptual loss evaluation

## Contact

For questions or issues, please contact: zhuxx23@mails.tsinghua.edu.cn 
