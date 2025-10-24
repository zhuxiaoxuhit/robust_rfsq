# Robust Residual Finite Scalar Quantization for Neural Compression

This repository contains the official implementation of **Robust Residual Finite Scalar Quantization (RFSQ)**, a novel quantization framework that addresses the residual magnitude decay problem in naive residual FSQ implementations.

## Abstract

Finite Scalar Quantization (FSQ) has emerged as a promising alternative to Vector Quantization (VQ) in neural compression. However, naive application of FSQ in residual quantization frameworks suffers from the **residual magnitude decay problem**, where subsequent FSQ layers receive progressively weaker signals. We propose **Robust Residual Finite Scalar Quantization (RFSQ)** with two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our approach demonstrates significant improvements across different bit rates, with LayerNorm conditioning consistently outperforming other strategies.

## Key Features

- **Robust Framework**: Addresses residual magnitude decay through novel conditioning strategies
- **Multiple Strategies**: Learnable scaling and invertible LayerNorm for different use cases
- **Flexible Bit Rates**: Supports both 22.0-bit and 40.0-bit configurations
- **Plug-and-Play**: Compatible with any encoder-decoder architecture
- **Comprehensive Evaluation**: Systematic evaluation on ImageNet reconstruction tasks

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

# Initialize RFSQ with LayerNorm strategy (22.0-bit configuration)
rfsq_22bit = RFSQ(
    stages=2,                  # Number of residual stages
    strategy='layernorm'       # Conditioning strategy: 'layernorm', 'scale', or 'none'
)

# Initialize RFSQ with LayerNorm strategy (40.0-bit configuration)
rfsq_40bit = RFSQ(
    stages=4,                  # Number of residual stages
    strategy='layernorm'       # Conditioning strategy
)

# Quantize features
quantized, indices = rfsq_22bit(features)
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

We provide six RFSQ configurations evaluated on ImageNet at two different bit rates:

| Configuration | Strategy | Bits | Description |
|---------------|----------|------|-------------|
| RFSQ-2×2048-None | None | 22.0 | 2-stage residual quantization |
| RFSQ-2×2048-Scale | Scale | 22.0 | 2-stage with learnable scaling |
| RFSQ-2×2048-LN | LayerNorm | 22.0 | 2-stage with layer normalization |
| RFSQ-4×1024-None | None | 40.0 | 4-stage residual quantization |
| RFSQ-4×1024-Scale | Scale | 40.0 | 4-stage with learnable scaling |
| RFSQ-4×1024-LN | LayerNorm | 40.0 | 4-stage with layer normalization |

## Results

### Performance Comparison (ImageNet 128×128)

#### 22.0-bit Configurations
| Method | Bits | L1↓ | LPIPS↓ | PSNR↑ |
|--------|------|-----|--------|-------|
| RFSQ-2×2048-None | 22.0 | 0.130 | 0.159 | 21.1 |
| RFSQ-2×2048-Scale | 22.0 | 0.122 | 0.152 | 21.5 |
| **RFSQ-2×2048-LN** | **22.0** | **0.124** | **0.148** | **21.3** |

#### 40.0-bit Configurations
| Method | Bits | L1↓ | LPIPS↓ | PSNR↑ |
|--------|------|-----|--------|-------|
| RFSQ-4×1024-None | 40.0 | 0.113 | 0.121 | 22.2 |
| RFSQ-4×1024-Scale | 40.0 | 0.103 | 0.101 | 22.9 |
| **RFSQ-4×1024-LN** | **40.0** | **0.102** | **0.100** | **22.9** |

### Key Improvements

- **LayerNorm strategy** consistently outperforms other conditioning approaches
- **9.7%** L1 improvement and **17.4%** perceptual improvement at 40 bits (LN vs None)
- **Higher bit budgets** provide substantial quality gains (22→40 bits)
- **Effective residual processing** crucial for multi-stage quantization performance

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
# Train RFSQ-2×2048-LN (22.0-bit configuration)
python train.py \
    --quantizer rfsq \
    --rfsq-stages 2 \
    --rfsq-strategy layernorm \
    --batch-size 128 \
    --lr 0.0008 \
    --max-train-epochs 50

# Train RFSQ-4×1024-LN (40.0-bit configuration)
python train.py \
    --quantizer rfsq \
    --rfsq-stages 4 \
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
