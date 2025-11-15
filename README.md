# GPFQ: A Greedy Algorithm for Quantizing Neural Networks

This repository implements the GPFQ (Greedy Post-training Quantization for Neural Networks) algorithm, a post-training quantization method that uses a greedy layer-wise approach to efficiently quantize neural networks while maintaining accuracy.

## Overview

GPFQ is a post-training quantization method that uses a greedy layer-wise approach to quantize neural networks. The algorithm iteratively selects the most important weights to quantize based on a second-order approximation of the loss function, enabling efficient quantization with minimal accuracy loss.

## Key Features

- **Multiple Model Architectures**: Support for MLP, CNN, and VGG16 models
- **Multiple Datasets**: MNIST, CIFAR-10, and ImageNet support
- **Flexible Quantization**: Various bit-widths (2-bit, 4-bit, 8-bit) with configurable parameters
- **Comprehensive Training Pipeline**: Full training, evaluation, and quantization workflow
- **Performance Analysis**: Model size reduction and inference timing comparisons
- **Configuration-based**: JSON configuration files for reproducible experiments

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- See `pyproject.toml` for complete dependencies

## Installation

### From Source
```bash
git clone <repository-url>
cd quantized_neural_network
pip install -e .
```

### Dependencies
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 tqdm>=4.64.0
```

## Quick Start

### Training and Quantization
```bash
# Train a CNN on CIFAR-10 and quantize it
python train.py --model cnn --dataset cifar10 --epochs 50 --quantize --quantize-bits 4

# Use configuration file
python train.py --config config/cnn_cifar10.json

# Quantization-only (requires pre-trained model)
python train.py --config config/cnn_cifar10.json --quantize-only
```

### Python API Usage
```python
from gpfq import QuantizedNeuralNetwork, SimpleCNN
from gpfq.dataset.cifar_dataset import CIFAR10Dataset
import torch

# Load dataset
dataset = CIFAR10Dataset(data_dir="./data", batch_size=128)
dataset.load_data()
train_loader, val_loader, test_loader = dataset.train_test_split()

# Create model
model = SimpleCNN(input_channels=3, num_classes=10)

# Quantize model
quantizer = QuantizedNeuralNetwork(
    model=model,
    dataloader=train_loader,
    bits=4,
    mini_batch_size=128
)
quantizer.quantize_network()
quantized_model = quantizer.quantized
```

## Project Structure

```
├── gpfq/                      # Main package
│   ├── quantization/          # Core quantization algorithms
│   │   └── gpfq.py           # GPFQ implementation
│   ├── models/               # Neural network models
│   │   ├── mlp.py           # Multi-layer Perceptron
│   │   ├── cnn.py           # Convolutional Neural Network
│   │   └── vgg16.py         # VGG16 architecture
│   ├── dataset/             # Dataset loaders
│   │   ├── mnist_dataset.py # MNIST dataset handler
│   │   ├── cifar_dataset.py # CIFAR-10 dataset handler
│   │   └── imagenet_dataset.py # ImageNet dataset handler
│   └── utils/               # Utility functions
│       └── utils.py         # Evaluation and utility functions
├── config/                  # Configuration files
│   ├── mlp_mnist.json      # MLP on MNIST config
│   ├── cnn_cifar10.json    # CNN on CIFAR-10 config
│   └── vgg16_imagenet.json # VGG16 on ImageNet config
├── data/                   # Dataset storage
├── train.py               # Main training script
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Supported Configurations

### Models
- **MLP**: Multi-layer Perceptron for MNIST
- **SimpleCNN**: Convolutional Neural Network for CIFAR-10
- **VGG16**: VGG16 architecture for ImageNet

### Datasets
- **MNIST**: 28x28 grayscale handwritten digits
- **CIFAR-10**: 32x32 color images, 10 classes
- **ImageNet**: High-resolution images, 1000 classes

### Quantization Options
- **Bits**: 2, 4, 8-bit quantization
- **Calibration**: Configurable number of calibration samples
- **Layer Types**: Support for both linear and convolutional layers

## Training Options

### Basic Training
```bash
python train.py --model cnn --dataset cifar10 --epochs 100 --lr 0.001
```

### Advanced Options
```bash
python train.py \
  --model vgg16 \
  --dataset imagenet \
  --epochs 200 \
  --batch-size 256 \
  --optimizer sgd \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --use-batch-norm \
  --dropout 0.5
```

### Resume Training
```bash
# Resume from checkpoint
python train.py --model cnn --dataset cifar10 --resume logs/cnn_cifar10/latest_checkpoint.pt

# Auto-resume from latest checkpoint
python train.py --model cnn --dataset cifar10 --auto-resume
```

### Quantization
```bash
# Post-training quantization
python train.py --model cnn --dataset cifar10 --quantize --quantize-bits 4

# Quantization-only (no training)
python train.py --model cnn --dataset cifar10 --quantize-only --quantize-bits 4
```

## Results and Logging

The training script provides comprehensive logging and evaluation:

- **TensorBoard logs**: Training/validation curves, model graphs
- **Model checkpoints**: Best model and latest checkpoint saving
- **Performance metrics**: Accuracy, loss, training time
- **Quantization analysis**: Model size reduction, accuracy drop, inference speedup
- **JSON results**: Complete experiment results and configuration

Logs are saved to `logs/MODEL_DATASET/` directory.

## Configuration Files

Use JSON configuration files for reproducible experiments:

```json
{
  "model": "cnn",
  "dataset": "cifar10",
  "epochs": 100,
  "batch_size": 128,
  "lr": 0.001,
  "optimizer": "adam",
  "quantize": true,
  "quantize_bits": 4,
  "calibration_samples": 1024
}
```

## Performance Analysis

The framework provides detailed performance analysis:

- **Model Size**: Original vs. quantized model size comparison
- **Accuracy**: Validation and test accuracy before/after quantization
- **Inference Time**: Speed comparison between original and quantized models
- **Memory Usage**: Memory footprint analysis

## Development

### Code Quality
The project uses Ruff for linting and formatting:

```bash
pip install ruff
ruff check .
ruff format .
```

## License

MIT License - see LICENSE file for details.