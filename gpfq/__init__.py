"""GPFQ: A Greedy Algorithm for Quantizing Neural Networks

This package implements the GPFQ quantization algorithm for neural networks.
"""

from .models.cnn import SimpleCNN
from .models.mlp import MLP
from .models.vgg16 import VGG16
from .quantization.gpfq import QuantizedCNN, QuantizedNeuralNetwork

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    # Quantization
    "QuantizedNeuralNetwork",
    "QuantizedCNN",
    "MLP",
    "SimpleCNN",
    "VGG16",
]
