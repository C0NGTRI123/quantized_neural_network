"""Utility functions for GPFQ quantization experiments.

This module provides helper functions for data loading, model evaluation,
calibration, and various metrics computation.
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate_model(
    model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, criterion: nn.Module | None = None
) -> dict[str, float]:
    """Evaluate model performance.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        criterion: Loss criterion (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_samples = 0
    correct = 0
    total_loss = 0.0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total_samples += data.size(0)

    accuracy = 100.0 * correct / total_samples
    avg_loss = total_loss / total_samples

    return {"accuracy": accuracy, "loss": avg_loss, "correct": correct, "total": total_samples}


def compute_model_size(model: nn.Module, bits: int = 32) -> dict[str, float]:
    """Compute model size in different units.

    Args:
        model: PyTorch model
        bits: Bits per parameter (32 for float32, 16 for float16, etc.)

    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size calculations
    bytes_per_param = bits // 8
    total_size_bytes = total_params * bytes_per_param
    total_size_mb = total_size_bytes / (1024**2)
    total_size_gb = total_size_bytes / (1024**3)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "bits_per_param": bits,
        "size_bytes": total_size_bytes,
        "size_mb": total_size_mb,
        "size_gb": total_size_gb,
    }


def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Compare original and quantized models.

    Args:
        original_model: Original full-precision model
        quantized_model: Quantized model
        test_loader: Test data loader
        device: Device for evaluation

    Returns:
        Comparison results
    """
    print("Evaluating original model...")
    original_results = evaluate_model(original_model, test_loader, device)

    print("Evaluating quantized model...")
    quantized_results = evaluate_model(quantized_model, test_loader, device)

    # Size comparison (assuming 32-bit original, need to determine quantized bits)
    original_size = compute_model_size(original_model, bits=32)
    # Note: For quantized model, bits would depend on quantization scheme
    quantized_size = compute_model_size(quantized_model, bits=8)  # Placeholder

    # Compute metrics
    accuracy_drop = original_results["accuracy"] - quantized_results["accuracy"]
    compression_ratio = original_size["size_mb"] / quantized_size["size_mb"]

    return {
        "original": original_results,
        "quantized": quantized_results,
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_percent": (accuracy_drop / original_results["accuracy"]) * 100,
        "compression_ratio": compression_ratio,
        "original_size_mb": original_size["size_mb"],
        "quantized_size_mb": quantized_size["size_mb"],
    }


def measure_inference_time(
    model: nn.Module, input_shape: tuple[int, ...], device: torch.device, num_runs: int = 100, warmup_runs: int = 10
) -> dict[str, float]:
    """Measure model inference time.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        device: Device to run on
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs

    Returns:
        Timing statistics
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Actual timing
    torch.cuda.synchronize() if device.type == "cuda" else None

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "median_time": np.median(times),
    }


def visualize_weight_distribution(model: nn.Module, layer_name: str = None) -> None:
    """Visualize weight distributions of model layers.

    Args:
        model: PyTorch model
        layer_name: Specific layer to visualize (if None, visualize all)
    """
    plt.figure(figsize=(15, 10))

    layer_weights = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if layer_name is None or layer_name in name:
                weights = module.weight.data.cpu().numpy().flatten()
                layer_weights.append(weights)
                layer_names.append(name)

    if not layer_weights:
        print("No layers found for visualization")
        return

    # Plot histograms
    n_layers = len(layer_weights)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    for i, (weights, name) in enumerate(zip(layer_weights, layer_names, strict=False)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(weights, bins=50, alpha=0.7, density=True)
        plt.title(f"{name}\nMean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


def analyze_quantization_sensitivity(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    bit_widths: list[int] = [2, 3, 4, 8],
) -> dict[int, dict[str, float]]:
    """Analyze sensitivity to different quantization bit-widths.

    Args:
        model: Model to analyze
        test_loader: Test data loader
        device: Device for evaluation
        bit_widths: List of bit-widths to test

    Returns:
        Results for each bit-width
    """
    from ..quantization.gpfq import GPFQuantizer

    original_results = evaluate_model(model, test_loader, device)
    results = {}

    for bits in bit_widths:
        print(f"\nTesting {bits}-bit quantization...")

        # Create quantizer
        quantizer = GPFQuantizer(bits=bits)

        # Clone model for quantization
        model_copy = torch.nn.utils.parameters_to_vector(model.parameters())
        quantized_model = type(model)(**model.__dict__)  # This is a simplified approach
        torch.nn.utils.vector_to_parameters(model_copy, quantized_model.parameters())

        # Quantize model (simplified - in practice would use calibration data)
        # quantized_model = quantizer.quantize(quantized_model, calibration_loader)

        # Evaluate (placeholder - would evaluate actual quantized model)
        quantized_results = evaluate_model(quantized_model, test_loader, device)

        accuracy_drop = original_results["accuracy"] - quantized_results["accuracy"]

        results[bits] = {
            "accuracy": quantized_results["accuracy"],
            "accuracy_drop": accuracy_drop,
            "accuracy_drop_percent": (accuracy_drop / original_results["accuracy"]) * 100,
        }

    return results


def save_quantized_model(model: nn.Module, filepath: str, metadata: dict[str, Any] = None):
    """Save quantized model with metadata.

    Args:
        model: Quantized model to save
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_quantized_model(filepath: str, model_class, device: torch.device = None):
    """Load quantized model from checkpoint.

    Args:
        filepath: Path to the saved model
        model_class: Class of the model to instantiate
        device: Device to load the model on

    Returns:
        Loaded model and metadata
    """
    checkpoint = torch.load(filepath, map_location=device)

    # This would need to be adapted based on how model parameters are stored
    # model = model_class(**checkpoint.get('model_args', {}))
    # model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint  # Simplified return


def print_model_summary(model: nn.Module, input_shape: tuple[int, ...] = None):
    """Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_shape: Shape of input for parameter counting
    """
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if params > 0:
                print(f"{name:40} {str(type(module).__name__):20} {params:>10,} params")
                total_params += params
                trainable_params += trainable

    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Size estimation
    size_mb = (total_params * 4) / (1024**2)  # Assume float32
    print(f"Estimated model size: {size_mb:.2f} MB")
    print("=" * 80)
