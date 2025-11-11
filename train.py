#!/usr/bin/env python3
"""Training script for GPFQ quantized neural networks.

This script provides functionality to train models with optional quantization,
comprehensive logging, checkpointing, and evaluation.
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import our modules
from gpfq import MLP, VGG16, QuantizedCNN, QuantizedNeuralNetwork, SimpleCNN
from gpfq.dataset.cifar_dataset import CIFAR10Dataset
from gpfq.dataset.imagenet_dataset import ImageNetDataset
from gpfq.dataset.mnist_dataset import MNISTDataset
from gpfq.utils.utils import compute_model_size, evaluate_model, measure_inference_time


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name: str, data_dir: str, batch_size: int, num_workers: int):
    """Get dataset loaders based on dataset name.

    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', 'imagenet')
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, test_loader, num_classes, input_channels)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
        dataset.load_data()
        dataset.preprocess_data()
        train_loader, val_loader, test_loader = dataset.train_test_split()

        return train_loader, val_loader, test_loader

    elif dataset_name == "mnist":
        dataset = MNISTDataset(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
        dataset.load_data()
        dataset.preprocess_data()
        train_loader, val_loader, test_loader = dataset.train_test_split()

        return train_loader, val_loader, test_loader

    elif dataset_name == "imagenet":
        dataset = ImageNetDataset(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
        dataset.load_data()
        dataset.preprocess_data()
        train_loader, val_loader, test_loader = dataset.train_test_split()

        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_calibration_loader(train_loader: torch.utils.data.DataLoader, num_samples: int):
    """Create a calibration data loader with a subset of training data.

    Args:
        train_loader: Original training data loader
        num_samples: Number of samples for calibration

    Returns:
        Calibration data loader
    """
    calibration_data = []
    calibration_targets = []

    samples_collected = 0
    for data, targets in train_loader:
        batch_size = data.size(0)
        samples_to_take = min(batch_size, num_samples - samples_collected)

        calibration_data.append(data[:samples_to_take])
        calibration_targets.append(targets[:samples_to_take])

        samples_collected += samples_to_take
        if samples_collected >= num_samples:
            break

    # Concatenate all collected data
    calibration_data = torch.cat(calibration_data, dim=0)
    calibration_targets = torch.cat(calibration_targets, dim=0)

    # Create dataset and loader
    calibration_dataset = torch.utils.data.TensorDataset(calibration_data, calibration_targets)
    calibration_loader = torch.utils.data.DataLoader(
        calibration_dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers
    )

    return calibration_loader


def create_model(model_name: str, dataset: str, **kwargs) -> nn.Module:
    """Create model based on name and dataset.

    Args:
        model_name: Name of the model ('mlp', 'cnn', 'vgg16')
        dataset: Dataset name for input/output dimensions
        **kwargs: Additional model arguments

    Returns:
        Initialized model
    """
    # Determine input channels and classes based on dataset
    if dataset.lower() == "cifar10":
        input_channels, num_classes = 3, 10
        input_size = 32 * 32 * 3
    elif dataset.lower() == "mnist":
        input_channels, num_classes = 1, 10
        input_size = 28 * 28
    elif dataset.lower() == "imagenet":
        input_channels, num_classes = 3, 1000
        input_size = 224 * 224 * 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Create model
    if model_name.lower() == "mlp":
        model = MLP(
            input_size=input_size,
            hidden_sizes=kwargs.get("hidden_sizes", [500, 300]),
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.0),
            use_batch_norm=kwargs.get("use_batch_norm", False),
        )
    elif model_name.lower() == "cnn":
        model = SimpleCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.5),
            use_batch_norm=kwargs.get("use_batch_norm", True),
        )
    elif model_name.lower() == "vgg16":
        model = VGG16(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.5),
            use_batch_norm=kwargs.get("use_batch_norm", True),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    log_interval: int = 100,
) -> dict[str, float]:
    """Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
        writer: TensorBoard writer
        log_interval: Logging interval

    Returns:
        Training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}")

    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * data.size(0)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total_samples += data.size(0)

        # Update progress bar
        current_acc = 100.0 * correct / total_samples
        current_loss = total_loss / total_samples
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % log_interval == 0:
            writer.add_scalar("Train/Loss_Step", loss.item(), global_step)
            writer.add_scalar("Train/Accuracy_Step", current_acc, global_step)

        # Log periodically
        if batch_idx % log_interval == 0:
            logger.debug(
                f"Epoch {epoch}, Batch {batch_idx:4d}/{len(train_loader):4d}: "
                f"Loss: {loss.item():.6f}, Acc: {current_acc:.2f}%"
            )

    # Calculate epoch metrics
    epoch_loss = total_loss / total_samples
    epoch_acc = 100.0 * correct / total_samples

    # Log epoch results
    logger.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.2f}%")
    writer.add_scalar("Train/Loss_Epoch", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy_Epoch", epoch_acc, epoch)

    return {"loss": epoch_loss, "accuracy": epoch_acc, "correct": correct, "total": total_samples}


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
) -> dict[str, float]:
    """Validate model for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
        writer: TensorBoard writer

    Returns:
        Validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")

        for data, targets in progress_bar:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total_samples += data.size(0)

            # Update progress bar
            current_acc = 100.0 * correct / total_samples
            current_loss = total_loss / total_samples
            progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

    # Calculate metrics
    val_loss = total_loss / total_samples
    val_acc = 100.0 * correct / total_samples

    # Log results
    logger.info(f"Epoch {epoch} Validation - Loss: {val_loss:.6f}, Accuracy: {val_acc:.2f}%")
    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    return {"loss": val_loss, "accuracy": val_acc, "correct": correct, "total": total_samples}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    log_dir: Path,
    is_best: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        log_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        metadata: Additional metadata to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_acc,
        "model_class": model.__class__.__name__,
        "metadata": metadata or {},
    }

    # Save latest checkpoint
    latest_path = log_dir / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)

    # Save best model
    if is_best:
        best_path = log_dir / "best_model.pt"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load checkpoint and restore model and optimizer state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state
        device: Device to load tensors on
        logger: Logger instance

    Returns:
        Dictionary containing checkpoint metadata
    """
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
    best_val_acc = checkpoint.get("val_accuracy", 0.0)
    metadata = checkpoint.get("metadata", {})

    if logger:
        logger.info("Checkpoint loaded successfully")
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best validation accuracy so far: {best_val_acc:.2f}%")

    return {"start_epoch": start_epoch, "best_val_acc": best_val_acc, "metadata": metadata}


def get_input_shape_for_dataset(dataset: str, batch_size: int = 1) -> tuple[int, ...]:
    """Get input shape for a given dataset.

    Args:
        dataset: Dataset name
        batch_size: Batch size for input tensor

    Returns:
        Input tensor shape (batch_size, channels, height, width)
    """
    if dataset.lower() == "cifar10":
        return (batch_size, 3, 32, 32)
    elif dataset.lower() == "mnist":
        return (batch_size, 1, 28, 28)
    elif dataset.lower() == "imagenet":
        return (batch_size, 3, 224, 224)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def measure_model_inference_times(
    original_model: nn.Module,
    quantized_model: nn.Module,
    dataset: str,
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Measure and compare inference times between original and quantized models.

    Args:
        original_model: Original full-precision model
        quantized_model: Quantized model
        dataset: Dataset name for determining input shape
        device: Device to run inference on
        num_runs: Number of runs for timing
        warmup_runs: Number of warmup runs
        logger: Logger instance

    Returns:
        Dictionary containing timing results and comparisons
    """
    if logger:
        logger.info("Measuring inference times...")

    # Get input shape for the dataset
    input_shape = get_input_shape_for_dataset(dataset, batch_size=1)

    # Measure original model inference time
    if logger:
        logger.info("Measuring original model inference time...")
    original_timing = measure_inference_time(original_model, input_shape, device, num_runs, warmup_runs)

    # Measure quantized model inference time
    if logger:
        logger.info("Measuring quantized model inference time...")
    quantized_timing = measure_inference_time(quantized_model, input_shape, device, num_runs, warmup_runs)

    # Calculate speedup and comparison metrics
    speedup = original_timing["mean_time"] / quantized_timing["mean_time"]
    time_reduction = original_timing["mean_time"] - quantized_timing["mean_time"]
    time_reduction_percent = (time_reduction / original_timing["mean_time"]) * 100

    results = {
        "original": {
            "mean_time_ms": original_timing["mean_time"] * 1000,
            "std_time_ms": original_timing["std_time"] * 1000,
            "min_time_ms": original_timing["min_time"] * 1000,
            "max_time_ms": original_timing["max_time"] * 1000,
            "median_time_ms": original_timing["median_time"] * 1000,
        },
        "quantized": {
            "mean_time_ms": quantized_timing["mean_time"] * 1000,
            "std_time_ms": quantized_timing["std_time"] * 1000,
            "min_time_ms": quantized_timing["min_time"] * 1000,
            "max_time_ms": quantized_timing["max_time"] * 1000,
            "median_time_ms": quantized_timing["median_time"] * 1000,
        },
        "comparison": {
            "speedup": speedup,
            "time_reduction_ms": time_reduction * 1000,
            "time_reduction_percent": time_reduction_percent,
        },
        "measurement_config": {
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "input_shape": input_shape,
            "device": str(device),
        },
    }

    if logger:
        logger.info("Inference time comparison:")
        logger.info(
            f"  Original model: {results['original']['mean_time_ms']:.2f} ± {results['original']['std_time_ms']:.2f} ms"
        )
        logger.info(
            f"  Quantized model: {results['quantized']['mean_time_ms']:.2f} ± {results['quantized']['std_time_ms']:.2f} ms"
        )
        logger.info(f"  Speedup: {results['comparison']['speedup']:.2f}x")
        logger.info(
            f"  Time reduction: {results['comparison']['time_reduction_ms']:.2f} ms ({results['comparison']['time_reduction_percent']:.1f}%)"
        )

    return results


def quantize_model(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device,
    bits: int = 4,
    alphabet_scalar: int = 1,
    logger: logging.Logger | None = None,
    model_type: str = "mlp",
) -> nn.Module:
    """Quantize model using GPFQ.

    Args:
        model: Model to quantize
        calibration_loader: Data for calibration
        device: Device to run quantization on
        bits: Number of bits for quantization
        logger: Logger instance
        model_type: Type of model ('mlp', 'cnn', 'vgg16')

    Returns:
        Quantized model
    """
    if logger:
        logger.info(f"Starting {bits}-bit quantization...")

    # Ensure model is on the correct device
    model = model.to(device)

    # Create a device-aware calibration loader
    device_calibration_data = []
    device_calibration_targets = []

    for data, targets in calibration_loader:
        device_calibration_data.append(data.to(device))
        device_calibration_targets.append(targets.to(device))

    # Create new calibration loader with device-aware data
    device_calibration_dataset = torch.utils.data.TensorDataset(
        torch.cat(device_calibration_data, dim=0), torch.cat(device_calibration_targets, dim=0)
    )
    device_calibration_loader = torch.utils.data.DataLoader(
        device_calibration_dataset,
        batch_size=calibration_loader.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with GPU tensors
    )

    # Choose appropriate quantizer based on model type
    if model_type.lower() in ["cnn", "vgg16"]:
        # Use QuantizedCNN for models with convolutional layers
        quantizer = QuantizedCNN(
            model=model,
            dataloader=device_calibration_loader,
            mini_batch_size=calibration_loader.batch_size,
            logger=logger,
            bits=bits,
            alphabet_scalar=alphabet_scalar,
            quantize_conv=True,
        )
    else:
        # Use QuantizedNeuralNetwork for MLP models
        quantizer = QuantizedNeuralNetwork(
            model=model,
            dataloader=device_calibration_loader,
            mini_batch_size=calibration_loader.batch_size,
            logger=logger,
            bits=bits,
            alphabet_scalar=alphabet_scalar,
        )

    # Perform quantization
    quantizer.quantize_network()

    # Get the quantized model
    quantized_model = quantizer.quantized

    if logger:
        logger.info("Quantization completed")

    return quantized_model


def find_latest_checkpoint(log_base_dir: str, model_name: str, dataset_name: str) -> str | None:
    """Find the most recent checkpoint for a given model and dataset.

    Args:
        log_base_dir: Base directory for logs
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Path to the most recent checkpoint, or None if not found
    """
    logs_base_dir = Path(log_base_dir)
    log_dir = logs_base_dir / f"{model_name}_{dataset_name}"

    if not log_dir.exists():
        return None

    # Look for latest checkpoint first
    latest_checkpoint = log_dir / "latest_checkpoint.pt"
    if latest_checkpoint.exists():
        return str(latest_checkpoint)

    # Fallback to best model if latest doesn't exist
    best_checkpoint = log_dir / "best_model.pt"
    if best_checkpoint.exists():
        return str(best_checkpoint)

    return None


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train GPFQ quantized neural networks")

    # Configuration file argument
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")

    # Model arguments
    parser.add_argument("--model", type=str, default="cnn", choices=["mlp", "cnn", "vgg16"], help="Model architecture")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "imagenet"], help="Dataset to use"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")

    # Model specific arguments
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--use-batch-norm", action="store_true", help="Use batch normalization")

    # Quantization arguments
    parser.add_argument("--quantize", action="store_true", help="Apply post-training quantization")
    parser.add_argument("--quantize-bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--alphabet-scalar", type=int, default=1, help="Alphabet scalar for quantization")
    parser.add_argument("--calibration-samples", type=int, default=1024, help="Number of calibration samples")
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Only perform quantization without training (requires existing checkpoint)",
    )
    parser.add_argument(
        "--load-best-for-quantization",
        action="store_true",
        help="Load best checkpoint for quantization instead of training",
    )

    # Inference timing arguments
    parser.add_argument(
        "--skip-timing", action="store_true", help="Skip inference timing measurements to speed up quantization"
    )
    parser.add_argument(
        "--timing-runs", type=int, default=100, help="Number of runs for inference timing (default: 100)"
    )
    parser.add_argument(
        "--timing-warmup", type=int, default=10, help="Number of warmup runs for inference timing (default: 10)"
    )

    # Resume training arguments
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--resume-from-best", action="store_true", help="Resume from best model checkpoint instead of latest"
    )
    parser.add_argument(
        "--auto-resume", action="store_true", help="Automatically find and resume from the most recent checkpoint"
    )

    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging arguments
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        config = load_config(args.config)

        # Override args with config file values, but preserve command line overrides
        for key, value in config.items():
            # Convert underscores to hyphens for consistency
            attr_key = key.replace("-", "_")

            # Only set if the argument exists and wasn't explicitly set on command line
            if hasattr(args, attr_key):
                # Check if this was set via command line by comparing with default
                parser_default = parser.get_default(attr_key)
                current_value = getattr(args, attr_key)

                # If current value is the default, use config value
                if current_value == parser_default:
                    setattr(args, attr_key, value)

    # Set random seed
    set_seed(args.seed)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create log directory
    log_dir = Path(args.log_dir) / f"{args.model}_{args.dataset}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir, args.log_level)
    logger.info(f"Starting training with args: {vars(args)}")
    logger.info(f"Using device: {device}")

    # Setup TensorBoard
    writer = SummaryWriter(log_dir / "tensorboard")

    # Save configuration
    config_path = log_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    try:
        # Load data
        logger.info(f"Loading {args.dataset} dataset...")
        train_loader, val_loader, test_loader = get_dataset(
            args.dataset, data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
        logger.info(
            f"Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches, {len(val_loader)} validation batches"
        )

        # Create model
        logger.info(f"Creating {args.model} model...")
        model_kwargs = {"dropout": args.dropout, "use_batch_norm": args.use_batch_norm}
        model = create_model(args.model, args.dataset, **model_kwargs)
        model = model.to(device)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")

        # Log model size
        model_size = compute_model_size(model)
        logger.info(f"Model size: {model_size['size_mb']:.2f} MB")

        # Create optimizer
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )

        # Create loss criterion
        criterion = nn.CrossEntropyLoss()

        # Initialize default values
        start_epoch = 1
        best_val_acc = 0.0

        # Handle quantization-only mode
        if args.quantize_only:
            if not args.quantize:
                logger.error("--quantize-only requires --quantize flag to be set")
                raise ValueError("--quantize-only requires --quantize flag to be set")

            # Look for best checkpoint
            best_path = log_dir / "best_model.pt"
            if not best_path.exists():
                logger.error(f"No best checkpoint found at {best_path} for quantization")
                raise FileNotFoundError(f"No best checkpoint found at {best_path}")

            logger.info("Quantization-only mode: Loading best checkpoint...")
            checkpoint_info = load_checkpoint(str(best_path), model, optimizer, device, logger)
            best_val_acc = checkpoint_info["best_val_acc"]

            # Skip training and go directly to quantization
            logger.info("Skipping training, proceeding directly to quantization...")

        elif args.load_best_for_quantization and args.quantize:
            # Load best checkpoint but still do training if not quantize-only
            best_path = log_dir / "best_model.pt"
            if best_path.exists():
                logger.info("Loading best checkpoint for quantization...")
                checkpoint_info = load_checkpoint(str(best_path), model, optimizer, device, logger)
                start_epoch = 1  # Reset to start training from beginning but with loaded weights
                best_val_acc = checkpoint_info["best_val_acc"]
                logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
            else:
                logger.warning("No best checkpoint found, starting training from scratch")

        if not args.quantize_only and args.resume:
            checkpoint_path = args.resume
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, device, logger)
            start_epoch = checkpoint_info["start_epoch"]
            best_val_acc = checkpoint_info["best_val_acc"]

            logger.info(f"Resumed training from epoch {start_epoch}")
            logger.info(f"Best validation accuracy so far: {best_val_acc:.2f}%")

        elif not args.quantize_only and args.auto_resume:
            # Automatically find the most recent checkpoint
            checkpoint_path = find_latest_checkpoint(args.log_dir, args.model, args.dataset)

            if checkpoint_path:
                logger.info(f"Auto-resuming training from checkpoint: {checkpoint_path}")
                checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, device, logger)
                start_epoch = checkpoint_info["start_epoch"]
                best_val_acc = checkpoint_info["best_val_acc"]

                logger.info(f"Auto-resumed training from epoch {start_epoch}")
                logger.info(f"Best validation accuracy so far: {best_val_acc:.2f}%")
            else:
                logger.info("No previous checkpoint found for auto-resume, starting training from scratch")
        elif not args.quantize_only and args.resume_from_best:
            # Look for best model in the log directory
            best_path = log_dir / "best_model.pt"

            if best_path.exists():
                checkpoint_path = str(best_path)
                logger.info(f"Resuming training from best model: {checkpoint_path}")
                checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, device, logger)
                start_epoch = checkpoint_info["start_epoch"]
                best_val_acc = checkpoint_info["best_val_acc"]

                logger.info(f"Resumed training from epoch {start_epoch}")
                logger.info(f"Best validation accuracy so far: {best_val_acc:.2f}%")

                # Handle epoch extension when resuming from best
                if args.extend_epochs:
                    original_epochs = args.epochs
                    args.epochs = start_epoch - 1 + args.extend_epochs
                    logger.info(f"Extended training from {original_epochs} to {args.epochs} epochs")
            else:
                logger.warning("No best model checkpoint found, starting training from scratch")

        # Training loop (skip if quantization-only)
        if not args.quantize_only:
            logger.info("Starting training...")
            start_time = time.time()

            for epoch in range(start_epoch, args.epochs + 1):
                epoch_start = time.time()

                # Train
                train_metrics = train_epoch(
                    model, train_loader, criterion, optimizer, device, epoch, logger, writer, args.log_interval
                )

                # Validate
                val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, logger, writer)

                # Log learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Train/Learning_Rate", current_lr, epoch)

                # Check if best model
                is_best = val_metrics["accuracy"] > best_val_acc
                # Save checkpoint only if it's the best model
                if is_best:
                    best_val_acc = val_metrics["accuracy"]
                    logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
                    metadata = {
                        "args": vars(args),
                        "epoch": epoch,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "model_size": model_size,
                    }
                    save_checkpoint(model, optimizer, epoch, val_metrics["accuracy"], log_dir, is_best, metadata)

                # Log epoch time
                epoch_time = time.time() - epoch_start
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                writer.add_scalar("Time/Epoch_Duration", epoch_time, epoch)

            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")
            logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        else:
            logger.info("Skipping training (quantization-only mode)")
            # Set dummy values for results
            train_metrics = {"accuracy": 0.0}
            total_time = 0.0

        # Quantization (if requested)
        if args.quantize:
            logger.info("Starting post-training quantization process...")

            # If not quantization-only mode and we haven't loaded best model yet, load it now
            if not args.quantize_only and not args.load_best_for_quantization:
                best_path = log_dir / "best_model.pt"
                if best_path.exists():
                    logger.info("Loading best model for quantization...")
                    checkpoint_info = load_checkpoint(str(best_path), model, optimizer, device, logger)
                    logger.info(f"Loaded best model with validation accuracy: {checkpoint_info['best_val_acc']:.2f}%")
                else:
                    logger.warning("No best checkpoint found, quantizing current model state")

            # Create calibration data
            logger.info(f"Creating calibration data with {args.calibration_samples} samples...")
            calibration_loader = get_calibration_loader(train_loader, args.calibration_samples)

            # Quantize model
            quantized_model = quantize_model(
                model, calibration_loader, device, args.quantize_bits, args.alphabet_scalar, logger, args.model
            )

            # Evaluate quantized model
            logger.info("Evaluating quantized model...")
            original_results = evaluate_model(model, test_loader, device)
            quantized_results = evaluate_model(quantized_model, test_loader, device)

            # Log quantization results
            accuracy_drop = original_results["accuracy"] - quantized_results["accuracy"]
            logger.info(f"Original model accuracy: {original_results['accuracy']:.2f}%")
            logger.info(f"Quantized model accuracy: {quantized_results['accuracy']:.2f}%")
            logger.info(f"Accuracy drop: {accuracy_drop:.2f}%")

            # Calculate model size reduction
            original_size = compute_model_size(model)
            quantized_size = compute_model_size(quantized_model)
            size_reduction = ((original_size["size_mb"] - quantized_size["size_mb"]) / original_size["size_mb"]) * 100
            logger.info(f"Original model size: {original_size['size_mb']:.2f} MB")
            logger.info(f"Quantized model size: {quantized_size['size_mb']:.2f} MB")
            logger.info(f"Size reduction: {size_reduction:.1f}%")

            # Measure inference time comparison (optional)
            inference_timing = None
            if not args.skip_timing:
                logger.info("Measuring inference time comparison...")
                inference_timing = measure_model_inference_times(
                    original_model=model,
                    quantized_model=quantized_model,
                    dataset=args.dataset,
                    device=device,
                    num_runs=args.timing_runs,
                    warmup_runs=args.timing_warmup,
                    logger=logger,
                )
            else:
                logger.info("Skipping inference timing measurements (--skip-timing flag set)")

            # Save quantized model
            quantized_path = log_dir / "quantized_model.pt"
            quantized_metadata = {
                "args": vars(args),
                "quantization_bits": args.quantize_bits,
                "original_accuracy": original_results["accuracy"],
                "quantized_accuracy": quantized_results["accuracy"],
                "accuracy_drop": accuracy_drop,
                "original_size_mb": original_size["size_mb"],
                "quantized_size_mb": quantized_size["size_mb"],
                "size_reduction_percent": size_reduction,
                "calibration_samples": args.calibration_samples,
            }
            torch.save(
                {
                    "model_state_dict": quantized_model.state_dict(),
                    "model_class": quantized_model.__class__.__name__,
                    "metadata": quantized_metadata,
                },
                quantized_path,
            )

            logger.info(f"Quantized model saved to {quantized_path}")

            # If this is quantization-only mode, we're done
            if args.quantize_only:
                logger.info("Quantization completed successfully!")
                return

        # Final evaluation on test set (skip if quantization-only)
        if not args.quantize_only:
            logger.info("Evaluating final model on test set...")
            final_test_results = evaluate_model(model, test_loader, device)
            logger.info(f"Final test accuracy: {final_test_results['accuracy']:.2f}%")
        else:
            # Use quantized results as final results
            final_test_results = quantized_results if args.quantize else {"accuracy": 0.0}

        # Save final results
        results = {
            "final_train_accuracy": train_metrics["accuracy"],
            "best_val_accuracy": best_val_acc,
            "final_test_accuracy": final_test_results["accuracy"],
            "total_training_time": total_time,
            "total_epochs": args.epochs if not args.quantize_only else 0,
            "epochs_trained": (args.epochs - start_epoch + 1) if not args.quantize_only else 0,
            "model_parameters": total_params,
            "model_size_mb": model_size["size_mb"],
            "resumed_from_epoch": start_epoch if args.resume or args.resume_from_best else None,
            "quantization_only_mode": args.quantize_only,
        }

        if args.quantize:
            quantization_results = {
                "quantized_accuracy": quantized_results["accuracy"],
                "accuracy_drop": accuracy_drop,
                "quantization_bits": args.quantize_bits,
            }
            if inference_timing is not None:
                quantization_results["inference_timing"] = inference_timing
            results.update(quantization_results)

        results_path = log_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    finally:
        writer.close()


if __name__ == "__main__":
    main()
