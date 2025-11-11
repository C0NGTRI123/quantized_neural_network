import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR10Dataset:
    """CIFAR-10 dataset processor."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True,
        validation_split: float = 0.1,
    ):
        """Initialize CIFAR-10 dataset processor.

        Args:
            data_dir: Directory to store data
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            download: Whether to download dataset
            validation_split: Fraction of training data to use for validation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.validation_split = validation_split

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Dataset properties
        self.num_classes = 10
        self.input_channels = 3
        self.input_size = (32, 32)

    def load_data(self) -> None:
        """Load CIFAR-10 dataset from torchvision."""
        # Training data with augmentation
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        # Test data without augmentation
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        # Load datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=self.download, transform=train_transform
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=self.download, transform=test_transform
        )

        print("CIFAR-10 dataset loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

    def preprocess_data(self) -> None:
        """Apply preprocessing transforms (already handled in load_data)."""
        # Preprocessing is handled in the transforms during load_data
        # This method can be extended if additional preprocessing is needed
        pass

    def train_test_split(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Split training set into train and validation
        train_size = int((1 - self.validation_split) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size

        from torch.utils.data import random_split

        train_subset, val_subset = random_split(self.train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

        self.val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_sample_input(self) -> torch.Tensor:
        """Get a sample input tensor for model testing."""
        return torch.randn(1, self.input_channels, *self.input_size)

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "name": "CIFAR-10",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            "val_samples": len(self.val_loader.dataset) if self.val_loader else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
        }
