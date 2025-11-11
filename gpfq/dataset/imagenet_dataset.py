import concurrent.futures
import os
import shutil
from glob import glob
from pathlib import Path
from time import time

import cv2
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


class ImageNetValidationDataset(Dataset):
    """Custom dataset for preprocessed ImageNet validation data."""

    def __init__(self, preprocessed_dir: Path, labels: np.ndarray, transform=None):
        """Initialize dataset.

        Args:
            preprocessed_dir: Directory containing preprocessed .npy files
            labels: Array of labels
            transform: Optional transforms to apply
        """
        self.preprocessed_dir = preprocessed_dir
        self.labels = labels
        self.transform = transform

        # Get list of preprocessed files
        self.image_files = sorted(glob(str(preprocessed_dir / "preprocessed_*.npy")))

        # Ensure we have the same number of images and labels
        assert len(self.image_files) == len(self.labels), (
            f"Mismatch: {len(self.image_files)} images, {len(self.labels)} labels"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load preprocessed image
        image = np.load(self.image_files[idx])  # RGB format
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        from PIL import Image

        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageNetDataset:
    """ImageNet dataset processor following the BaseDataset interface.
    This implementation focuses on the validation set for quantization experiments.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = False,
        validation_split: float = 0.1,
        quantization_training_size: int = 1000,
    ):
        """Initialize ImageNet dataset processor.

        Args:
            data_dir: Directory containing ImageNet data
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            download: Not applicable for ImageNet (must be manually downloaded)
            validation_split: Fraction of validation data to use for calibration
            quantization_training_size: Number of samples for quantization calibration
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.validation_split = validation_split
        self.quantization_training_size = quantization_training_size

        # Paths
        self.dir_images = self.data_dir / "val"
        self.dir_processed_images = self.data_dir / "preprocessed_val"
        self.path_labels = self.data_dir / "ILSVRC2012_validation_ground_truth.txt"
        self.path_synset_words = self.data_dir / "synset_words.txt"
        self.path_meta = self.data_dir / "meta.mat"
        self.path_labels_npy = self.data_dir / "y_val.npy"

        # Dataset properties
        self.num_classes = 1000
        self.input_channels = 3
        self.input_size = (224, 224)

        # Data containers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.labels = None

    def _check_data_exists(self) -> bool:
        """Check if required ImageNet files exist."""
        required_files = [self.dir_images, self.path_labels, self.path_synset_words, self.path_meta]

        for file_path in required_files:
            if not file_path.exists():
                print(f"Missing required file: {file_path}")
                return False
        return True

    def _clean_up_processed_images(self):
        """Remove existing preprocessed images directory."""
        if self.dir_processed_images.exists():
            shutil.rmtree(self.dir_processed_images)

    def _preprocess_image(self, image_path: str):
        """Preprocess a single image: resize to 256, center crop to 224."""
        try:
            # Load image in BGR format
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return

            # Resize to 256 on the shorter side
            height, width, _ = image.shape
            min_side = min(height, width)
            new_height = height * 256 // min_side
            new_width = width * 256 // min_side
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Center crop to 224x224
            height, width, _ = image.shape
            startx = width // 2 - 224 // 2
            starty = height // 2 - 224 // 2
            image = image[starty : starty + 224, startx : startx + 224]

            assert image.shape[0] == 224 and image.shape[1] == 224, f"Wrong shape: {image.shape}"

            # Save as numpy array (convert BGR to RGB)
            filename = os.path.basename(image_path).split(".")[0]
            new_filename = f"preprocessed_{filename}.npy"
            new_image_path = self.dir_processed_images / new_filename

            np.save(str(new_image_path), image[..., ::-1])  # BGR to RGB

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    def _generate_labels(self):
        """Generate labels from ImageNet metadata."""
        if self.path_labels_npy.exists():
            print("Loading existing labels...")
            self.labels = np.load(str(self.path_labels_npy))
            return

        print("Generating labels from metadata...")

        # Load metadata
        meta = scipy.io.loadmat(str(self.path_meta))
        original_idx_to_synset = {}
        synset_to_name = {}

        # Build synset mappings
        for i in range(1000):
            ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
            synset = meta["synsets"][i, 0][1][0]
            name = meta["synsets"][i, 0][2][0]
            original_idx_to_synset[ilsvrc2012_id] = synset
            synset_to_name[synset] = name

        # Build keras idx mappings
        synset_to_keras_idx = {}
        keras_idx_to_name = {}
        with open(str(self.path_synset_words)) as f:
            for idx, line in enumerate(f):
                parts = line.split(" ")
                synset_to_keras_idx[parts[0]] = idx
                keras_idx_to_name[idx] = " ".join(parts[1:])

        # Convert function
        def convert_original_idx_to_keras_idx(idx):
            return synset_to_keras_idx[original_idx_to_synset[idx]]

        # Load and convert labels
        with open(str(self.path_labels)) as f:
            y_val = f.read().strip().split("\n")
            y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

        # Save labels
        np.save(str(self.path_labels_npy), y_val)
        self.labels = y_val
        print(f"Generated {len(y_val)} labels")

    def load_data(self) -> None:
        """Load and preprocess ImageNet validation data."""
        if not self._check_data_exists():
            raise FileNotFoundError(
                "ImageNet data files not found. Please download ImageNet validation set "
                "and place the required files in the data directory."
            )

        # Generate labels
        self._generate_labels()

        # Clean up and create processed images directory
        self._clean_up_processed_images()
        os.makedirs(self.dir_processed_images, exist_ok=True)

        # Get image paths
        image_paths = sorted(glob(str(self.dir_images / "*")))
        n_images = len(image_paths)

        print(f"Preprocessing {n_images} images...")
        tic = time()

        # Preprocess images in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_path = {
                executor.submit(self._preprocess_image, image_path): image_path for image_path in image_paths
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_path):
                future.result()
                completed += 1
                if completed % 1000 == 0:
                    print(f"Processed {completed}/{n_images} images")

        print(f"Preprocessing completed in {time() - tic:.2f} seconds")

        # Verify preprocessed images
        preprocessed_paths = sorted(glob(str(self.dir_processed_images / "preprocessed_*.npy")))
        print(f"Successfully preprocessed {len(preprocessed_paths)} images")

    def preprocess_data(self) -> None:
        """Apply preprocessing transforms (handled during load_data)."""
        # Additional preprocessing can be added here if needed
        pass

    def train_test_split(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for calibration, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.labels is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Define transforms
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Create full dataset
        full_dataset = ImageNetValidationDataset(self.dir_processed_images, self.labels, transform=test_transform)

        # Split dataset
        total_size = len(full_dataset)

        # Use specified number of samples for quantization training (calibration)
        train_size = min(self.quantization_training_size, total_size)
        remaining_size = total_size - train_size

        # Split remaining data into validation and test
        val_size = int(remaining_size * self.validation_split)
        test_size = remaining_size - val_size

        # Random split
        self.train_dataset, val_test_dataset = random_split(full_dataset, [train_size, remaining_size])

        self.val_dataset, self.test_dataset = random_split(val_test_dataset, [val_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        print("Dataset split:")
        print(f"  Calibration samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

        return self.train_loader, self.val_loader, self.test_loader

    def get_sample_input(self) -> torch.Tensor:
        """Get a sample input tensor for model testing."""
        return torch.randn(1, self.input_channels, *self.input_size)

    def get_dataset_info(self) -> dict:
        """Get dataset information."""
        return {
            "name": "ImageNet",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
            "total_preprocessed": len(glob(str(self.dir_processed_images / "*.npy")))
            if self.dir_processed_images.exists()
            else 0,
        }
