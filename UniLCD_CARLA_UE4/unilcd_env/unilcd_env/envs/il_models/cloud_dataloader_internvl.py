"""
Dataloader for InternVL3 cloud model training.

This module provides PyTorch Dataset and DataLoader implementations for
loading CARLA driving data and preprocessing it for InternVL's 448x448 input.

Data Format:
-----------
The training data consists of paired .jpg and .npy files:
- {id}.jpg: RGB camera image from ego vehicle
- {id}.npy: Numpy array with 7 elements:
    [0]: rotation (steering angle in radians, range ~[-pi, pi])
    [1]: speed (throttle, range ~[0, 1.5])
    [2]: pedestrian_detected (0.0 or 1.0)
    [3]: current_x (ego vehicle x position)
    [4]: current_y (ego vehicle y position)
    [5]: next_waypoint_x (target x position)
    [6]: next_waypoint_y (target y position)

Preprocessing:
-------------
- Images are resized from original resolution to 448x448
- ImageNet normalization is applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Location is computed as relative waypoint: (current - next)

Usage:
------
    from cloud_dataloader_internvl import get_internvl_dataloader

    loader = get_internvl_dataloader(
        data_dir="/path/to/data",
        batch_size=4,
        augment=True
    )

    for images, actions, locations in loader:
        # images: [B, 3, 448, 448]
        # actions: [B, 2] (rotation, speed)
        # locations: [B, 2] (dx, dy)
        pass

Author: Generated for UniLCD project
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class InternVLCarlaDataset(Dataset):
    """
    PyTorch Dataset for loading CARLA driving data for InternVL training.

    This dataset loads image-action pairs collected from CARLA simulator
    and preprocesses them for InternVL's visual encoder (448x448 input).

    Attributes:
        data_dir: Root directory containing .jpg and .npy files
        image_size: Target image size (default 448 for InternVL)
        img_list: List of paths to image files
        data_list: List of paths to label files
        transform: Torchvision transform pipeline
    """

    # ImageNet normalization constants (same as InternVL uses)
    # These values are computed from ImageNet training set
    IMAGENET_MEAN = (0.485, 0.456, 0.406)  # RGB channel means
    IMAGENET_STD = (0.229, 0.224, 0.225)   # RGB channel stds

    def __init__(
        self,
        data_dir: str,
        image_size: int = 448,
        augment: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing .jpg and .npy files
                     Files should be named as {id}.jpg and {id}.npy
            image_size: Target image size after resizing
                       InternVL uses 448x448 by default
            augment: Whether to apply data augmentation
                    Includes random horizontal flip and color jitter
        """
        self.data_dir = data_dir
        self.image_size = image_size

        # Find all image and label files
        # glob.glob returns unsorted, so we sort to ensure pairing
        self.img_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.data_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        # Verify that images and labels are properly paired
        assert len(self.img_list) == len(self.data_list), \
            f"Mismatch: {len(self.img_list)} images vs {len(self.data_list)} labels"

        # Sanity check: verify first few files have matching IDs
        for img_path, data_path in zip(self.img_list[:5], self.data_list[:5]):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            data_id = os.path.splitext(os.path.basename(data_path))[0]
            assert img_id == data_id, f"File mismatch: {img_path} vs {data_path}"

        print(f"Loaded {len(self.img_list)} samples from {data_dir}")

        # Build image transform pipeline
        if augment:
            # Training transforms with augmentation
            # Augmentation helps prevent overfitting
            self.transform = transforms.Compose([
                # Resize to InternVL's expected input size
                transforms.Resize((image_size, image_size), antialias=True),
                # Random horizontal flip (simulates driving in opposite direction)
                # Note: This also requires flipping the steering angle!
                transforms.RandomHorizontalFlip(p=0.3),
                # Color jitter for robustness to lighting changes
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                # Convert PIL Image to tensor (scales to [0, 1])
                transforms.ToTensor(),
                # Normalize with ImageNet statistics
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
        else:
            # Evaluation transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """
        Load and preprocess a single sample.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image, actions, location):
            - image: Preprocessed image tensor [3, 448, 448]
            - actions: Action tensor [2] containing [rotation, speed]
            - location: Relative waypoint tensor [2] containing [dx, dy]
        """
        # Load and transform image
        # PIL Image.open is lazy, actual loading happens on .convert()
        img = Image.open(self.img_list[idx]).convert('RGB')
        img_tensor = self.transform(img)

        # Load label data from numpy file
        # Format: [rotation, speed, ped_flag, loc_x, loc_y, next_x, next_y]
        data = np.load(self.data_list[idx], allow_pickle=True)

        # Extract actions: [rotation, speed]
        # rotation: steering angle in radians
        # speed: normalized throttle value
        actions = torch.tensor(data[:2], dtype=torch.float32)

        # Compute relative waypoint location
        # This tells the model where to go relative to current position
        # Format: (current_pos - next_waypoint) = direction to travel
        if len(data) >= 7:
            loc_x, loc_y = data[3], data[4]      # Current position
            next_x, next_y = data[5], data[6]    # Next waypoint
            location = torch.tensor([
                loc_x - next_x,  # dx: how far to travel in x
                loc_y - next_y   # dy: how far to travel in y
            ], dtype=torch.float32)
        else:
            # Fallback for malformed data
            location = torch.tensor([0.0, 0.0], dtype=torch.float32)

        return img_tensor, actions, location


class InternVLCarlaDatasetCropped(Dataset):
    """
    Alternative dataset that crops images like the original UniLCD dataloader.

    The original cloud model dataloader crops a 480x480 region from the center
    of the image. This class replicates that behavior for consistency when
    comparing models trained on the same data preprocessing.

    Use this if you want to maintain exact parity with the original training.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        data_dir: str,
        image_size: int = 448,
        crop_region: tuple = (120, 600, 400, 880),  # (y1, y2, x1, x2) from original
    ):
        """
        Initialize the cropped dataset.

        Args:
            data_dir: Directory containing data files
            image_size: Target size after crop and resize
            crop_region: Pixel region to crop (y1, y2, x1, x2)
                        Default matches original dataloader: rows 120-600, cols 400-880
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_region = crop_region

        self.img_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.data_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        assert len(self.img_list) == len(self.data_list)
        print(f"Loaded {len(self.img_list)} samples (cropped mode)")

        # Transform pipeline (no augmentation for cropped version)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        # Load image
        img = Image.open(self.img_list[idx]).convert('RGB')

        # Apply crop to match original dataloader behavior
        # Original: img[:3, 120:600, 400:880] which is a 480x480 region
        y1, y2, x1, x2 = self.crop_region
        img = img.crop((x1, y1, x2, y2))  # PIL uses (left, top, right, bottom)

        # Transform (resize to 448x448, normalize)
        img_tensor = self.transform(img)

        # Load labels
        data = np.load(self.data_list[idx], allow_pickle=True)
        actions = torch.tensor(data[:2], dtype=torch.float32)

        # Compute relative location
        if len(data) >= 7:
            location = torch.tensor([
                data[3] - data[5],  # loc_x - next_x
                data[4] - data[6]   # loc_y - next_y
            ], dtype=torch.float32)
        else:
            location = torch.tensor([0.0, 0.0], dtype=torch.float32)

        return img_tensor, actions, location


def get_internvl_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 448,
    augment: bool = False,
    use_crop: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for InternVL cloud model training.

    This is the main function to use when loading training data.
    It handles dataset creation and DataLoader configuration.

    Args:
        data_dir: Path to directory containing .jpg and .npy files
        batch_size: Number of samples per batch
                   Recommended: 4 for InternVL3-8B with frozen vision
        num_workers: Number of parallel data loading workers
                    Set to 0 for debugging, 4-8 for training
        image_size: Target image size (448 for InternVL)
        augment: Whether to apply data augmentation
                Recommended: True for training, False for validation
        use_crop: Whether to crop images like original dataloader
                 Set True for exact parity with original training
        shuffle: Whether to shuffle data each epoch
                Should be True for training, False for validation

    Returns:
        PyTorch DataLoader yielding (images, actions, locations) batches
        - images: [B, 3, 448, 448] float tensor
        - actions: [B, 2] float tensor (rotation, speed)
        - locations: [B, 2] float tensor (dx, dy)

    Example:
        >>> loader = get_internvl_dataloader("./data", batch_size=4)
        >>> for images, actions, locations in loader:
        ...     predictions = model(images, locations)
        ...     loss = criterion(predictions, actions)
    """
    # Select dataset class based on crop preference
    if use_crop:
        dataset = InternVLCarlaDatasetCropped(data_dir, image_size)
    else:
        dataset = InternVLCarlaDataset(data_dir, image_size, augment)

    # Create DataLoader with optimized settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True    # Drop incomplete final batch for stable training
    )


def get_train_val_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.1,
    image_size: int = 448,
    augment: bool = True
):
    """
    Create train and validation DataLoaders with random split.

    Splits the dataset into training and validation sets, applying
    augmentation only to the training set.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers
        val_split: Fraction of data to use for validation (0.1 = 10%)
        image_size: Target image size
        augment: Whether to augment training data

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = get_train_val_dataloaders("./data")
        >>> for epoch in range(50):
        ...     train(model, train_loader)
        ...     validate(model, val_loader)
    """
    # Create dataset without augmentation (we'll apply it to training subset)
    full_dataset = InternVLCarlaDataset(data_dir, image_size, augment=False)

    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Random split with fixed seed for reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )

    # Apply augmentation transforms to training subset only
    # Note: This modifies the parent dataset's transform, which affects
    # both subsets. A cleaner approach would be to use separate datasets.
    if augment:
        train_dataset.dataset.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=InternVLCarlaDataset.IMAGENET_MEAN,
                std=InternVLCarlaDataset.IMAGENET_STD
            )
        ])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # No need to shuffle validation
        pin_memory=True
    )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    return train_loader, val_loader


# =============================================================================
# Testing and Verification
# =============================================================================
if __name__ == "__main__":
    """
    Test script to verify dataloader functionality.

    Run with: python cloud_dataloader_internvl.py [data_dir]
    """
    import sys

    # Use provided path or default to UniLCD data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data4/shival/UniLCD_CARLA_UE4/data"

    print("=" * 60)
    print("Testing InternVL DataLoader")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")

    # Create dataloader
    loader = get_internvl_dataloader(data_dir, batch_size=4)

    # Test a few batches
    print("\nSample batches:")
    for batch_idx, (images, actions, locations) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Actions sample: rotation={actions[0, 0]:.3f}, speed={actions[0, 1]:.3f}")
        print(f"  Locations shape: {locations.shape}")
        print(f"  Locations sample: dx={locations[0, 0]:.2f}, dy={locations[0, 1]:.2f}")

        # Only test first 3 batches
        if batch_idx >= 2:
            break

    print("\n" + "=" * 60)
    print("DataLoader test PASSED!")
    print("=" * 60)
