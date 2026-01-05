import glob

import numpy as np

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
import os


class CarlaDataset(Dataset):
    """
    Dataset for CARLA data. Supports two formats:

    Legacy format (UE4):
        data_dir/
        ├── *.jpg     (images)
        └── *.npy     (actions)

    New format (UE5):
        data_dir/
        ├── rgb_up/   (or rgb_down/, specify with img_folder)
        │   └── *.png
        └── actions/
            └── *.npy
    """

    def __init__(self, data_dir, img_folder=None, crop=None):
        """
        Args:
            data_dir: Root data directory
            img_folder: Subfolder for images (e.g., 'rgb_up'). If None, uses legacy format.
            crop: Tuple of (y1, y2, x1, x2) for cropping. If None, uses full image or default crop.
        """
        self.data_dir = data_dir
        self.img_folder = img_folder
        self.crop = crop

        # Detect format and load file lists
        if img_folder is not None:
            # New UE5 format: images in subfolder, actions in 'actions/'
            img_path = os.path.join(data_dir, img_folder)
            action_path = os.path.join(data_dir, 'actions')
            self.img_list = sorted(glob.glob(os.path.join(img_path, '*.png')))
            self.data_list = sorted(glob.glob(os.path.join(action_path, '*.npy')))
            self.format = 'ue5'
        else:
            # Legacy format: images and actions in root
            self.img_list = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
            if not self.img_list:
                self.img_list = sorted(glob.glob(os.path.join(data_dir, '*.png')))
            self.data_list = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
            self.format = 'legacy'

        # Verify matching counts
        if len(self.img_list) != len(self.data_list):
            print(f"Warning: Image count ({len(self.img_list)}) != Action count ({len(self.data_list)})")
            # Use minimum to avoid index errors
            min_count = min(len(self.img_list), len(self.data_list))
            self.img_list = self.img_list[:min_count]
            self.data_list = self.data_list[:min_count]

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"CarlaDataset: Loaded {len(self.data_list)} samples ({self.format} format)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action.

        Returns:
            (image, actions, location_delta) as torch.Tensor
            - image: normalized RGB tensor
            - actions: [steering, speed]
            - location_delta: [delta_x, delta_y] to next waypoint
        """
        # Load action data
        data = np.load(self.data_list[idx], allow_pickle=True)

        # Load image
        img = read_image(self.img_list[idx])
        img = img[:3]  # Ensure RGB only (drop alpha if present)

        # Apply crop if specified
        if self.crop is not None:
            y1, y2, x1, x2 = self.crop
            img = img[:, y1:y2, x1:x2]
        elif self.format == 'legacy':
            # Default legacy crop (for 640x320 images -> 480x480 center crop)
            img = img[:, 120:600, 400:880]
        # For UE5 format without explicit crop, use full image

        # Normalize
        normalized_image = self.normalize(img.float() / 255.0)

        # Actions: [steering, speed]
        actions = torch.Tensor(data[:2])

        # Locations: compute delta to next waypoint
        # data format: [steering, speed, throttle, cur_x, cur_y, next_x, next_y]
        cur_x, cur_y = data[3], data[4]
        next_x, next_y = data[5], data[6]
        location_delta = torch.Tensor([next_x - cur_x, next_y - cur_y])

        return (normalized_image, actions, location_delta)

def get_dataloader(data_dir, batch_size, num_workers=4, img_folder=None, crop=None):
    """
    Create a DataLoader for CARLA dataset.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        img_folder: For UE5 format, specify image subfolder (e.g., 'rgb_up')
        crop: Optional (y1, y2, x1, x2) crop tuple

    Examples:
        # Legacy format (UE4)
        loader = get_dataloader('./data/', batch_size=8)

        # UE5 format
        loader = get_dataloader('./_output_2024-01-01/', batch_size=8, img_folder='rgb_up')
    """
    return torch.utils.data.DataLoader(
        CarlaDataset(data_dir, img_folder=img_folder, crop=crop),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )