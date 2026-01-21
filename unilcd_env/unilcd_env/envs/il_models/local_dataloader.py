import glob

import numpy as np

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2
import os


class CarlaDatasetLocal(Dataset):
    """
    Dataset for local model training. Resizes images to 96x96.
    Supports both legacy (UE4) and new (UE5) folder formats.
    """

    def __init__(self, data_dir, img_folder=None, crop=None):
        """
        Args:
            data_dir: Root data directory
            img_folder: Subfolder for images (e.g., 'rgb_up'). If None, uses legacy format.
            crop: Tuple of (y1, y2, x1, x2) for cropping before resize.
        """
        self.data_dir = data_dir
        self.img_folder = img_folder
        self.crop = crop

        # Detect format and load file lists
        images_dir = os.path.join(data_dir, 'Images')
        info_dir = os.path.join(data_dir, 'Info')

        if os.path.isdir(images_dir) and os.path.isdir(info_dir):
            # UniLCD format (Images/ and Info/ subfolders)
            self.img_list = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
            if not self.img_list:
                self.img_list = sorted(glob.glob(os.path.join(images_dir, '*.png')))
            self.data_list = sorted(glob.glob(os.path.join(info_dir, '*.npy')))
            self.format = 'unilcd'
        elif img_folder is not None:
            # New UE5 format
            img_path = os.path.join(data_dir, img_folder)
            action_path = os.path.join(data_dir, 'actions')
            self.img_list = sorted(glob.glob(os.path.join(img_path, '*.png')))
            self.data_list = sorted(glob.glob(os.path.join(action_path, '*.npy')))
            self.format = 'ue5'
        else:
            # Legacy format
            self.img_list = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
            if not self.img_list:
                self.img_list = sorted(glob.glob(os.path.join(data_dir, '*.png')))
            self.data_list = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
            self.format = 'legacy'

        # Verify matching counts
        if len(self.img_list) != len(self.data_list):
            print(f"Warning: Image count ({len(self.img_list)}) != Action count ({len(self.data_list)})")
            min_count = min(len(self.img_list), len(self.data_list))
            self.img_list = self.img_list[:min_count]
            self.data_list = self.data_list[:min_count]

        self.resize = transforms.Resize((96, 96), antialias=True)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"CarlaDatasetLocal: Loaded {len(self.data_list)} samples ({self.format} format)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action.

        Returns:
            (image, actions, location_delta) as torch.Tensor
            - image: normalized 96x96 RGB tensor
            - actions: [steering, speed]
            - location_delta: [delta_x, delta_y] to next waypoint
        """
        # Load action data
        data = np.load(self.data_list[idx], allow_pickle=True)

        # Load image
        img = read_image(self.img_list[idx])
        img = img[:3]  # Ensure RGB only

        # Apply crop
        if self.crop is not None:
            y1, y2, x1, x2 = self.crop
            img = img[:, y1:y2, x1:x2]
        elif self.format == 'legacy':
            # Default legacy crop (center crop for 640x320)
            img = img[:, :, 80:560]
        # For UE5/unilcd format without explicit crop, use full image

        # Resize to 96x96 for local model
        img = self.resize(img)

        # Normalize
        normalized_image = self.normalize(img.float() / 255.0)

        # Actions: [steering, speed]
        actions = torch.Tensor(data[:2])

        # Locations: compute delta to next waypoint
        cur_x, cur_y = data[3], data[4]
        next_x, next_y = data[5], data[6]
        location_delta = torch.Tensor([next_x - cur_x, next_y - cur_y])

        return (normalized_image, actions, location_delta)


def get_dataloader(data_dir, batch_size, num_workers=4, img_folder=None, crop=None):
    """
    Create a DataLoader for local model training.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        img_folder: For UE5 format, specify image subfolder (e.g., 'rgb_up')
        crop: Optional (y1, y2, x1, x2) crop tuple
    """
    return torch.utils.data.DataLoader(
        CarlaDatasetLocal(data_dir, img_folder=img_folder, crop=crop),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )