"""
PyTorch Dataset for digital clock images (Stage 1)
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path


class DigitalClockDataset(Dataset):
    """Dataset for digital clock images."""

    def __init__(
        self,
        data_dir,
        split='train',
        transform=None,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225]
    ):
        """
        Args:
            data_dir: Directory containing images and metadata.json
            split: 'train', 'val', or 'test'
            transform: Optional transforms for data augmentation
            normalize_mean, normalize_std: Normalization parameters
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Split data by time_index (not by image, to prevent data leakage)
        unique_time_indices = sorted(set(item['time_index'] for item in self.metadata))
        np.random.seed(42)
        np.random.shuffle(unique_time_indices)

        n_train = int(0.7 * len(unique_time_indices))
        n_val = int(0.15 * len(unique_time_indices))

        if split == 'train':
            valid_indices = set(unique_time_indices[:n_train])
        elif split == 'val':
            valid_indices = set(unique_time_indices[n_train:n_train + n_val])
        else:  # test
            valid_indices = set(unique_time_indices[n_train + n_val:])

        # Filter metadata for this split
        self.data = [item for item in self.metadata if item['time_index'] in valid_indices]

        print(f"✓ {split} split: {len(self.data)} images from {len(valid_indices)} unique times")

        # Setup transforms
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=normalize_mean, std=normalize_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=normalize_mean, std=normalize_std)
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, H, W)
            time: Tensor of shape (3,) with normalized (hour, minute, second)
        """
        item = self.data[idx]

        # Load image
        img_path = self.data_dir / item['filename']
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        # Get time (normalized to [0, 1])
        hour = item['hour'] / 23.0
        minute = item['minute'] / 59.0
        second = item['second'] / 59.0
        time = torch.tensor([hour, minute, second], dtype=torch.float32)

        return image, time


def get_digital_dataloaders(config_dict, batch_size=32, num_workers=4):
    """
    Create train/val/test dataloaders for digital clocks.

    Args:
        config_dict: Configuration dictionary
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    from torch.utils.data import DataLoader
    import config

    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataset = DigitalClockDataset(
            data_dir=config.DIGITAL_CLOCKS_DIR,
            split=split,
            normalize_mean=config_dict.get('normalize_mean'),
            normalize_std=config_dict.get('normalize_std')
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=getattr(config, "PIN_MEMORY", False)
        )

    return dataloaders
