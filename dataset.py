"""
dataset.py — PolypDataset
=========================
PURPOSE: Teach PyTorch HOW to load our colonoscopy images and masks.

PyTorch needs a Dataset class with three methods:
  __init__  → set up paths and transforms
  __len__   → tell PyTorch how many samples exist
  __getitem__ → load one image+mask pair by index

Think of it like a librarian: the DataLoader asks "give me sample 42"
and the Dataset opens the right files and returns them as tensors.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation pipelines ──────────────────────────────────────────────────
# Augmentation creates fake variety so the model doesn't just memorize
# the training images. We apply the SAME transform to image AND mask.

def get_train_transforms(img_size=256):
    """
    Training transforms: aggressive augmentation for generalization.
    Albumentations applies the same spatial transforms to image AND mask
    simultaneously, which is exactly what we need for segmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Spatial transforms — flip/rotate the whole image+mask together
        A.HorizontalFlip(p=0.5),          # mirror 50% of the time
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(                # random zoom + tilt
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),

        # Color transforms — only applied to image, not mask
        A.ColorJitter(                     # vary brightness/contrast/color
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),              # add random pixel noise

        # Convert to PyTorch tensor and normalize pixel values to ~[-1, 1]
        # These mean/std values match what ResNet was trained on (ImageNet)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),   # converts numpy HWC → PyTorch CHW tensor
    ])


def get_val_transforms(img_size=256):
    """
    Validation transforms: NO augmentation, just resize + normalize.
    We want clean measurements of model performance, not augmented ones.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


# ── Dataset class ────────────────────────────────────────────────────────────

class PolypDataset(Dataset):
    """
    Loads colonoscopy image + binary mask pairs.

    Folder structure expected:
        data/kvasir-seg/
            images/   ← colonoscopy photos (.jpg or .png)
            masks/    ← binary masks (white=polyp, black=background)

    Each image file must have a MATCHING file in masks/ with the same name.
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir   : path to folder containing colonoscopy images
            mask_dir  : path to folder containing binary masks
            transform : albumentations transform pipeline (train or val)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get sorted list of filenames so image[i] matches mask[i]
        self.filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        print(f"  Dataset loaded: {len(self.filenames)} samples from {img_dir}")

    def __len__(self):
        """How many samples are in this dataset."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Load one image + mask pair by index.

        Returns:
            image : float32 tensor of shape (3, H, W)  — RGB colonoscopy frame
            mask  : float32 tensor of shape (1, H, W)  — 0.0=background, 1.0=polyp
        """
        fname = self.filenames[idx]

        # ── Load image ──
        img_path = os.path.join(self.img_dir, fname)
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        # shape: (H, W, 3)  values: 0-255

        # ── Load mask ──
        mask_path = os.path.join(self.mask_dir, fname)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # shape: (H, W)  values: 0-255

        # Binarize mask: any pixel > 127 = polyp (1), rest = background (0)
        mask = (mask > 127).astype(np.float32)
        # shape: (H, W)  values: 0.0 or 1.0

        # ── Apply transforms ──
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # now a PyTorch tensor (3, H, W)
            mask = augmented["mask"]     # now a PyTorch tensor (H, W)

        # Add channel dimension to mask: (H, W) → (1, H, W)
        # The model outputs (batch, 1, H, W) so mask needs to match
        mask = mask.unsqueeze(0)

        return image, mask


# ── DataLoader factory ───────────────────────────────────────────────────────

def create_dataloaders(data_root, img_size=256, batch_size=8, val_split=0.2):
    """
    Build train and validation DataLoaders from a single image folder.

    Args:
        data_root  : path to 'kvasir-seg' folder (contains images/ and masks/)
        img_size   : resize images to this square size
        batch_size : how many images per batch (reduce if you run out of memory)
        val_split  : fraction of data to use for validation (0.2 = 20%)

    Returns:
        train_loader, val_loader
    """
    img_dir  = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    # Load full dataset to get the list of filenames
    all_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Split filenames into train and validation sets
    n_val   = int(len(all_files) * val_split)
    n_train = len(all_files) - n_val

    # Use first n_train for training, last n_val for validation
    # In real projects: use random_split or sklearn train_test_split
    train_files = all_files[:n_train]
    val_files   = all_files[n_train:]

    print(f"Split: {n_train} train / {n_val} val")

    # Create temporary filtered datasets by subsetting the file lists
    # We'll do this by creating minimal dataset wrappers
    train_dataset = _SubsetPolypDataset(
        img_dir, mask_dir, train_files,
        transform=get_train_transforms(img_size)
    )
    val_dataset = _SubsetPolypDataset(
        img_dir, mask_dir, val_files,
        transform=get_val_transforms(img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle training data each epoch
        num_workers=0,          # 0 = load on main thread (safe for Windows too)
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # never shuffle validation
        num_workers=0,
    )

    return train_loader, val_loader


class _SubsetPolypDataset(Dataset):
    """Internal helper: PolypDataset that operates on a specific file list."""
    def __init__(self, img_dir, mask_dir, filenames, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = filenames
        self.transform = transform
        print(f"  Subset dataset: {len(filenames)} samples")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"), dtype=np.uint8)
        mask  = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"), dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]
        return image, mask.unsqueeze(0)
