"""
DataLoader setup for Fakeddit training.

All images are read directly from the local Google Drive folder —
no special Drive API needed. Google Drive for Desktop makes the
folder look like any normal local directory.

Usage:
    from src.dataloader import get_dataloaders

    loaders = get_dataloaders(
        data_dir="G:/My Drive/fakeddit/data",
        image_dir="G:/My Drive/fakeddit/images",
    )
    train_loader = loaders["train"]
    val_loader   = loaders["validate"]
    test_loader  = loaders["test"]

    for batch in train_loader:
        input_ids      = batch["input_ids"]       # (B, 77)
        attention_mask = batch["attention_mask"]   # (B, 77)
        pixel_values   = batch["pixel_values"]     # (B, 3, 224, 224)
        labels         = batch["label"]            # (B,)
"""

import os
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from src.dataset import FakedditDataset


# ---------------------------------------------------------------------------
# Batch size guide
# ---------------------------------------------------------------------------
# These are safe starting points. If you hit CUDA OOM, halve the batch size.
# If training is fast and VRAM usage is low, try doubling it.
#
#   GPU VRAM      Recommended batch_size
#   8 GB          16
#   12 GB         32   <-- default
#   16 GB         48-64
#   24 GB+        64-128
#
# num_workers: CPU processes that load/preprocess data while the GPU trains.
#   Windows: 0 (Windows multiprocessing has overhead that often makes it slower)
#   Mac/Linux: 4
#
# pin_memory: True speeds up CPU→GPU transfer. Always use when training on GPU.
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 4


def get_dataloaders(
    data_dir: str,
    image_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    label_col: str = "2_way_label",
    subset: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build and return DataLoaders for all three splits.

    Args:
        data_dir:         Path to folder containing the TSV files.
                            Mac:     '/Users/you/Library/CloudStorage/GoogleDrive-.../My Drive/fakeddit/data'
                            Windows: 'G:/My Drive/fakeddit/data'
        image_dir:        Path to folder containing downloaded .jpg images.
                            Mac:     '/Users/you/Library/CloudStorage/GoogleDrive-.../My Drive/fakeddit/images'
                            Windows: 'G:/My Drive/fakeddit/images'
        batch_size:       Samples per batch. Start at 32; halve if you get CUDA OOM.
        num_workers:      CPU workers for data loading. 0 on Windows, 4 on Mac/Linux.
        clip_model_name:  Must match the CLIP variant used in model.py.
        label_col:        Which label to use: '2_way_label', '3_way_label', or '6_way_label'.
        subset:           If True, load subset_*.tsv files (small dev set, no images needed).

    Returns:
        Dict with keys 'train', 'validate', 'test', each a DataLoader.
    """
    data_dir = Path(data_dir)

    if subset:
        split_files = {
            "train":    data_dir / "subset_train.tsv",
            "validate": data_dir / "subset_validate.tsv",
            "test":     data_dir / "subset_test.tsv",
        }
    else:
        split_files = {
            "train":    data_dir / "multimodal_train.tsv",
            "validate": data_dir / "multimodal_validate.tsv",
            "test":     data_dir / "multimodal_test_public.tsv",
        }

    loaders = {}
    for split, tsv_path in split_files.items():
        if not tsv_path.exists():
            print(f"  Skipping {split} — {tsv_path} not found")
            continue

        dataset = FakedditDataset(
            tsv_path=str(tsv_path),
            image_dir=image_dir,
            clip_model_name=clip_model_name,
            label_col=label_col,
            require_image=not subset,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=(split == "train"),
        )

        print(f"  {split}: {len(dataset):,} samples → {len(loaders[split]):,} batches of {batch_size}")

    return loaders
