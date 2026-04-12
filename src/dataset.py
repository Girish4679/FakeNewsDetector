"""
PyTorch Dataset for Fakeddit multimodal classification.

Each sample returns:
    input_ids       : (seq_len,)  tokenized text
    attention_mask  : (seq_len,)
    image           : (3, H, W)   normalized image tensor
    label           : int         2-way label (0 = real, 1 = fake)

Rows whose image file is not present on disk are silently skipped so the
dataset works regardless of how many images have been downloaded so far.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


# Default image transform matches ViT-B/16 and ResNet-50 expectations
DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FakedditDataset(Dataset):
    """
    Args:
        tsv_path:         Path to a Fakeddit split TSV (train/validate/test).
        image_dir:        Directory containing downloaded images named <id>.jpg
        tokenizer_name:   HuggingFace tokenizer identifier (default: roberta-base)
        max_length:       Max token sequence length for the tokenizer
        label_col:        Which label column to use: "2_way_label", "3_way_label", or "6_way_label"
        image_transform:  torchvision transform applied to each image
        require_image:    If True, skip rows without a downloaded image.
                          If False, return a blank image tensor for missing images.
    """

    def __init__(
        self,
        tsv_path: str,
        image_dir: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 128,
        label_col: str = "2_way_label",
        image_transform: Optional[transforms.Compose] = None,
        require_image: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.label_col = label_col
        self.image_transform = image_transform or DEFAULT_IMAGE_TRANSFORM
        self.require_image = require_image

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        df = pd.read_csv(tsv_path, sep="\t")
        df = df.fillna("")
        df = df[df["hasImage"] == True]
        df = df[df["image_url"] != ""]

        if require_image:
            available = {p.stem for p in self.image_dir.glob("*.jpg")}
            df = df[df["id"].astype(str).isin(available)]

        df = df.reset_index(drop=True)
        self.df = df

        print(f"FakedditDataset loaded {len(self.df):,} samples from {tsv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["clean_title"])
        label = int(row[self.label_col])
        img_id = str(row["id"])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Load image
        img_path = self.image_dir / f"{img_id}.jpg"
        image = self._load_image(img_path)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                return self.image_transform(img)
            except (UnidentifiedImageError, OSError):
                pass
        # Return blank image if file is missing or corrupt
        return torch.zeros(3, 224, 224)
