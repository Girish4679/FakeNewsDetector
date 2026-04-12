"""
PyTorch Dataset for Fakeddit multimodal classification.

Uses CLIP's processor to handle both text tokenization and image preprocessing,
so both modalities are in exactly the format CLIP expects.

Each sample returns:
    input_ids       : (77,)        CLIP-tokenized text (max 77 tokens)
    attention_mask  : (77,)
    pixel_values    : (3, 224, 224) CLIP-normalized image tensor
    label           : int           2-way label (0 = real, 1 = fake)

Rows whose image file is not present on disk are silently skipped.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import CLIPProcessor


BLANK_IMAGE = Image.new("RGB", (224, 224), color=0)  # fallback for missing images


class FakedditDataset(Dataset):
    """
    Args:
        tsv_path:       Path to a Fakeddit split TSV (train/validate/test).
        image_dir:      Directory containing downloaded images named <id>.jpg
        clip_model_name HuggingFace CLIP model name — must match the model in model.py.
        label_col:      Which label column to use: "2_way_label", "3_way_label", or "6_way_label"
        require_image:  If True (default), skip rows without a downloaded image.
                        If False, substitute a blank image for missing files.
    """

    def __init__(
        self,
        tsv_path: str,
        image_dir: str,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        label_col: str = "2_way_label",
        require_image: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.label_col = label_col

        # CLIPProcessor handles both tokenization AND image preprocessing in one call
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        df = pd.read_csv(tsv_path, sep="\t")
        df = df.fillna("")
        df = df[df["hasImage"] == True]
        df = df[df["image_url"] != ""]

        if require_image:
            available = {p.stem for p in self.image_dir.glob("*.jpg")}
            df = df[df["id"].astype(str).isin(available)]

        self.df = df.reset_index(drop=True)
        print(f"FakedditDataset: {len(self.df):,} samples  ({tsv_path})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row    = self.df.iloc[idx]
        text   = str(row["clean_title"])
        label  = int(row[self.label_col])
        img_id = str(row["id"])

        image = self._load_image(self.image_dir / f"{img_id}.jpg")

        # CLIPProcessor tokenizes text (max 77 tokens) and normalizes the image
        encoded = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),       # (77,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (77,)
            "pixel_values":   encoded["pixel_values"].squeeze(0),    # (3, 224, 224)
            "label":          torch.tensor(label, dtype=torch.long),
        }

    def _load_image(self, path: Path) -> Image.Image:
        if path.exists():
            try:
                return Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                pass
        return BLANK_IMAGE
