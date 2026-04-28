"""
Evaluation script — loads a saved checkpoint and runs it on the test set.

Prints F1, AUC-ROC, accuracy, and a confusion matrix.

Usage:
    python src/evaluate.py \
        --checkpoint "G:/My Drive/fakeddit/checkpoints/checkpoint_best.pt" \
        --data_dir   "G:/My Drive/fakeddit/data" \
        --image_dir  "G:/My Drive/fakeddit/images"

    # Evaluate on validate split instead of test
    python src/evaluate.py \
        --checkpoint checkpoints/checkpoint_best.pt \
        --data_dir data --image_dir images \
        --split validate
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import FakedditDataset
from src.model import FakeNewsDetector
from torch.utils.data import DataLoader


SPLIT_FILES = {
    "train":    "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test":     "multimodal_test_public.tsv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint on Fakeddit")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--split",      default="test", choices=["train", "validate", "test"])
    p.add_argument("--label_col",  default="2_way_label")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ----- Load checkpoint -----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint from epoch {ckpt['epoch']}")
    if ckpt.get("metrics"):
        m = ckpt["metrics"]
        print(f"  Saved val metrics — F1: {m.get('f1', 'n/a'):.4f}  AUC: {m.get('auc_roc', 'n/a'):.4f}\n")

    # ----- Rebuild model -----
    model = FakeNewsDetector(clip_model_name=args.clip_model)
    state_dict = {k: v for k, v in ckpt["model"].items() if not k.startswith("loss_fn")}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ----- Dataset -----
    tsv_path = Path(args.data_dir) / SPLIT_FILES[args.split]
    dataset  = FakedditDataset(
        tsv_path=str(tsv_path),
        image_dir=args.image_dir,
        clip_model_name=args.clip_model,
        label_col=args.label_col,
        require_image=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # ----- Inference -----
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask, pixel_values)

            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    # ----- Metrics -----
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs > 0.5).astype(int)

    f1  = f1_score(all_labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, probs)
    cm  = confusion_matrix(all_labels, preds)

    print(f"Results on {args.split} split ({len(all_labels):,} samples)")
    print(f"{'─'*40}")
    print(f"  F1-score:  {f1:.4f}   (primary metric)")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred Real  Pred Fake")
    print(f"  Actual Real   {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual Fake   {cm[1][0]:>6}     {cm[1][1]:>6}")
    print(f"\nClassification report:")
    print(classification_report(all_labels, preds, target_names=["Real", "Fake"]))


if __name__ == "__main__":
    main()
