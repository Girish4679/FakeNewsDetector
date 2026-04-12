"""
Training script for Fakeddit multimodal fake news detection.

Key features:
    - Mixed precision training (fp16) — halves VRAM usage, ~2x faster on modern GPUs
    - Gradient accumulation — simulates larger batch sizes without extra VRAM
    - AdamW optimizer with separate LRs for CLIP vs fusion head
    - Cosine LR schedule with linear warmup
    - Weighted loss to handle class imbalance
    - Saves best checkpoint (by val F1) to checkpoints/

Usage:
    # Frozen CLIP — only trains the fusion MLP (~200K params, fast)
    python src/train.py \
        --data_dir "G:/My Drive/fakeddit/data" \
        --image_dir "G:/My Drive/fakeddit/images" \
        --checkpoint_dir "G:/My Drive/fakeddit/checkpoints"

    # Fine-tune last 3 CLIP layers too (~10M params, slower but better)
    python src/train.py \
        --data_dir "G:/My Drive/fakeddit/data" \
        --image_dir "G:/My Drive/fakeddit/images" \
        --checkpoint_dir "G:/My Drive/fakeddit/checkpoints" \
        --unfreeze_clip

    # Quick smoke test on the small dev subset (no images needed)
    python src/train.py \
        --data_dir data \
        --image_dir images \
        --checkpoint_dir checkpoints \
        --subset \
        --epochs 2 \
        --batch_size 8

IMPORTANT (Windows): Always run this script via:
    python src/train.py ...
from the project root, not from inside src/.
The DataLoader requires the __main__ guard below to work correctly
on Windows with num_workers > 0.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Make sure imports work when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataloader import get_dataloaders
from src.model import FakeNewsDetector


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Fakeddit fake news detector")

    # Paths
    p.add_argument("--data_dir",       required=True,  help="Folder containing TSV files")
    p.add_argument("--image_dir",      required=True,  help="Folder containing downloaded images")
    p.add_argument("--checkpoint_dir", default="checkpoints", help="Where to save model checkpoints")

    # Data
    p.add_argument("--subset",      action="store_true", help="Use subset_*.tsv for quick dev runs")
    p.add_argument("--label_col",   default="2_way_label", choices=["2_way_label", "3_way_label", "6_way_label"])
    p.add_argument("--clip_model",  default="openai/clip-vit-base-patch16")

    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=32,  help="Per-GPU batch size. Halve if CUDA OOM.")
    p.add_argument("--accum_steps",  type=int,   default=1,   help="Gradient accumulation steps. Effective batch = batch_size * accum_steps")
    p.add_argument("--num_workers",  type=int,   default=None,  help="DataLoader workers. Defaults to 0 on Windows, 4 on Mac/Linux")

    # Model
    p.add_argument("--unfreeze_clip", action="store_true", help="Fine-tune last 3 CLIP layers in addition to fusion head")
    p.add_argument("--dropout",       type=float, default=0.3)

    # Optimizer
    p.add_argument("--lr_head",  type=float, default=1e-3, help="LR for fusion MLP (default: 1e-3)")
    p.add_argument("--lr_clip",  type=float, default=1e-5, help="LR for unfrozen CLIP layers (default: 1e-5, ignored if --unfreeze_clip not set)")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, logits):
    """Compute F1, AUC-ROC, and accuracy from raw logits."""
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  # only one class present in batch

    return {"f1": f1, "accuracy": acc, "auc_roc": auc}


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, accum_steps, epoch):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"].to(device).float()

        # Mixed precision forward pass
        with autocast():
            logits = model(input_ids, attention_mask, pixel_values)
            loss   = model.loss_fn(logits, labels)
            loss   = loss / accum_steps  # scale loss for accumulation

        scaler.scale(loss).backward()

        # Only update weights every accum_steps batches
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        all_logits.extend(logits.detach().cpu().tolist())
        all_labels.extend(batch["label"].tolist())

        if (step + 1) % 50 == 0:
            print(f"  Epoch {epoch} | step {step+1}/{len(loader)} | loss {total_loss/(step+1):.4f}")

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["label"].to(device).float()

        with autocast():
            logits = model(input_ids, attention_mask, pixel_values)
            loss   = model.loss_fn(logits, labels)

        total_loss += loss.item()
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, tag="best"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{tag}.pt"
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)
    print(f"  Saved checkpoint → {path}  (val F1: {metrics['f1']:.4f})")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {checkpoint_path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["metrics"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Data -----
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    print("\nBuilding dataloaders...")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        clip_model_name=args.clip_model,
        label_col=args.label_col,
        subset=args.subset,
    )

    if "train" not in loaders:
        print("ERROR: train split not found. Check --data_dir and --subset flag.")
        sys.exit(1)

    # ----- Class imbalance weight -----
    # Count positive labels in training set to weight the loss
    train_labels = loaders["train"].dataset.df[args.label_col].values
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"\nClass balance — real: {n_neg:,}  fake: {n_pos:,}  pos_weight: {pos_weight.item():.2f}")

    # ----- Model -----
    print("\nBuilding model...")
    model = FakeNewsDetector(
        clip_model_name=args.clip_model,
        fusion="concat",
        freeze_clip=not args.unfreeze_clip,
        dropout=args.dropout,
    )
    # Attach weighted loss to the model so train/eval loops can access it
    model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

    # ----- Optimizer -----
    # Separate parameter groups so CLIP layers get a lower LR than the fusion head
    clip_params   = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params   = list(model.fusion.parameters())

    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if clip_params:
        param_groups.append({"params": clip_params, "lr": args.lr_clip})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ----- LR Scheduler -----
    total_steps   = len(loaders["train"]) * args.epochs // args.accum_steps
    warmup_steps  = min(args.warmup_steps, total_steps // 10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----- Mixed precision scaler -----
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ----- Resume if checkpoint exists -----
    checkpoint_dir = Path(args.checkpoint_dir)
    resume_path = checkpoint_dir / "checkpoint_latest.pt"
    start_epoch = 1
    best_val_f1 = 0.0

    if resume_path.exists():
        start_epoch, prev_metrics = load_checkpoint(model, optimizer, resume_path, device)
        best_val_f1 = prev_metrics.get("f1", 0.0)
        start_epoch += 1

    # ----- Training loop -----
    print(f"\nTraining for {args.epochs} epochs (effective batch size: {args.batch_size * args.accum_steps})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            scaler, device, args.accum_steps, epoch
        )

        val_metrics = {}
        if "validate" in loaders:
            val_metrics = evaluate(model, loaders["validate"], device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) | "
            f"train loss {train_metrics['loss']:.4f}  F1 {train_metrics['f1']:.4f} | "
            + (f"val loss {val_metrics['loss']:.4f}  F1 {val_metrics['f1']:.4f}  AUC {val_metrics['auc_roc']:.4f}"
               if val_metrics else "no val split")
        )

        # Save latest checkpoint every epoch (safe to resume from)
        save_checkpoint(model, optimizer, epoch, val_metrics or train_metrics, checkpoint_dir, tag="latest")

        # Save best checkpoint by val F1
        val_f1 = val_metrics.get("f1", train_metrics["f1"])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, val_metrics or train_metrics, checkpoint_dir, tag="best")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'checkpoint_best.pt'}")


if __name__ == "__main__":
    main()
