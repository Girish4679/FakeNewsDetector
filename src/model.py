"""
Multimodal Fake News Detection Model — CLIP backbone.

CLIP (Contrastive Language-Image Pretraining) is a pretrained model from
OpenAI, available off the shelf via HuggingFace. It was trained on 400M
image-text pairs to learn whether an image and caption match — which is
exactly the signal we need for fake news detection.

We use it as a frozen (or lightly fine-tuned) feature extractor, then
train a small fusion head on top.

Architecture:
    Text  → CLIP text encoder  → 512-dim embedding (L2-normalized)
    Image → CLIP image encoder → 512-dim embedding (L2-normalized)
    cosine_similarity(text, image) → scalar alignment score

    Fusion (concat baseline):
        [text_emb || image_emb || cos_sim] → MLP → binary logit

Fine-tuning strategy:
    By default, ALL of CLIP is frozen and only the fusion MLP is trained.
    This is fast and surprisingly effective because CLIP features are
    already alignment-aware. Set freeze_clip=False to also fine-tune
    the last 3 layers of both encoders for extra task-specific adaptation.

Ablations (swap via constructor args):
    fusion="concat"         → concatenation + cosine sim (default, implemented)
    fusion="cross_attention"→ cross-modal attention (TODO)
    fusion="gated"          → gated fusion (TODO)

    clip_model_name="openai/clip-vit-base-patch32"  → faster, less VRAM
    clip_model_name="openai/clip-vit-base-patch16"  → better, matches proposal (default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


CLIP_EMBED_DIM = 512   # output dim for both encoders in CLIP ViT-B variants
N_FINE_TUNE_LAYERS = 3  # how many tail layers to unfreeze when freeze_clip=False


# ---------------------------------------------------------------------------
# CLIP Encoder (shared backbone for text + image)
# ---------------------------------------------------------------------------

class CLIPEncoder(nn.Module):
    """
    Wraps HuggingFace CLIPModel to extract:
        - text embeddings  (B, 512)  from tokenized input
        - image embeddings (B, 512)  from pixel tensors

    Both outputs are L2-normalized (unit vectors), so cosine similarity
    between them is just a dot product.

    Args:
        model_name:   HuggingFace model ID.
                        "openai/clip-vit-base-patch16"  ← default, better
                        "openai/clip-vit-base-patch32"  ← faster, less VRAM
        freeze_clip:  If True (default), freeze all CLIP weights.
                      If False, freeze early layers but fine-tune the last
                      N_FINE_TUNE_LAYERS of both text and vision transformers.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)

        if freeze_clip:
            # Freeze everything — only the fusion MLP will be trained
            for param in self.clip.parameters():
                param.requires_grad = False
        else:
            # Freeze everything first, then selectively unfreeze tail layers
            for param in self.clip.parameters():
                param.requires_grad = False

            # Unfreeze last N layers of the text transformer
            text_layers = self.clip.text_model.encoder.layers
            for layer in text_layers[-N_FINE_TUNE_LAYERS:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Always unfreeze text final layer norm + projection
            for param in self.clip.text_model.final_layer_norm.parameters():
                param.requires_grad = True
            for param in self.clip.text_projection.parameters():
                param.requires_grad = True

            # Unfreeze last N layers of the vision transformer
            vision_layers = self.clip.vision_model.encoder.layers
            for layer in vision_layers[-N_FINE_TUNE_LAYERS:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Always unfreeze vision final layer norm + projection
            for param in self.clip.vision_model.post_layernorm.parameters():
                param.requires_grad = True
            for param in self.clip.visual_projection.parameters():
                param.requires_grad = True

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.normalize(features, dim=-1)   # (B, 512)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(features, dim=-1)   # (B, 512)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """
    Concatenation baseline.

    Input:  text_emb (B, 512), image_emb (B, 512)
    Builds: [text_emb || image_emb || cos_sim] → (B, 1025)
    Output: logit (B,)

    The cosine similarity is included as an explicit alignment feature —
    this is the core signal CLIP was trained to produce. A large gap
    between text and image embeddings → high fake probability.
    """

    def __init__(self, embed_dim: int = CLIP_EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        in_dim = embed_dim * 2 + 1  # text + image + cosine_sim scalar
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        cos_sim = (text_emb * image_emb).sum(dim=-1, keepdim=True)  # (B, 1) dot product of unit vecs
        fused = torch.cat([text_emb, image_emb, cos_sim], dim=-1)   # (B, 1025)
        return self.mlp(fused).squeeze(-1)                           # (B,)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class FakeNewsDetector(nn.Module):
    """
    Full multimodal fake news detection model.

    Args:
        clip_model_name: HuggingFace CLIP variant.
                           "openai/clip-vit-base-patch16"  (default, better)
                           "openai/clip-vit-base-patch32"  (faster, less VRAM)
        fusion:          Fusion strategy. Currently: "concat".
        freeze_clip:     If True (default), freeze all CLIP weights and only
                         train the fusion MLP. If False, also fine-tune the
                         last 3 layers of both CLIP encoders.
        dropout:         Dropout in the classifier MLP (default: 0.3).

    Example:
        model = FakeNewsDetector().cuda()
        logits = model(input_ids, attention_mask, pixel_values)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        fusion: str = "concat",
        freeze_clip: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = CLIPEncoder(model_name=clip_model_name, freeze_clip=freeze_clip)

        if fusion == "concat":
            self.fusion = ConcatFusion(dropout=dropout)
        else:
            raise NotImplementedError(
                f"Fusion '{fusion}' not yet implemented. Available: 'concat'"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        text_emb  = self.encoder.encode_text(input_ids, attention_mask)  # (B, 512)
        image_emb = self.encoder.encode_image(pixel_values)              # (B, 512)
        return self.fusion(text_emb, image_emb)                          # (B,)


# ---------------------------------------------------------------------------
# Quick sanity check — run with: python src/model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading CLIP (this downloads weights on first run ~600MB)...")
    model = FakeNewsDetector(freeze_clip=True)
    model.eval()

    B = 4
    # CLIP tokenizer produces sequences up to length 77
    input_ids      = torch.randint(0, 49408, (B, 77))
    attention_mask = torch.ones(B, 77, dtype=torch.long)
    pixel_values   = torch.randn(B, 3, 224, 224)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, pixel_values)
        probs  = torch.sigmoid(logits)

    print(f"Logits: {logits}")
    print(f"Probs:  {probs}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")
    print("\nExpected: ~150M total, ~200K trainable (frozen CLIP, MLP only)")
