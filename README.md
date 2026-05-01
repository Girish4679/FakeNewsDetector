# Beyond Text: A Multimodal Fake News Detector

**Rutgers CS 462 — Deep Learning Final Project**  
Girish Ranganathan, Rithvik Sriram, James Dequina, Armaan Shivansh, Arvid Eapen

---

## Overview

Most fake news detectors only read text. This project detects misinformation by looking at **both the post title and its image**, and specifically at whether they are semantically consistent with each other. A recycled or out-of-context image paired with a misleading headline is a common misinformation pattern that text-only models miss entirely.

We use [CLIP](https://openai.com/research/clip) (OpenAI's Contrastive Language-Image Pretraining model) as a frozen feature extractor. CLIP encodes both the title text and the image into the same 512-dimensional embedding space, making it possible to directly measure their semantic alignment via cosine similarity. A small fusion MLP trained on top of these embeddings predicts real (0) or fake (1).

**Model architecture:**
```
Post Title  →  CLIP Text Encoder  →  512-dim embedding ──┐
                                                          ├─→ cosine similarity
Post Image  →  CLIP Image Encoder →  512-dim embedding ──┘
                                                          ↓
                              [text_emb || image_emb || cos_sim]  (1025-dim)
                                                          ↓
                                               Fusion MLP (3 layers)
                                                          ↓
                                                   Real / Fake
```

~150M total parameters; only ~279K are trained (the fusion MLP). CLIP is fully frozen.

---

## Results

Trained on a balanced 12,000-sample subset of Fakeddit, evaluated on the held-out public test set:

| Metric | Score |
|---|---|
| Accuracy | **87.2%** |
| F1-score | **0.849** |
| AUC-ROC | **0.939** |
| Real precision / recall | 91% / 87% |
| Fake precision / recall | 82% / 88% |

---

## Project Structure

```
FakeNewsDetector/
├── fakenewsdetector(FINAL).ipynb   # Main notebook — full training + evaluation with outputs
├── src/
│   ├── model.py                    # FakeNewsDetector model, CLIPEncoder, ConcatFusion
│   ├── dataset.py                  # FakedditDataset (PyTorch Dataset)
│   ├── dataloader.py               # get_dataloaders() helper
│   ├── train.py                    # Training script (CLI)
│   ├── evaluate.py                 # Evaluation script (CLI)
│   ├── comment_agent.py            # LLM-based comment analysis agent (bonus feature)
│   └── comment_loader.py           # Loads Fakeddit comments.tsv for the agent
├── demo/
│   └── app.py                      # Gradio web demo
├── scripts/
│   └── make_subset.py              # Creates a small balanced subset for local dev
├── tests/
│   └── test_comment_agent.py       # Unit tests for the comment agent
├── data/
│   ├── multimodal_train.tsv        # Fakeddit training split metadata
│   ├── multimodal_validate.tsv     # Fakeddit validation split metadata
│   └── multimodal_test_public.tsv  # Fakeddit test split metadata
├── image_downloader.py             # Resumable concurrent image downloader
└── requirements.txt
```

---

## Setup

### Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `Pillow`, `scikit-learn`, `pandas`, `tqdm`, `gradio`, `anthropic`.

### Data

The Fakeddit TSV metadata files are already included in `data/`. You also need the post images.

**Download images** using the included downloader:
```bash
# Download up to 5,000 images from the training split
python image_downloader.py --split train --output_dir images --max_images 5000

# Download up to 50 GB worth of images
python image_downloader.py --split train --output_dir images --max_gb 50
```

The downloader is resumable — interrupt it with Ctrl+C and re-run. Already-downloaded images are skipped automatically via a manifest file.

---

## Running the Notebook (Recommended)

The primary artifact is **`fakenewsdetector(FINAL).ipynb`**, which is pre-run with all outputs visible. It was trained on Google Colab (Tesla T4 GPU) and contains the full pipeline: data prep, training, and evaluation.

To re-run it yourself on Colab:
1. Upload the project to Google Drive
2. Place the Fakeddit TSV files at `My Drive/fakeddit/data/`
3. Place downloaded images at `My Drive/fakeddit/images/`
4. Open the notebook in Colab and run cells top to bottom

---

## Running Training from the Command Line

For a quick smoke test on a small subset (no GPU required):
```bash
python scripts/make_subset.py --n 1000 --data_dir data

python src/train.py \
  --data_dir data \
  --image_dir images \
  --checkpoint_dir checkpoints \
  --subset \
  --epochs 3 \
  --batch_size 8
```

For a full training run:
```bash
python src/train.py \
  --data_dir data \
  --image_dir images \
  --checkpoint_dir checkpoints \
  --epochs 10 \
  --batch_size 32
```

Optional flags:
- `--unfreeze_clip` — also fine-tunes the last 3 layers of CLIP (slower, may improve accuracy)
- `--batch_size 16` — reduce if you get out-of-memory errors

Training saves two checkpoints to `--checkpoint_dir`:
- `checkpoint_best.pt` — best validation F1 seen so far
- `checkpoint_latest.pt` — end of last completed epoch (safe to resume from)

---

## Running Evaluation

```bash
python src/evaluate.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --data_dir data \
  --image_dir images
```

Prints F1, AUC-ROC, accuracy, a confusion matrix, and a full classification report.

---

## Bonus: Comment Agent

In addition to the CLIP model, we built an AI agent that reads the Reddit comments on a post and adjusts the model's confidence score at inference time. It uses the Anthropic API (Claude) to look for debunking language, fact-check links, and crowd skepticism.

To use it, set your API key:
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

Then run the interactive Gradio demo:
```bash
python demo/app.py
# Open http://localhost:7860
```

---

## Citation

```
@article{nakamura2019r,
    title={r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection},
    author={Nakamura, Kai and Levy, Sharon and Wang, William Yang},
    journal={arXiv preprint arXiv:1911.03854},
    year={2019}
}
```
