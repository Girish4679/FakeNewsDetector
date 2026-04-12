# Presentation Guide
### Beyond Text: A Multimodal Fusion Approach to Misinformation Detection

---

## General Tips Before You Start

- **Find the CLIP paper talk:** Search YouTube for "CLIP OpenAI paper explained" or "Learning Transferable Visual Models From Natural Language Supervision." There are great 5-minute explainers you can clip into your slides.
- **Pause points:** Good natural pauses are after the Method slide and after Results. Something like "Before we get into results, any questions on the architecture?"
- **Demo moment:** If you have a trained model, the single most impressive thing you can do is type in a real headline + show an image live and watch it predict. Professors love this.

---

## Slide 1: Problem + Input/Output

### What to say
Open with a real example — don't start with abstract definitions. Lead with something visceral.

**Hook (1 slide, maybe animated):**
> "In 2018, a photo of a crowd went viral claiming to show 'millions protesting in Brazil.' The crowd was actually from a 2013 concert. The image was real. The caption was fake. A text-only model would have no way to know."

Then define the task formally:
- **Input:** A social media post consisting of a text title `T` and an image `I`
- **Output:** Binary label — Real (0) or Fake/Misleading (1), with a confidence probability
- **The core challenge:** The lie isn't always in the text or the image alone — it's in the *mismatch between them*

### Visuals to include
1. **A real fake news example** — find one on Google Images or from the Fakeddit dataset itself. Look for posts where the image is clearly from a different event than the caption claims. Out-of-context images work best visually.
2. **Pipeline diagram** (already in APPROACH.md — redraw it cleanly):
   ```
   [Post Title Text] ──→ [Text Encoder] ──┐
                                           ├──→ [Fusion] ──→ Real / Fake?
   [Post Image]      ──→ [Image Encoder] ─┘
   ```
3. **"Why not text only?" mini-slide:** Show the same headline with two different images — one that makes it look real, one that makes it look fake. This makes the argument for multimodal instantly obvious.

### Numbers to mention
- Misinformation reach: mention something like "65% of misinformation on social media contains an image" (look up the actual stat — MIT Media Lab and Reuters Institute have good ones)
- Fakeddit scale: "over 560,000 Reddit posts, collected over 5 years across dozens of subreddits"

---

## Slide 2: Method

This is your longest section. Split into 2-3 sub-slides.

### Sub-slide 2a: Why CLIP?

**What to say:**
> "Most prior work uses separate text and image models that were never trained together. The problem is their embeddings live in completely different mathematical spaces — there's no meaningful way to compare them. We use CLIP, which was trained on 400 million image-caption pairs specifically to learn whether an image and a piece of text match. That's exactly our task."

Key point to emphasize: **CLIP's cosine similarity between text and image is literally a fake news signal out of the box.**

**Visuals:**
1. The classic CLIP training diagram from the original paper — it shows a grid of images and captions with the diagonal being matching pairs. Find it by Googling "CLIP paper figure 1 OpenAI." It's free to use and instantly explains contrastive training.
2. Show two examples:
   - Real post: image and headline are related → cosine similarity ≈ 0.8
   - Fake post: image and headline are unrelated → cosine similarity ≈ 0.1
3. **Screenshot** the HuggingFace model page for `openai/clip-vit-base-patch16` to show it's a real, widely-used pretrained model (legitimizes the approach)

**Reference to cite:**
> Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.

### Sub-slide 2b: Architecture

Walk through the architecture step by step:

```
Text title  ──→  CLIP Text Encoder (ViT-B/16 transformer)  ──→  512-dim embedding
                                                                        │
                                                              cosine similarity  ← this is the key signal
                                                                        │
Image       ──→  CLIP Image Encoder (ViT-B/16 patches)   ──→  512-dim embedding
                                                                        │
                                              [text_emb || image_emb || cos_sim]  (1025-dim)
                                                                        │
                                                               MLP Classifier
                                                                        │
                                                             sigmoid → probability
                                                                        │
                                                              Real (0) / Fake (1)
```

**Fine-tuning strategy:**
- CLIP is frozen (weights don't change) — we only train the fusion head
- ~150M total parameters, only ~200K are actually trained
- This is intentional: CLIP already understands image-text alignment; we just need to learn to classify on top of it

### Sub-slide 2c: Fusion Strategies (your research contribution)

This is the part your team owns. Present it as an empirical question:
> "We compared four ways of combining the text and image representations. Which one actually works best?"

| Strategy | What it does | Status |
|---|---|---|
| Concatenation | [text \|\| image \|\| cos_sim] → MLP | Baseline (done) |
| Cross-modal attention | Text queries attend over image patches | Research contribution |
| Gated fusion | Learned weight decides how much to trust each modality | Research contribution |
| Contrastive (CLIP similarity) | Cosine similarity alone as the feature | Ablation |

**Visual:** A side-by-side diagram of each fusion strategy. Even simple boxes-and-arrows work. The key is showing the conceptual difference — attention is the most visually interesting one.

---

## Slide 3: Dataset

### What to say
Introduce Fakeddit, then immediately show examples. Don't spend too long here — one solid slide.

### Stats to show
| Split | Posts | Images |
|---|---|---|
| Train | ~563,000 | downloaded subset |
| Validate | ~58,000 | downloaded subset |
| Test | ~58,000 | downloaded subset |
| **Total** | **~680,000** | |

- Sourced from Reddit across many subreddits (r/worldnews, r/politics, r/photoshopbattles, r/mildlyinteresting, etc.)
- Labels come from Reddit's community moderation — posts removed for misinformation are labeled fake
- 2-way label (real/fake), 3-way, and 6-way fine-grained labels available
- We focus on **2-way** (binary) as the primary task

### Visuals
1. **Show 4-6 real examples** from the dataset — two clearly real posts, two clearly fake, two ambiguous. Ambiguous ones are the most interesting to show because they motivate why this is hard.
2. **Label distribution pie chart** — is the dataset balanced? (Fakeddit is roughly balanced for 2-way labels)
3. **Screenshot** of the Fakeddit paper or GitHub page: Nakamura et al., "r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection" (LREC 2020)
4. Optional: a t-SNE or UMAP plot of CLIP embeddings colored by label (real vs fake) — if there's visible clustering, this alone makes a compelling visual argument for using CLIP

### Cite:
> Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection. LREC.

---

## Slide 4: Results

This is the most important slide. Split into numbers first, then case studies.

### Sub-slide 4a: Numbers

**Primary table — fusion strategy comparison (your ablation):**

| Model | F1 | AUC-ROC | Accuracy |
|---|---|---|---|
| Text only (CLIP text + MLP) | xx.x | x.xxx | xx.x% |
| Image only (CLIP image + MLP) | xx.x | x.xxx | xx.x% |
| Concat (text + image + cos_sim) | xx.x | x.xxx | xx.x% |
| Gated fusion | xx.x | x.xxx | xx.x% |
| Cross-modal attention | xx.x | x.xxx | xx.x% |

Fill these in with your actual results. The text-only and image-only baselines are important — they show that multimodal > unimodal, which motivates your whole project.

**Key result to highlight:**
- Does cross-modal attention beat simple concatenation? (This is your stated research question from the proposal)
- Does the cosine similarity feature alone do surprisingly well? (If it does, that's a great story about CLIP)

**Tip:** If cross-modal attention doesn't beat concat, that's still a valid result. Many papers find simple baselines are hard to beat. Frame it as: "Simple concatenation with CLIP similarity is a surprisingly strong baseline" — not as a failure.

### Sub-slide 4b: Case Studies (the part that gets called "fantastic")

Pick 3-4 specific posts from the test set and walk through them. Best mix:

**1. A clear win — model correctly flags a fake post:**
Show the image, the headline, the cosine similarity score (low), and the model's output probability (high fake). Explain *why* the image and text don't match.

**2. A subtler win — out-of-context image:**
This is where your model shines over text-only approaches. The headline text sounds perfectly reasonable, but the image is from a different event/year. Show that:
- Text-only model: predicts REAL (text sounds fine)
- Your model: predicts FAKE (low CLIP cosine similarity catches the mismatch)

**3. A failure case — model gets it wrong:**
Show a post the model confidently mislabels. Discuss why: is the image ambiguous? Is the text misleading? Does the domain matter? This shows intellectual honesty and depth of analysis.

**4. If you implement cross-modal attention — an attention visualization:**
Show which image patches the text attended to (or vice versa). If the headline says "flooding in Texas" and the attention highlights the background buildings rather than the water, that's a great failure mode to show. Tools like `BertViz` or a simple heatmap over the 14×14 image patches work well.

### Visual ideas
- Side-by-side: [image] | [headline] | [predicted probability bar] | [actual label]
- Cosine similarity histogram: distribution of cos_sim values for real vs fake posts (they should be separable)
- Confusion matrix heatmap (2x2 for binary)
- ROC curve comparing fusion strategies

---

## Slide 5: What You Learned

Be honest here. Professors respect candor much more than spin.

### Technical lessons

**What worked:**
- CLIP's pretrained cosine similarity is a stronger signal than we expected — even frozen CLIP features are highly informative for this task
- Cosine similarity as an explicit feature in the MLP makes a measurable difference vs. not including it
- Mixed precision training (fp16) roughly doubled training speed with no accuracy loss

**What was harder than expected:**
- Getting the data pipeline right (downloading 560K images, handling missing/corrupted files, resumable downloading) took significantly more engineering work than the model itself
- Reddit images expire — some fraction of URLs are dead, so the effective dataset is smaller than the TSV count suggests
- Batch size and VRAM management: CLIP + a fusion head uses more memory than expected

**What we'd do differently:**
- Start with a 10K subset to iterate faster before scaling up
- Pre-compute and cache CLIP embeddings for all posts rather than re-running the encoder every epoch (major speedup — each post's embedding never changes if CLIP is frozen)
- Try a retrieval-augmented approach: query Google for each headline and include a credibility score as an additional feature

**Broader takeaways:**
- Pretrained models change the equation — you don't need to design an architecture from scratch when something like CLIP exists that was trained specifically on your relevant signal
- Multimodal models are harder to debug than unimodal ones. When the model is wrong, you don't know if it's the text path, the image path, or the fusion that failed. Unimodal ablations (text-only, image-only) are essential for this reason
- Dataset quality matters more than model complexity — many of the wrong predictions come from ambiguous or mislabeled data, not model limitations

### Pause point
> "We have about X minutes left — any questions before we open it up?"

---

## Demo Slide (Optional but highly recommended)

If you have a trained model, add a live demo at the end:

1. Open a terminal / notebook
2. Find a recent news story with an image (something from the last week works best — shows the model generalizes)
3. Feed in the headline and image
4. Show the probability output and the cosine similarity score

Even if the model is wrong, it's compelling. You can say: "The model predicts real with 62% confidence — and actually, this post was later confirmed as misleading, so this is a near-miss the model almost caught."

Alternatively, prepare 5-6 examples ahead of time in a notebook and run them live.

---

## Slide Order Summary

| Slide | Content | Est. time |
|---|---|---|
| 1 | Title + team | 30 sec |
| 2 | Problem — real fake news example, why images matter | 2 min |
| 3 | Input/Output — pipeline diagram, task definition | 1 min |
| 4 | Background — CLIP paper, what contrastive training is | 2 min |
| *(pause for questions)* | | |
| 5 | Architecture — encoders, fusion strategies diagram | 3 min |
| 6 | Dataset — Fakeddit, examples, stats | 1.5 min |
| 7 | Results — ablation table, key numbers | 2 min |
| 8 | Case studies — 3-4 specific posts with analysis | 3 min |
| 9 | What we learned | 2 min |
| 10 | Demo (live or pre-recorded) | 2 min |
| *(pause for questions)* | | |
