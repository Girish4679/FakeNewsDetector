# How Our Fake News Detector Works
### A plain-English guide for the whole team

---

## The Big Picture

We're building a model that looks at a Reddit post — the title text and the image — and decides whether it's real or fake/misleading. That's it. One input (text + image), one output (real or fake).

The challenge is that neither piece of information alone is enough. A headline like *"Senator caught in scandal"* could be real or fake based purely on the text. But if the image attached to it is a 10-year-old unrelated photo, that mismatch is a strong signal that something is off. **Our model needs to learn that the relationship between the image and text matters just as much as each one individually.**

Here's the full pipeline at a glance:

```
Reddit post title (text)
        │
        ▼
  [ Text Encoder ]  ──────────────────────┐
                                          │
                                    [ Fusion Layer ]──▶ [ MLP ] ──▶ Real or Fake?
                                          │
  [ Image Encoder ] ──────────────────────┘
        ▲
        │
  Image from post
```

Every component in this diagram has a specific job. Let's go through them one by one.

---

## Part 1: What Is a "Model" and How Does It Learn?

Before anything else, it helps to understand what a neural network actually is.

A neural network is essentially a very large mathematical function with millions of *parameters* (numbers). You feed it an input, it does a bunch of math, and it spits out a prediction. At the start, the parameters are random, so the predictions are garbage. **Training** is the process of slowly adjusting those parameters so the predictions get better.

Here's how training works in a loop:

1. Feed the model a batch of posts (say, 32 at a time)
2. The model predicts "real" or "fake" for each one
3. We compare predictions to the actual labels and compute a **loss** — a single number measuring how wrong the model was (higher = worse)
4. **Backpropagation** figures out which parameters were responsible for the error and by how much
5. The **optimizer** nudges each parameter slightly in the direction that reduces the loss
6. Repeat millions of times across the whole dataset

After enough iterations, the parameters have been adjusted so the model makes good predictions on data it's never seen before.

---

## Part 2: Embeddings — Turning Words and Pixels Into Numbers

Neural networks can only work with numbers. So the first challenge is: how do you convert a sentence or an image into a form the model can process?

The answer is an **embedding** — a list of numbers (a vector) that represents the meaning of the input. The key insight is that similar things should have similar embeddings.

For example, after training, a good text encoder might produce embeddings where:
- *"President signs new bill"* and *"Politician approves legislation"* have very similar vectors
- *"Cat sits on mat"* has a very different vector from both

For images, similar visual concepts cluster together in embedding space — photos of sunsets are near each other, faces near faces, etc.

In our model, both the text and the image get converted into embeddings, and then we combine them to make the final prediction.

---

## Part 3: Why CLIP Instead of Separate Encoders?

The obvious approach would be to use one model for text (like BERT or RoBERTa) and a completely separate model for images (like ResNet or ViT), then combine their outputs. That works, but there's a problem: **the two models were trained completely independently, so their embeddings live in totally different mathematical spaces.** There's no reason they'd be compatible.

CLIP (Contrastive Language-Image Pretraining) solves this in an elegant way.

OpenAI trained CLIP on **400 million image-caption pairs** scraped from the internet. The training objective was simple: given an image and a caption, learn to tell whether they belong together. In practice, CLIP was shown a batch of pairs and had to figure out which caption goes with which image (and vice versa), with all other combinations being wrong.

```
Training signal for CLIP:
   ✅ photo of a dog + "a golden retriever playing fetch"  → high similarity
   ❌ photo of a dog + "the Eiffel Tower at night"         → low similarity
```

After 400 million examples of this, CLIP learned to put matching image-text pairs close together in embedding space and non-matching pairs far apart. **Both the text encoder and image encoder were trained together toward the same goal**, so their embeddings are naturally compatible.

This is exactly the signal we need for fake news detection. If a post's image and headline are semantically mismatched, CLIP's similarity score will be low — and that's a direct red flag for misinformation.

**What we get from CLIP:**
- A text encoder that produces a 512-dimensional embedding for any piece of text
- An image encoder that produces a 512-dimensional embedding for any image
- Both live in the same embedding space, so they can be directly compared

We're using the pretrained version (`openai/clip-vit-base-patch16`) — this means we download the weights OpenAI already trained. We don't train CLIP ourselves (that would require weeks on hundreds of GPUs). We just use what they built.

---

## Part 4: The Text Encoder

The "ViT" in `clip-vit-base-patch16` refers to the vision side. The text side is a **transformer**, which is the same architecture that powers GPT and BERT.

### Tokenization

Before the text encoder sees any text, it has to be tokenized. Tokenization is the process of splitting text into small chunks called tokens, then converting each chunk to a number.

```
"Senator caught in scandal" 
    → ["senator", "caught", "in", "scan", "##dal"]   (split into subwords)
    → [1233, 6245, 1999, 8442, 3683]                  (each mapped to an ID)
```

CLIP's tokenizer caps text at **77 tokens** (a CLIP-specific constraint). Sentences shorter than 77 tokens are padded; longer ones are truncated.

### What the transformer does

The transformer takes those token IDs and processes them through multiple layers of **self-attention**. Self-attention is the mechanism that lets the model relate words to each other — in the sentence *"the bank by the river"*, attention helps the model understand that "bank" is geographical, not financial, by looking at "river."

After all the layers, the transformer outputs one vector per token, but we only keep the first one (a special `[CLS]` token that's been trained to summarize the whole sentence). This gives us our **512-dimensional text embedding**.

---

## Part 5: The Image Encoder

CLIP uses a **Vision Transformer (ViT)** for images. Despite the name, it works very similarly to the text transformer.

### How ViT processes an image

1. The image (224×224 pixels) is cut into a grid of non-overlapping **patches**, each 16×16 pixels. The "patch16" in the model name refers to this.
2. Each patch gets flattened into a vector and then processed as if it were a word in a sentence.

```
224×224 image → 14×14 grid of 16×16 patches → 196 patches
Each patch treated like a "word" in a sequence of length 196
```

3. Just like the text transformer, a self-attention mechanism lets each patch "look at" every other patch to understand context. This is powerful — a patch showing half a face can attend to other patches to understand the full face.
4. A special `[CLS]` patch token (added at the start) summarizes the whole image. Its output vector is the **512-dimensional image embedding**.

### Why ViT instead of ResNet?

We considered ResNet-50 (a popular convolutional network), but ViT is better for this task. ResNet uses local convolution filters that slide across the image — it's good at detecting textures and simple objects. ViT's attention mechanism can capture long-range relationships across the whole image, which matters more for detecting manipulated or out-of-context images where the issue is often about global context, not local texture.

---

## Part 6: Fine-Tuning vs. Freezing

When we load pretrained CLIP weights, we have a choice: do we update those weights during our training, or leave them fixed?

**Freezing** means we treat CLIP as a fixed feature extractor. The CLIP weights never change — we only train the new layers we add on top (the fusion layer and classifier). This is fast, uses less memory, and works surprisingly well because CLIP's features are already so good.

**Fine-tuning** means we also let some of CLIP's weights update during training, so they can adapt specifically to Reddit misinformation patterns. We do this carefully:
- We freeze the early layers (they learn fundamental things like edges, shapes, and basic grammar that are useful everywhere)
- We only unfreeze the **last 3 layers** of both encoders, since those are the layers that produce task-specific high-level features

We use a much lower learning rate for these CLIP layers (1e-5 or 1e-6) compared to our new fusion head (1e-3). This is called **differential learning rates** — if we used the same learning rate, the large gradient updates would scramble the carefully pretrained CLIP weights.

**Strategy:**
- Start with frozen CLIP to get a strong baseline fast
- Then try unfreezing the last 3 layers if we want to squeeze out more performance

---

## Part 7: The Fusion Layer

Now we have two 512-dimensional embeddings — one for the text, one for the image. The fusion layer's job is to combine them into a single representation that the classifier can use.

### Why not just concatenate?

We actually do concatenate — but we also add one extra feature that makes a big difference:

**Cosine similarity.** Since both CLIP embeddings are unit vectors (length = 1, thanks to L2 normalization), their dot product directly measures how semantically aligned they are. A score near 1 means the image and text match well. A score near 0 or negative means they're semantically unrelated.

```
text_emb  = CLIP text encoder output   (512 numbers, unit vector)
image_emb = CLIP image encoder output  (512 numbers, unit vector)
cos_sim   = dot product of the two     (1 number, between -1 and 1)

fusion input = [text_emb, image_emb, cos_sim] = 1025 numbers
```

This cosine similarity score is the single most important feature in our model — it's a direct measure of image-text alignment that CLIP was specifically trained to produce.

### The other fusion strategies (your team's contribution)

The concatenation baseline above is the simplest approach. Your proposal lists three more sophisticated strategies. These are the parts where your team makes a research contribution beyond just using pretrained CLIP:

**Cross-modal attention:**
Instead of just taking a single summary vector from each encoder, we let the text attend over the individual image patch tokens (and vice versa). The idea is that specific words in the headline might align (or misalign) with specific regions in the image. For example, if the headline says "protest in London" but the image's patches corresponding to signage don't match English text, attention can catch that. This is the most technically interesting fusion strategy.

```
"protest in London"
      │
      ▼  attends over
[patch 1] [patch 2] ... [patch 196]   ← image patches
```

**Gated fusion:**
A small network learns a "gate" — a set of weights between 0 and 1 that decides how much to trust the text vs. the image for each individual post. For a post with a blurry or missing image, the gate should learn to weight the text more heavily. For a post where the text is vague but the image is very specific, weight the image more.

```
gate = sigmoid(W * [text_emb, image_emb])   → values between 0 and 1
output = gate * text_emb + (1 - gate) * image_emb
```

**Contrastive alignment:**
Rather than just using CLIP's pretrained similarity, we could add a training objective that pushes the model to explicitly maximize the embedding distance for fake posts and minimize it for real ones. This is basically asking CLIP to fine-tune its alignment understanding specifically for the misinformation domain.

---

## Part 8: The MLP Classifier

MLP stands for Multi-Layer Perceptron — it's just a stack of fully connected layers with non-linear activations in between. After the fusion layer gives us a single vector, the MLP turns that into one number (a logit), which gets converted to a probability via sigmoid.

```
1025-dim fused vector
    → Linear layer (1025 → 256)
    → GELU activation           ← non-linearity; lets the network learn complex patterns
    → Dropout (30%)             ← randomly zeros out 30% of neurons during training to prevent overfitting
    → Linear layer (256 → 64)
    → GELU
    → Dropout (30%)
    → Linear layer (64 → 1)     ← single number (logit)
    → Sigmoid                   ← converts to probability between 0 and 1
```

**GELU** is just a smooth non-linear function (similar to ReLU). Without non-linearities, stacking linear layers would just collapse into one linear layer — the network couldn't learn anything complex.

**Dropout** randomly switches off 30% of neurons during each training step. This forces the network to not rely too heavily on any single neuron, which helps it generalize to data it hasn't seen (prevents overfitting).

The final output is a probability. If it's > 0.5, we predict "fake"; otherwise "real."

---

## Part 9: The Loss Function

Our loss function is **Binary Cross-Entropy with Logits** (BCEWithLogitsLoss). This measures how far off our probability prediction is from the true label.

```
True label = 1 (fake),  predicted probability = 0.9  → low loss (correct and confident)
True label = 1 (fake),  predicted probability = 0.3  → high loss (wrong and confident)
True label = 0 (real),  predicted probability = 0.4  → low loss (correct)
```

We also use a `pos_weight` parameter. If the dataset has more real posts than fake posts (class imbalance), without correction the model might learn to just always predict "real" — it would be right most of the time but useless. `pos_weight` makes fake examples count more in the loss so the model is penalized more for missing them.

---

## Part 10: The Optimizer and Learning Rate Schedule

**AdamW** is our optimizer — the algorithm that adjusts parameters after each batch. It keeps a running average of recent gradients to smooth out noisy updates, and adds a small penalty for large weights (weight decay) to keep the model from overfitting.

**Learning rate (LR)** controls how big each update step is. Too large and the model oscillates and never converges. Too small and training takes forever.

We use a **cosine schedule with warmup**:
- For the first ~100 steps, LR ramps up slowly from 0 (warmup). This prevents unstable large updates right at the start when gradients are noisiest.
- After warmup, LR follows a cosine curve, gradually decreasing to 0. This helps the model settle into a good solution at the end of training rather than bouncing around.

```
LR
▲
│    /‾‾──────────────────────────────────────╮
│   /                                          ╰─────╮
│  /                                                  ╰───────
│ /  (warmup)              (cosine decay)
└──────────────────────────────────────────────────▶ steps
```

---

## Part 11: Mixed Precision Training

Modern NVIDIA GPUs have dedicated hardware for 16-bit floating point math (fp16), which is faster and uses half the memory of standard 32-bit (fp32). Mixed precision training uses fp16 for the forward pass (where most of the memory and compute goes) and fp32 for critical things like the gradient accumulation and weight updates.

In practice this means: **same results, half the VRAM, roughly 2x faster.** We use `torch.cuda.amp.autocast()` which handles all of this automatically.

---

## Part 12: Evaluation Metrics

We use three metrics to evaluate the model:

**F1-score (primary metric):**
The harmonic mean of precision and recall.
- *Precision* = of all posts we predicted as fake, what fraction actually were?
- *Recall* = of all fake posts, what fraction did we catch?
- F1 balances both. A model that just predicts "fake" for everything would have high recall but terrible precision, giving a bad F1.
- We use this as the primary metric because the dataset may be imbalanced.

**AUC-ROC:**
Measures how well the model separates real from fake across all possible thresholds (not just 0.5). A score of 1.0 is perfect; 0.5 is random chance. This is useful for comparing models independently of the threshold you pick.

**Accuracy:**
Plain percentage of correct predictions. We report it but don't rely on it because it can be misleading with imbalanced classes.

---

## Part 13: Checkpointing

Training takes a long time. If the computer crashes or the session ends, we don't want to start over. Every epoch, we save the model's parameters (and optimizer state) to disk — this is called a **checkpoint**.

We save two files:
- `checkpoint_latest.pt` — saved every epoch, so you can always resume from the last completed epoch
- `checkpoint_best.pt` — only overwritten when validation F1 improves, so you always have the best-performing version

Checkpoints go into Google Drive so they're backed up and accessible to the whole team.

---

## Summary: The Full Pipeline

```
INPUT: Reddit post
  title: "Breaking: Senator caught stealing funds"
  image: [some .jpg file]

STEP 1 — Tokenize text
  "Breaking: Senator caught stealing funds"
  → [49406, 10629, 25, 4939, 6200, 12543, 8449, 49407]  (CLIP token IDs, padded to 77)

STEP 2 — CLIP text encoder
  Token IDs → 12 transformer layers → [CLS] token → 512-dim text embedding
  e_text = [0.23, -0.41, 0.87, ...]   (512 numbers)

STEP 3 — Load and preprocess image
  image.jpg → resize to 224×224 → normalize pixel values

STEP 4 — CLIP image encoder
  224×224 image → cut into 14×14 grid of patches → 12 transformer layers
  → [CLS] patch → 512-dim image embedding
  e_image = [0.11, 0.73, -0.22, ...]  (512 numbers)

STEP 5 — Compute cosine similarity
  cos_sim = dot(e_text, e_image) = 0.18   (low: image and text don't match well → suspicious)

STEP 6 — Fusion
  fused = [e_text, e_image, cos_sim]   (1025 numbers)

STEP 7 — MLP classifier
  1025 → 256 → 64 → 1
  logit = 2.3

STEP 8 — Sigmoid
  probability = sigmoid(2.3) = 0.91

OUTPUT: 91% probability of being fake → predicted label: FAKE
```

---

## What Each Team Member Owns

| Role | Files | What you're building |
|---|---|---|
| **Vision Pipeline** | `src/model.py` → `VisionEncoder` (inside `CLIPEncoder`) | The image processing path through CLIP's ViT encoder |
| **Text Pipeline** | `src/model.py` → `TextEncoder` (inside `CLIPEncoder`) | The text tokenization and transformer encoding |
| **Fusion Layer** | `src/model.py` → `ConcatFusion`, `CrossAttentionFusion`, `GatedFusion` | All fusion strategies; the concatenation baseline is done, the other three are yours |
| **Evaluation** | `src/evaluate.py`, metrics in `src/train.py` | Metrics, ablation tables, confusion matrices, attention visualizations |
