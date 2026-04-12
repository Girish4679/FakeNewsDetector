# Slide Deck Draft
### Beyond Text: A Multimodal Fusion Approach to Misinformation Detection

> Format notes: Keep text on slides SHORT. The words here are what go on the slide.
> Everything in (parentheses) is a speaker note — say it out loud, don't put it on the slide.
> [PLACE X HERE] = grab this image/screenshot and drop it in.

---

---

## SLIDE 1 — Title Slide

**Title (large, centered):**
> Beyond Text: A Multimodal Fusion Approach to Misinformation Detection on Social Media

**Subtitle:**
> CS [Course Number] · Spring 2025

**Team:**
> Girish Ranganathan · Rithvik Sriram · James Dequina · Armaan Shivansh · Arvid Eapen

[PLACE BACKGROUND IMAGE HERE — a split image: one half a real news photo, other half the same photo with a fake caption overlaid. Search "out of context image misinformation" on Google Images for inspiration. Keep it dark/muted so the title text reads clearly on top.]

---

---

## SLIDE 2 — The Problem

**Title:** The Problem with Detecting Fake News Today

**Left column — bullet points:**
- Most detection systems only read the text
- Images are ignored entirely
- But misinformation often lives in the *gap* between image and caption

**Right column — the key example:**

[PLACE EXAMPLE 1 HERE — find a well-known out-of-context image example. Good options:
  - The "millions protesting in Brazil" photo that was actually a 2013 Lula concert
  - Any viral COVID-era misattributed photo
  - Search "out of context image fake news example" — Reuters Fact Check and Snopes have dozens
  Show the image on the right with the false caption below it, and a red ✗ badge]

**Bottom of slide (small text):**
> A text-only model reads this caption and sees nothing wrong.

(Speaker note: "The image is real. The caption is real. But together, they're a lie. This is the problem text-only models completely miss — and it's the problem we're solving.")

---

---

## SLIDE 3 — Input / Output

**Title:** What Our Model Does

**Center of slide — clean diagram:**

```
INPUT                                    OUTPUT

"Senator arrested for fraud"  ──┐
                                 ├──▶  [ Our Model ]  ──▶  91% FAKE
[image of a random politician]  ─┘
```

[PLACE DIAGRAM HERE — recreate the above as a clean graphic. Use PowerPoint/Canva shapes. Text box on left for the post, arrow into a box labeled "FakeNewsDetector," arrow out to a red badge saying "91% FAKE." Keep it simple and large.]

**Below the diagram, three bullet points:**
- Input: any social media post with a title + image
- Output: probability score (0 = real, 1 = fake)
- Key insight: the *relationship* between image and text is the signal

(Speaker note: "Given any post, we output a single number — the probability it's fake. The key design decision in our model is how we combine the text and the image to compute that number.")

---

---

## SLIDE 4 — Background: Why CLIP?

**Title:** Our Backbone: CLIP

**Left side — 3 bullet points:**
- Pretrained by OpenAI on **400 million** image-caption pairs
- Trained to ask: *does this image match this caption?*
- Released publicly — we download it, we don't train it ourselves

**Right side — THE CLIP diagram:**

[PLACE CLIP FIGURE 1 HERE — Google "CLIP paper figure 1 OpenAI" or go to:
  https://openai.com/research/clip
  It's the grid showing images on one axis, captions on the other, with the diagonal highlighted.
  This image is from a public OpenAI blog post — free to use in a class presentation.
  Caption it: "Radford et al., ICML 2021"]

**Bottom callout box (colored, stands out):**
> CLIP's similarity score between text and image = a fake news signal out of the box

(Speaker note: "OpenAI trained CLIP to solve almost exactly our problem — is this image and this caption a match? After 400 million examples of that, the model learned really powerful features for detecting mismatches. We just use what they built.")

---

---

## SLIDE 5 — Background: CLIP in Action

**Title:** What CLIP "Sees"

**Two examples side by side:**

**Example A — Real post (left):**

[PLACE REAL REDDIT POST HERE — find a Reddit post from r/worldnews or r/news where the image genuinely matches the headline. Screenshot it or mock it up. Below it write:]
> Cosine similarity: **0.81** ✅ Image and text align

**Example B — Fake post (right):**

[PLACE FAKE REDDIT POST HERE — find an out-of-context example from Fakeddit or a fact-check site. Below it write:]
> Cosine similarity: **0.09** ❌ Image and text do NOT align

**Bottom of slide:**
> Our model learns: low similarity → likely fake

(Speaker note: "This cosine similarity number is the dot product of the two CLIP embeddings. It's between -1 and 1. Real posts cluster high. Fake posts cluster low. We feed this directly into our classifier as an explicit feature.")

---

---

## SLIDE 6 — Architecture

**Title:** Our Architecture

[PLACE ARCHITECTURE DIAGRAM HERE — this is the most important visual in the deck. Build it in PowerPoint/Canva:

  Left side:
    Box: "Post Title Text"  →  Arrow  →  Box: "CLIP Text Encoder" → "512-dim embedding"

  Right side:
    Box: "Post Image" → Arrow → Box: "CLIP Image Encoder" → "512-dim embedding"

  Middle:
    Both arrows pointing down to: "cosine similarity (1 number)"

  All three (text emb + image emb + cos_sim) → Box: "Fusion Layer" → Box: "MLP" → Box: "Real / Fake"

  Color code: blue for text path, orange for image path, green for fusion.
  Keep it clean — no more than 8 boxes total.]

**Right side bullets (small):**
- CLIP weights: **frozen** (pretrained alignment knowledge preserved)
- Only the fusion MLP is trained from scratch
- ~150M total params, only ~200K trained (0.1%)

(Speaker note: "We freeze CLIP entirely. We're not retraining it — we're just plugging a small classifier on top of features it already knows how to compute. This means we need very little data and very little compute compared to training from scratch.")

---

---

## SLIDE 7 — Fusion Strategies

**Title:** Our Research Contribution: Fusion Strategies

**Table — clean, centered:**

| Strategy | How it combines text + image | Our hypothesis |
|---|---|---|
| **Concatenation** | [text \|\| image \|\| cos_sim] → MLP | Strong baseline |
| **Cross-modal Attention** | Text attends over image patches | Best for subtle mismatches |
| **Gated Fusion** | Learned weight per modality | Best when one modality is weak |
| **Cosine Only** | cos_sim alone → MLP | How far does CLIP alone get? |

**Bottom callout:**
> Research question: Does cross-modal attention outperform simple concatenation?

[PLACE CROSS-ATTENTION DIAGRAM HERE — draw a simple diagram showing:
  "protest in London" text → attention arrows pointing to specific patches in the image
  (patches showing background buildings, not the people)
  This shows attention catching a mismatch at the patch level.
  Can be a rough sketch — it just needs to convey the concept.]

(Speaker note: "Our core research question is whether sophisticated fusion actually helps. Simple concatenation has been shown to be surprisingly hard to beat. We're testing whether giving the text and image direct access to each other's internal representations — through attention — changes that.")

*[PAUSE FOR QUESTIONS HERE]*

---

---

## SLIDE 8 — Dataset

**Title:** Dataset: r/Fakeddit

**Left — stats:**

| | Count |
|---|---|
| Training posts | 563,000 |
| Validation posts | 58,000 |
| Test posts | 58,000 |
| Subreddits | 22 |
| Label types | 2-way, 3-way, 6-way |

Source: Nakamura et al., LREC 2020

**Right — example posts grid:**

[PLACE 4 EXAMPLE POSTS HERE — 2x2 grid:
  Top left:  real post (clear match between image and headline)   → green "REAL" badge
  Top right: fake post (obvious mismatch)                         → red "FAKE" badge
  Bottom left:  real post (different example)                     → green "REAL" badge
  Bottom right: fake post (subtle, harder example)                → red "FAKE" badge

  To find examples: open data/multimodal_train.tsv, sort by 2_way_label,
  pick posts where hasImage=True and image_url still works.
  Screenshot the actual Reddit posts by pasting the URLs into a browser.]

**Bottom:**
> We focus on the 2-way label (real vs. fake) as our primary task

(Speaker note: "Fakeddit was collected by scraping Reddit posts that were flagged and removed by moderators for being misleading. The labels come from community moderation, not manual annotation — which means real-world noise, but also real-world scale.")

---

---

## SLIDE 9 — Results: Numbers

**Title:** Results

**Main ablation table — large, centered:**

| Model | F1 ↑ | AUC-ROC ↑ | Accuracy ↑ |
|---|---|---|---|
| Text only | [XX.X] | [X.XXX] | [XX.X%] |
| Image only | [XX.X] | [X.XXX] | [XX.X%] |
| Concat (ours) | **[XX.X]** | **[X.XXX]** | **[XX.X%]** |
| Gated fusion | [XX.X] | [X.XXX] | [XX.X%] |
| Cross-attention | **[XX.X]** | **[X.XXX]** | **[XX.X%]** |

> F1 is our primary metric — the dataset is imbalanced, so accuracy alone is misleading.

[PLACE ROC CURVE HERE — after training, generate this with:
  from sklearn.metrics import RocCurveDisplay
  Plot one curve per fusion strategy, all on the same axes.
  Color-code to match the table. This is a standard figure in ML papers and looks professional.]

**Callout box (highlight your best result):**
> Cross-modal attention: F1 = [X.XX] — [X.X]% improvement over text-only baseline

(Speaker note: "The most important comparison is multimodal vs. text-only. If multimodal doesn't beat text-only, our whole premise fails. The second most important comparison is which fusion strategy wins — that's our actual research contribution.")

---

---

## SLIDE 10 — Results: Case Studies

**Title:** Case Studies

**Layout: 3 columns, each a post example**

---

**Column 1 — "The Clear Win"**
[PLACE POST IMAGE HERE]
> Headline: *"[actual headline from a fake post in your test set]"*
> Cosine similarity: **0.07**
> Model: **94% FAKE** ✅ Correct

Small note below: *Image is from [year/event] — unrelated to the headline*

---

**Column 2 — "Where We Beat Text-Only"**
[PLACE POST IMAGE HERE]
> Headline: *"[a fake post where the text sounds completely real]"*
> Text-only model: **73% REAL** ✗ Wrong
> Our model: **81% FAKE** ✅ Correct

Small note below: *The text is plausible — only the image mismatch reveals the fake*

---

**Column 3 — "A Failure Case"**
[PLACE POST IMAGE HERE]
> Headline: *"[a real post our model incorrectly labels]"*
> Model: **78% FAKE** ✗ Wrong
> Actual label: **REAL**

Small note below: *Satirical subreddit post — model misled by unusual image style*

---

(Speaker note: "We're showing you a failure case because we think it's important. The model struggles with satire and irony — posts from subreddits like r/TheOnion where the text is intentionally absurd but the post is labeled real. This is a known hard case for automated fact-checking.")

---

---

## SLIDE 11 — Results: Attention Visualization
### (Include this slide ONLY if you implement cross-modal attention)

**Title:** What the Model Attends To

[PLACE ATTENTION HEATMAP HERE — after training, take a specific fake post and visualize
  which image patches the text [CLS] token attended to most.
  To generate: extract attention weights from your cross-attention layer,
  reshape to 14×14 (the ViT patch grid), overlay as a heatmap on the original image.
  Libraries: matplotlib imshow with alpha, or seaborn heatmap.

  Best case: find an example where the attention highlights a suspicious part of the image —
  e.g., a date stamp, a flag, a logo — that contradicts the headline.
  Caption it clearly: "Red = high attention. The model focuses on [X] which contradicts the claim that [Y]."]

**One sentence below:**
> The model learns to focus on parts of the image most relevant (or irrelevant) to the headline

---

---

## SLIDE 12 — What We Learned

**Title:** What We Learned

**Three columns:**

**What worked ✅**
- CLIP's pretrained similarity is a strong signal even before any training
- Cosine similarity as an explicit feature matters
- Frozen CLIP + small MLP is a surprisingly competitive baseline
- Mixed precision (fp16) — 2x speedup, same accuracy

**Harder than expected ⚠️**
- Data pipeline: downloading 560K images, handling dead URLs, resumable scripts
- VRAM management: CLIP + fusion head uses more memory than expected
- Reddit image CDN links expire — effective dataset is smaller than TSV count

**Do differently next time 🔄**
- Pre-cache CLIP embeddings (huge speedup — recomputing them every epoch is wasteful)
- Start with 10K subset to iterate, then scale up
- Try retrieval-augmented features: query Google for each headline, feed credibility score to model

(Speaker note: "The biggest surprise was how much engineering work went into the data pipeline before we even touched the model. Getting 560K images downloaded, resumably, across multiple machines, with a shared Google Drive — that alone took more time than writing the model code.")

---

---

## SLIDE 13 — Demo
### (Live or pre-recorded)

**Title:** Live Demo

[PLACE NOTEBOOK SCREENSHOT HERE if doing pre-recorded — show:
  1. A recent real news headline + image fed into the model
  2. The cosine similarity score printed
  3. The final fake/real probability
  Annotate the screenshot with arrows pointing to the key numbers]

**OR run live:**
- Open a Jupyter notebook
- Have 5 examples ready (2 real, 2 fake, 1 ambiguous)
- Show the cosine similarity for each before showing the final prediction
- Ask the audience to guess real or fake before revealing

**Bottom of slide:**
> Try it yourself: `python src/evaluate.py --checkpoint checkpoints/checkpoint_best.pt ...`

---

---

## SLIDE 14 — Thank You / Questions

**Title:** Thank You

**Left side:**
> Questions?

**Right side — summary box:**
- CLIP backbone: 400M pretrained image-text pairs
- Dataset: r/Fakeddit, 563K Reddit posts
- Best model: [fusion strategy], F1 = [XX.X]
- Key finding: [your main result in one sentence]

**Bottom — citations (small):**
> Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
> Nakamura et al. "r/Fakeddit: A New Multimodal Benchmark Dataset." LREC 2020.

[PLACE TEAM PHOTO HERE — optional but humanizes the presentation]

---

---

## Quick Asset Checklist

Before you finalize the deck, make sure you have all of these:

- [ ] Out-of-context fake news example image (Slide 2) — check Reuters Fact Check or Snopes
- [ ] CLIP Figure 1 diagram (Slide 4) — openai.com/research/clip
- [ ] Architecture diagram built in Canva/PowerPoint (Slide 6)
- [ ] 4 real Fakeddit post examples, 2x2 grid (Slide 8) — paste image_urls from TSV into browser
- [ ] ROC curve plot from your trained model (Slide 9)
- [ ] 3 specific test set posts for case studies (Slide 10) — run evaluate.py and find interesting examples
- [ ] Attention heatmap if cross-attention is implemented (Slide 11)
- [ ] Demo notebook ready and tested (Slide 13)
