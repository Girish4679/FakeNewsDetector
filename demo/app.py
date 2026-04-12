"""
Gradio demo — paste in a Reddit post title + image URL + comments and
see the full pipeline in action: CLIP model score → Comment Agent adjustment → final verdict.

This is the DEMO for the mini conference presentation.

Setup:
    pip install gradio requests

Run:
    python demo/app.py

Then open http://localhost:7860 in a browser.

NOTE: For the demo, the CLIP model score is SIMULATED (random in a realistic range)
unless your team plugs in the real model checkpoint. Instructions below for both modes.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Run: pip install gradio")

try:
    import requests
    from PIL import Image
    from io import BytesIO
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.comment_agent import CommentAgent

# ── Optional: plug in your real model here ───────────────────────────────────
# Uncomment and adjust once you have a trained checkpoint:
#
# import torch
# import open_clip
# from src.model import FakeNewsDetector
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL = FakeNewsDetector(fusion="concat").to(DEVICE)
# MODEL.load_state_dict(torch.load("checkpoints/checkpoint_best.pt", map_location=DEVICE))
# MODEL.eval()
# CLIP_MODEL, _, CLIP_PREPROCESS = open_clip.create_model_and_transforms(
#     "ViT-B-32", pretrained="openai"
# )
# CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
#
# def real_model_score(title: str, image_pil) -> float:
#     """Returns fake probability from the trained model."""
#     tokens = CLIP_TOKENIZER([title])
#     img_tensor = CLIP_PREPROCESS(image_pil).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         logit = MODEL(tokens.to(DEVICE), None, img_tensor)
#     return torch.sigmoid(logit).item()

# ── Simulated model score (used when real model isn't plugged in) ─────────────
def simulated_model_score(title: str) -> float:
    """
    Simulates a CLIP model score for demo purposes.
    In a real run, replace this with real_model_score().
    Seeds on title text so the same title always gives the same score.
    """
    seed = sum(ord(c) for c in title)
    rng = random.Random(seed)
    return round(rng.uniform(0.35, 0.85), 3)


def load_image_from_url(url: str):
    """Try to load a PIL image from a URL for display."""
    if not HAS_PIL or not url.strip():
        return None
    try:
        resp = requests.get(url.strip(), timeout=5)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


# ── Core pipeline function called by Gradio ──────────────────────────────────
def run_pipeline(title: str, image_url: str, comments_raw: str) -> tuple:
    """
    Called when user clicks Submit.
    Returns: (verdict_html, score_breakdown_md, agent_rationale, image_or_none)
    """
    if not title.strip():
        return (
            "<p style='color:gray'>Enter a post title to get started.</p>",
            "", "", None
        )

    # Parse comments — one per line
    comments = [
        line.strip()
        for line in comments_raw.strip().splitlines()
        if line.strip()
    ]

    # Step 1: Get model score (simulated or real)
    base_score = simulated_model_score(title)

    # Step 2: Run comment agent
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        agent_result = {
            "crowd_signal": "neutral",
            "confidence_adjustment": 0.0,
            "final_score": base_score,
            "rationale": "⚠️ ANTHROPIC_API_KEY not set — agent disabled, showing model score only.",
            "had_comments": bool(comments),
        }
    else:
        agent = CommentAgent()
        try:
            agent_result = agent.analyze(
                title=title,
                base_score=base_score,
                comments=comments,
            )
        except Exception as e:
            agent_result = {
                "crowd_signal": "neutral",
                "confidence_adjustment": 0.0,
                "final_score": base_score,
                "rationale": f"Agent error: {e}",
                "had_comments": bool(comments),
            }

    final = agent_result["final_score"]
    adj   = agent_result["confidence_adjustment"]
    crowd = agent_result["crowd_signal"]

    # ── Verdict HTML ──────────────────────────────────────────────────────────
    if final >= 0.70:
        verdict_label = "⚠️ LIKELY FAKE"
        verdict_color = "#c0392b"
        bar_color = "#e74c3c"
    elif final >= 0.50:
        verdict_label = "🔍 POSSIBLY MISLEADING"
        verdict_color = "#d35400"
        bar_color = "#e67e22"
    else:
        verdict_label = "✅ LIKELY REAL"
        verdict_color = "#27ae60"
        bar_color = "#2ecc71"

    bar_width = int(final * 100)
    crowd_emoji = {"fake": "🚨", "real": "👥✅", "neutral": "💬"}.get(crowd, "💬")

    verdict_html = f"""
    <div style="font-family: 'Courier New', monospace; padding: 20px; border: 2px solid {verdict_color};
                border-radius: 8px; background: #1a1a1a; color: white;">
      <div style="font-size: 1.8em; font-weight: bold; color: {verdict_color}; margin-bottom: 12px;">
        {verdict_label}
      </div>
      <div style="margin-bottom: 8px; font-size: 0.9em; color: #aaa;">Fake probability</div>
      <div style="background: #333; border-radius: 4px; height: 24px; margin-bottom: 16px;">
        <div style="background: {bar_color}; width: {bar_width}%; height: 100%;
                    border-radius: 4px; display: flex; align-items: center;
                    padding-left: 8px; font-weight: bold; font-size: 0.85em;">
          {final:.1%}
        </div>
      </div>
      <div style="color: #ccc; font-size: 0.95em;">
        {crowd_emoji} Crowd signal: <strong style="color: white;">{crowd.upper()}</strong>
      </div>
    </div>
    """

    # ── Score breakdown ───────────────────────────────────────────────────────
    adj_str = f"{adj:+.3f}" if adj != 0 else "0.000 (neutral)"
    breakdown_md = f"""
| Step | Score |
|---|---|
| 🤖 CLIP model (base) | `{base_score:.3f}` |
| 💬 Comment agent adjustment | `{adj_str}` |
| **🎯 Final score** | **`{final:.3f}`** |

**Comments analyzed:** {len(comments)} comment{"s" if len(comments) != 1 else ""}
    """

    rationale = agent_result["rationale"]
    image = load_image_from_url(image_url)

    return verdict_html, breakdown_md, rationale, image


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Fake News Detector",
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
        ),
        css="""
        .container { max-width: 900px; margin: auto; }
        h1 { font-family: 'Courier New', monospace !important; }
        """
    ) as demo:

        gr.Markdown("""
# 🔍 Fake News Detector
**Multimodal detection using CLIP + Comment Agent**

*Rutgers CS 462 — Deep Learning Final Project*
        """)

        with gr.Row():
            with gr.Column(scale=2):
                title_input = gr.Textbox(
                    label="📰 Post Title (clean_title from Fakeddit)",
                    placeholder="e.g. Senator caught accepting bribes from lobbyists",
                    lines=2,
                )
                image_url_input = gr.Textbox(
                    label="🖼️ Image URL (optional — for display only)",
                    placeholder="https://i.redd.it/example.jpg",
                )
                comments_input = gr.Textbox(
                    label="💬 Comments (one per line — paste from Reddit or comments.tsv)",
                    placeholder=(
                        "This photo is from 2018, not current events\n"
                        "Already debunked: snopes.com/...\n"
                        "The source on this is completely wrong"
                    ),
                    lines=6,
                )
                submit_btn = gr.Button("🔍 Analyze Post", variant="primary")

            with gr.Column(scale=1):
                image_output = gr.Image(label="Post Image", height=200)

        verdict_output = gr.HTML(label="Verdict")

        with gr.Row():
            with gr.Column():
                breakdown_output = gr.Markdown(label="Score Breakdown")
            with gr.Column():
                rationale_output = gr.Textbox(
                    label="💡 Comment Agent Rationale",
                    lines=3,
                    interactive=False,
                )

        # ── Pre-loaded examples for the presentation ──────────────────────────
        gr.Examples(
            examples=[
                [
                    "Flood destroys thousands of homes in Texas",
                    "",
                    "This photo is from the 2017 Hurricane Harvey, not recent\nAlready debunked on Reuters fact check\nThe original photographer confirmed this is misattributed",
                ],
                [
                    "Local firefighters rescue family of five from apartment blaze",
                    "",
                    "My neighbor posted this, it definitely happened last night\nWas on the local Channel 7 news\nFirefighter here — can confirm this is our crew",
                ],
                [
                    "Politicians secretly meeting with foreign agents exposed",
                    "",
                    "No source cited anywhere\nThis account always posts fake stuff\nCan someone find an actual news link? This feels off",
                ],
            ],
            inputs=[title_input, image_url_input, comments_input],
            label="📋 Example Posts (click to load)",
        )

        submit_btn.click(
            fn=run_pipeline,
            inputs=[title_input, image_url_input, comments_input],
            outputs=[verdict_output, breakdown_output, rationale_output, image_output],
        )

    return demo


if __name__ == "__main__":
    print("Starting Fake News Detector Demo...")
    print("Open: http://localhost:7860\n")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set — comment agent will be disabled.")
        print("    Set it with: export ANTHROPIC_API_KEY='sk-ant-...'\n")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)