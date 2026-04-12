"""
Comment Agent — analyzes Reddit comments to adjust the CLIP model's fake news prediction.

This runs at INFERENCE TIME ONLY (no training needed). It calls the Anthropic API
with the top comments from a post and produces:
  - crowd_signal:            "real" | "fake" | "neutral"
  - confidence_adjustment:   float in [-0.3, 0.3] added to the model's base score
  - rationale:               one plain-English sentence explaining why

Usage:
    from src.comment_agent import CommentAgent
    agent = CommentAgent()
    result = agent.analyze(
        title="Senator caught stealing funds",
        base_score=0.73,       # model's raw fake probability
        comments=["This photo is from 2018 floods", "Fake! debunked here: ..."]
    )
    print(result)
    # {
    #   "crowd_signal": "fake",
    #   "confidence_adjustment": 0.18,
    #   "final_score": 0.91,
    #   "rationale": "Multiple commenters identified the image as recycled from a 2018 event."
    # }
"""

import json
import os
import re
from typing import Optional
import anthropic

# ── Anthropic client (picks up ANTHROPIC_API_KEY from environment) ──────────
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ── Prompt template ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a misinformation analyst. You will be given:
1. The title of a Reddit post
2. A fake-news detector's confidence that the post is fake (0.0 = definitely real, 1.0 = definitely fake)
3. The top comments on that post

Your job is to analyze whether the CROWD in the comments is treating this post as real or fake.

Look specifically for these signals in the comments:
- Debunking language: "this is fake", "already debunked", "misleading", "out of context"
- Source identification: "this photo/video is from [other event/year]", "this is old news"
- Fact-check links: links to Snopes, PolitiFact, Reuters, AP fact-checks
- Strong skepticism: "I don't believe this", "where's the source?", "citation needed"
- Strong belief: "this is real", "I saw this on the news", comments treating it as true
- Neutral/irrelevant: jokes, off-topic, no signal either way

Based on the above, output ONLY a valid JSON object with these exact keys:
{
  "crowd_signal": "fake" | "real" | "neutral",
  "confidence_adjustment": <float between -0.25 and 0.25>,
  "rationale": "<one concise sentence>"
}

Rules for confidence_adjustment:
- Positive values (up to +0.25) = comments suggest it's MORE likely fake
- Negative values (down to -0.25) = comments suggest it's MORE likely real
- 0.0 = comments are neutral or mixed
- Use the full range only when comments are very clear. Be conservative for ambiguous cases.

Output ONLY the JSON. No preamble, no explanation, no markdown.
"""


def _build_user_message(title: str, base_score: float, comments: list[str]) -> str:
    comments_block = "\n".join(
        f"{i+1}. {c.strip()}" for i, c in enumerate(comments[:15]) if c.strip()
    )
    if not comments_block:
        comments_block = "(no comments available)"

    return (
        f"Post title: {title}\n\n"
        f"Detector confidence (fake probability): {base_score:.2f}\n\n"
        f"Top comments:\n{comments_block}"
    )


def _parse_response(raw: str) -> dict:
    """Extract JSON from the model response, handling edge cases."""
    # Strip any accidental markdown fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: neutral agent output if parsing fails
        return {
            "crowd_signal": "neutral",
            "confidence_adjustment": 0.0,
            "rationale": "Could not parse agent response; defaulting to neutral.",
        }


# ── Public interface ─────────────────────────────────────────────────────────
class CommentAgent:
    """
    Wraps the LLM call and post-processing into a simple .analyze() method.

    Args:
        model: Anthropic model name to use (default: claude-haiku for speed/cost)
        max_tokens: Max tokens for the response
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 256,
    ):
        self.model = model
        self.max_tokens = max_tokens

    def analyze(
        self,
        title: str,
        base_score: float,
        comments: list[str],
    ) -> dict:
        """
        Run the agent on a single post.

        Args:
            title:       The Reddit post title (clean_title from TSV)
            base_score:  The CLIP model's fake probability (0.0 – 1.0)
            comments:    List of comment strings for this post

        Returns:
            dict with keys:
                crowd_signal          str   "fake" | "real" | "neutral"
                confidence_adjustment float clipped to [-0.25, 0.25]
                final_score           float base_score + adjustment, clipped to [0, 1]
                rationale             str   one-sentence explanation
                had_comments          bool  False if no comments were available
        """
        had_comments = bool(comments)

        client = _get_client()
        user_msg = _build_user_message(title, base_score, comments)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        parsed = _parse_response(raw)

        # Clamp adjustment to safe range
        adj = float(parsed.get("confidence_adjustment", 0.0))
        adj = max(-0.25, min(0.25, adj))

        final = float(base_score) + adj
        final = max(0.0, min(1.0, final))

        return {
            "crowd_signal": parsed.get("crowd_signal", "neutral"),
            "confidence_adjustment": round(adj, 4),
            "final_score": round(final, 4),
            "rationale": parsed.get("rationale", ""),
            "had_comments": had_comments,
        }

    def analyze_batch(
        self,
        posts: list[dict],
    ) -> list[dict]:
        """
        Run the agent on a list of posts.

        Each post dict should have:
            title       str
            base_score  float
            comments    list[str]

        Returns a list of result dicts in the same order.
        """
        results = []
        for post in posts:
            try:
                result = self.analyze(
                    title=post["title"],
                    base_score=post["base_score"],
                    comments=post.get("comments", []),
                )
            except Exception as e:
                result = {
                    "crowd_signal": "neutral",
                    "confidence_adjustment": 0.0,
                    "final_score": post.get("base_score", 0.5),
                    "rationale": f"Agent error: {e}",
                    "had_comments": False,
                }
            results.append(result)
        return results