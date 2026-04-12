"""
Comment loader for Fakeddit.

Fakeddit ships a separate comments.tsv file where each row is one comment,
linked to a post via the `submission_id` column.

This module reads that file once and provides fast O(1) lookups by post ID.

Usage:
    from src.comment_loader import CommentLoader

    loader = CommentLoader("data/comments.tsv")
    comments = loader.get(post_id="abc123")
    # ["Great point!", "This is fake, debunked here: snopes.com/..."]
"""

from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd


class CommentLoader:
    """
    Loads comments.tsv into memory once and supports fast per-post lookups.

    Args:
        comments_tsv:  Path to the Fakeddit comments.tsv file
        text_col:      Column name containing the comment text (default: "body")
        id_col:        Column linking comments to posts (default: "submission_id")
        max_per_post:  Maximum comments to keep per post (default: 15)
        min_length:    Minimum character length to keep a comment (filters out
                       deleted/removed comments like "[deleted]" or "[removed]")
    """

    DELETED_TOKENS = {"[deleted]", "[removed]", ""}

    def __init__(
        self,
        comments_tsv: str,
        text_col: str = "body",
        id_col: str = "submission_id",
        max_per_post: int = 15,
        min_length: int = 10,
    ):
        self.max_per_post = max_per_post
        self._index: dict[str, list[str]] = defaultdict(list)

        path = Path(comments_tsv)
        if not path.exists():
            raise FileNotFoundError(
                f"comments.tsv not found at {path}.\n"
                f"Download it from the Fakeddit Google Drive under the comments folder."
            )

        print(f"Loading comments from {path} ...")
        df = pd.read_csv(path, sep="\t", usecols=[id_col, text_col])
        df = df.fillna("")
        df[text_col] = df[text_col].astype(str).str.strip()

        # Filter out deleted/removed/too-short comments
        mask = (
            ~df[text_col].isin(self.DELETED_TOKENS)
            & (df[text_col].str.len() >= min_length)
        )
        df = df[mask]

        # Build index: submission_id → list of comment strings
        for _, row in df.iterrows():
            sid = str(row[id_col])
            if len(self._index[sid]) < max_per_post:
                self._index[sid].append(row[text_col])

        total_posts = len(self._index)
        total_comments = sum(len(v) for v in self._index.values())
        print(f"  Indexed {total_comments:,} comments across {total_posts:,} posts")

    def get(self, post_id: str) -> list[str]:
        """Return list of comment strings for the given post ID (may be empty)."""
        return self._index.get(str(post_id), [])

    def has_comments(self, post_id: str) -> bool:
        return str(post_id) in self._index

    def __len__(self) -> int:
        return len(self._index)


# ── Fallback stub for when comments.tsv isn't available ──────────────────────
class NoOpCommentLoader:
    """
    Drop-in replacement for CommentLoader when comments.tsv is not available.
    Always returns empty comment lists so the agent gracefully handles it.
    """

    def get(self, post_id: str) -> list[str]:
        return []

    def has_comments(self, post_id: str) -> bool:
        return False

    def __len__(self) -> int:
        return 0