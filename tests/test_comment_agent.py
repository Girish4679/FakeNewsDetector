"""
Tests for the comment agent and comment loader.

Run all tests:
    python tests/test_comment_agent.py

Run just the offline tests (no API key needed):
    python tests/test_comment_agent.py --offline

Run a live API test (requires ANTHROPIC_API_KEY):
    python tests/test_comment_agent.py --live
"""

import argparse
import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Make sure src/ is importable regardless of where you run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.comment_agent import CommentAgent, _parse_response, _build_user_message
from src.comment_loader import CommentLoader, NoOpCommentLoader


# ════════════════════════════════════════════════════════════════════════════
# OFFLINE TESTS  (no API key, no files needed)
# ════════════════════════════════════════════════════════════════════════════

class TestParseResponse(unittest.TestCase):
    """Tests for the JSON parser — fully offline."""

    def test_clean_json(self):
        raw = '{"crowd_signal": "fake", "confidence_adjustment": 0.2, "rationale": "Comments debunked it."}'
        result = _parse_response(raw)
        self.assertEqual(result["crowd_signal"], "fake")
        self.assertAlmostEqual(result["confidence_adjustment"], 0.2)

    def test_strips_markdown_fences(self):
        raw = '```json\n{"crowd_signal": "real", "confidence_adjustment": -0.1, "rationale": "Crowd agrees."}\n```'
        result = _parse_response(raw)
        self.assertEqual(result["crowd_signal"], "real")

    def test_malformed_json_returns_neutral(self):
        raw = "Sorry, I cannot analyze this."
        result = _parse_response(raw)
        self.assertEqual(result["crowd_signal"], "neutral")
        self.assertEqual(result["confidence_adjustment"], 0.0)

    def test_empty_string_returns_neutral(self):
        result = _parse_response("")
        self.assertEqual(result["crowd_signal"], "neutral")


class TestBuildUserMessage(unittest.TestCase):
    """Tests for prompt construction — fully offline."""

    def test_includes_title(self):
        msg = _build_user_message("Senator stole funds", 0.73, ["This is fake"])
        self.assertIn("Senator stole funds", msg)

    def test_includes_base_score(self):
        msg = _build_user_message("Test title", 0.85, [])
        self.assertIn("0.85", msg)

    def test_handles_empty_comments(self):
        msg = _build_user_message("Title", 0.5, [])
        self.assertIn("no comments available", msg)

    def test_caps_at_15_comments(self):
        many_comments = [f"Comment {i}" for i in range(30)]
        msg = _build_user_message("Title", 0.5, many_comments)
        # Only 15 should be numbered in the output
        self.assertIn("15.", msg)
        self.assertNotIn("16.", msg)


class TestCommentAgentMocked(unittest.TestCase):
    """Tests for CommentAgent with the API mocked — no API key needed."""

    def _make_mock_response(self, crowd_signal, adjustment, rationale):
        """Build a fake Anthropic API response object."""
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock()]
        mock_resp.content[0].text = json.dumps({
            "crowd_signal": crowd_signal,
            "confidence_adjustment": adjustment,
            "rationale": rationale,
        })
        return mock_resp

    @patch("src.comment_agent._get_client")
    def test_fake_signal_increases_score(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            "fake", 0.15, "Commenters said the image is recycled from 2019."
        )

        agent = CommentAgent()
        result = agent.analyze(
            title="Flood devastates city",
            base_score=0.60,
            comments=["This photo is from the 2019 flood, not current events"],
        )

        self.assertEqual(result["crowd_signal"], "fake")
        self.assertAlmostEqual(result["confidence_adjustment"], 0.15)
        self.assertAlmostEqual(result["final_score"], 0.75)
        self.assertTrue(result["had_comments"])

    @patch("src.comment_agent._get_client")
    def test_real_signal_decreases_score(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            "real", -0.20, "Multiple credible sources confirmed this is accurate."
        )

        agent = CommentAgent()
        result = agent.analyze(
            title="Local charity raises $1M",
            base_score=0.55,
            comments=["I was there, this is real", "Reuters covered this too"],
        )

        self.assertEqual(result["crowd_signal"], "real")
        self.assertAlmostEqual(result["confidence_adjustment"], -0.20)
        self.assertAlmostEqual(result["final_score"], 0.35)

    @patch("src.comment_agent._get_client")
    def test_adjustment_clamped_at_upper_bound(self, mock_get_client):
        """Even if the model returns 0.99, we clamp to 0.25."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            "fake", 0.99, "Extreme fake signal."
        )

        agent = CommentAgent()
        result = agent.analyze("Title", 0.80, ["Clearly fake!!!"])
        self.assertLessEqual(result["confidence_adjustment"], 0.25)
        self.assertLessEqual(result["final_score"], 1.0)

    @patch("src.comment_agent._get_client")
    def test_adjustment_clamped_at_lower_bound(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            "real", -0.99, "Extreme real signal."
        )

        agent = CommentAgent()
        result = agent.analyze("Title", 0.10, ["Definitely real"])
        self.assertGreaterEqual(result["confidence_adjustment"], -0.25)
        self.assertGreaterEqual(result["final_score"], 0.0)

    @patch("src.comment_agent._get_client")
    def test_no_comments_returns_neutral(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response(
            "neutral", 0.0, "No comments to analyze."
        )

        agent = CommentAgent()
        result = agent.analyze("Title", 0.70, comments=[])
        self.assertFalse(result["had_comments"])
        self.assertEqual(result["crowd_signal"], "neutral")

    @patch("src.comment_agent._get_client")
    def test_api_error_handled_gracefully_in_batch(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Network error")

        agent = CommentAgent()
        results = agent.analyze_batch([
            {"title": "Test", "base_score": 0.5, "comments": ["comment"]}
        ])
        # Should not raise; should return a safe fallback
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["crowd_signal"], "neutral")
        self.assertIn("Agent error", results[0]["rationale"])


class TestNoOpCommentLoader(unittest.TestCase):
    """Tests for the fallback loader — fully offline."""

    def test_returns_empty_list(self):
        loader = NoOpCommentLoader()
        self.assertEqual(loader.get("any_id"), [])

    def test_has_comments_always_false(self):
        loader = NoOpCommentLoader()
        self.assertFalse(loader.has_comments("any_id"))


# ════════════════════════════════════════════════════════════════════════════
# LIVE API TEST  (requires ANTHROPIC_API_KEY)
# ════════════════════════════════════════════════════════════════════════════

def run_live_test():
    """
    Sends a real request to the Anthropic API.
    Only run this manually: python tests/test_comment_agent.py --live
    """
    print("\n" + "="*60)
    print("LIVE API TEST — requires ANTHROPIC_API_KEY")
    print("="*60)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌  ANTHROPIC_API_KEY not set. Skipping live test.")
        return

    agent = CommentAgent()

    # Test Case 1: Comments strongly suggest fake
    print("\n[Test 1] Comments suggest FAKE:")
    result = agent.analyze(
        title="Shocking photo shows senator accepting bribe",
        base_score=0.65,
        comments=[
            "This photo is from a 2016 movie set, not real life",
            "Already debunked by Snopes: snopes.com/senator-bribe",
            "The original image was taken completely out of context",
            "I've seen this circulating since 2018, it's always fake",
        ],
    )
    print(f"  crowd_signal:          {result['crowd_signal']}")
    print(f"  confidence_adjustment: {result['confidence_adjustment']:+.3f}")
    print(f"  final_score:           {result['final_score']:.3f}  (was 0.650)")
    print(f"  rationale:             {result['rationale']}")

    # Test Case 2: Comments suggest real
    print("\n[Test 2] Comments suggest REAL:")
    result = agent.analyze(
        title="Local firefighters rescue family from burning building",
        base_score=0.40,
        comments=[
            "I live two blocks away, this definitely happened",
            "My cousin is one of the firefighters in this photo",
            "Was covered on local news channel 7 last night",
        ],
    )
    print(f"  crowd_signal:          {result['crowd_signal']}")
    print(f"  confidence_adjustment: {result['confidence_adjustment']:+.3f}")
    print(f"  final_score:           {result['final_score']:.3f}  (was 0.400)")
    print(f"  rationale:             {result['rationale']}")

    # Test Case 3: No comments
    print("\n[Test 3] No comments:")
    result = agent.analyze(
        title="Breaking news: something happened",
        base_score=0.55,
        comments=[],
    )
    print(f"  crowd_signal:          {result['crowd_signal']}")
    print(f"  confidence_adjustment: {result['confidence_adjustment']:+.3f}")
    print(f"  final_score:           {result['final_score']:.3f}  (was 0.550)")
    print(f"  had_comments:          {result['had_comments']}")
    print(f"  rationale:             {result['rationale']}")

    print("\n✅  Live tests complete.\n")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Run only offline unit tests (default)")
    parser.add_argument("--live", action="store_true",
                        help="Also run live API test (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    # Always run offline tests
    print("Running offline unit tests...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestParseResponse))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildUserMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestCommentAgentMocked))
    suite.addTests(loader.loadTestsFromTestCase(TestNoOpCommentLoader))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if args.live:
        run_live_test()

    sys.exit(0 if result.wasSuccessful() else 1)