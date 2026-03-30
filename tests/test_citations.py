"""Tests for the citation processing utilities (src/util/citations.py)."""

from unittest.mock import MagicMock

from util.citations import (
    CITATION_PLACEHOLDER_PATTERN,
    process_bing_citations,
    truncate_title,
)


# ---------------------------------------------------------------------------
# truncate_title
# ---------------------------------------------------------------------------

class TestTruncateTitle:
    def test_none_returns_none(self):
        assert truncate_title(None) is None

    def test_empty_string(self):
        assert truncate_title("") == ""

    def test_short_string_unchanged(self):
        assert truncate_title("Hello") == "Hello"

    def test_exact_length(self):
        title = "a" * 30
        assert truncate_title(title, 30) == title

    def test_truncates_at_space(self):
        title = "Hello World this is a long title with words"
        result = truncate_title(title, 20)
        assert result.endswith("...")
        assert len(result) <= 23  # 20 + '...'

    def test_truncates_without_space(self):
        title = "abcdefghijklmnopqrstuvwxyz1234567890"
        result = truncate_title(title, 10)
        assert result == "abcdefghij..."


# ---------------------------------------------------------------------------
# CITATION_PLACEHOLDER_PATTERN
# ---------------------------------------------------------------------------

class TestCitationPattern:
    def test_matches_standard_placeholder(self):
        assert CITATION_PLACEHOLDER_PATTERN.search("【3:0†source】")

    def test_matches_multi_digit(self):
        assert CITATION_PLACEHOLDER_PATTERN.search("【12:5†some file】")

    def test_no_match_on_normal_text(self):
        assert CITATION_PLACEHOLDER_PATTERN.search("Hello world") is None

    def test_sub_removes_placeholders(self):
        text = "Hello 【7:0†source】 world 【2:1†ref】"
        cleaned = CITATION_PLACEHOLDER_PATTERN.sub("", text)
        assert cleaned == "Hello  world "


# ---------------------------------------------------------------------------
# process_bing_citations
# ---------------------------------------------------------------------------

def _make_delta(text, annotations=None):
    """Build a mock MessageDeltaChunk for testing."""
    delta = MagicMock()
    delta.text = text

    # Build nested structure expected by process_bing_citations
    if annotations:
        piece = MagicMock()
        txt_obj = MagicMock()
        txt_obj.annotations = annotations
        piece.text = txt_obj

        raw = MagicMock()
        raw.content = [piece]
        delta.delta = raw
    else:
        delta.delta = None

    return delta


class TestProcessBingCitations:
    def test_none_text(self):
        delta = _make_delta(None)
        assert process_bing_citations(delta) is None

    def test_no_annotations_strips_placeholders(self):
        delta = _make_delta("Hello 【3:0†source】 world")
        result = process_bing_citations(delta)
        assert "【" not in result
        assert result == "Hello  world"

    def test_url_citation_replaced(self):
        ann = {
            "type": "url_citation",
            "text": "【1:0†source】",
            "url_citation": {
                "url": "https://example.com",
                "title": "Example",
            },
        }
        delta = _make_delta("Check 【1:0†source】 here", annotations=[ann])
        result = process_bing_citations(delta)
        assert "[Example](https://example.com)" in result
        assert "【" not in result

    def test_url_citation_no_title_uses_url(self):
        ann = {
            "type": "url_citation",
            "text": "【2:0†ref】",
            "url_citation": {
                "url": "https://example.com/page",
                "title": None,
            },
        }
        delta = _make_delta("See 【2:0†ref】", annotations=[ann])
        result = process_bing_citations(delta)
        assert "https://example.com/page" in result
