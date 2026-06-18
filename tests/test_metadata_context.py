"""Tests for surfacing indexed ``custom_metadata`` into the LLM context.

Covers Azure/GPT-RAG#506:

- The pure formatting helper (``format_custom_metadata`` / ``parse_allowed_keys``).
- The default-off flag wiring in the connector path (``search_knowledge_base``):
  when off, ``custom_metadata`` is not selected and the output is unchanged;
  when on, the formatted block is prepended to each document's content.
- Parity in one context provider (``SearchContextProvider``).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from util.metadata import format_custom_metadata, parse_allowed_keys, METADATA_HEADER


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_list_of_dicts_input_renders_sorted_block():
    raw = [
        {"key": "author", "value": "Jane Doe"},
        {"key": "department", "value": "Finance"},
    ]
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\nauthor: Jane Doe\ndepartment: Finance"


def test_json_string_input_is_parsed():
    raw = json.dumps([{"key": "author", "value": "Jane"}])
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\nauthor: Jane"


def test_plain_dict_input_supported():
    out = format_custom_metadata({"b": "2", "a": "1"}, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\na: 1\nb: 2"


def test_empty_and_none_values_skipped():
    raw = [
        {"key": "author", "value": "Jane"},
        {"key": "empty", "value": ""},
        {"key": "missing", "value": None},
        {"key": "", "value": "no-key"},
    ]
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\nauthor: Jane"


def test_sorting_is_deterministic():
    raw = [
        {"key": "zeta", "value": "z"},
        {"key": "alpha", "value": "a"},
        {"key": "mid", "value": "m"},
    ]
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\nalpha: a\nmid: m\nzeta: z"


def test_internal_newlines_collapsed():
    raw = [{"key": "note", "value": "line one\nline two   spaced"}]
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=None)
    assert out == f"{METADATA_HEADER}\nnote: line one line two spaced"


def test_truncation_to_max_chars():
    raw = [{"key": "k", "value": "x" * 1000}]
    out = format_custom_metadata(raw, max_chars=50, allowed_keys=None)
    assert len(out) <= 50


def test_allowlist_filters_keys():
    raw = [
        {"key": "author", "value": "Jane"},
        {"key": "secret", "value": "hidden"},
    ]
    out = format_custom_metadata(raw, max_chars=500, allowed_keys=["author"])
    assert out == f"{METADATA_HEADER}\nauthor: Jane"


def test_allowlist_with_no_matches_returns_empty():
    raw = [{"key": "author", "value": "Jane"}]
    assert format_custom_metadata(raw, max_chars=500, allowed_keys=["nope"]) == ""


def test_empty_inputs_return_empty_string():
    assert format_custom_metadata(None, 500, None) == ""
    assert format_custom_metadata([], 500, None) == ""
    assert format_custom_metadata("", 500, None) == ""
    assert format_custom_metadata("not json", 500, None) == ""
    assert format_custom_metadata(123, 500, None) == ""


def test_parse_allowed_keys():
    assert parse_allowed_keys("") == []
    assert parse_allowed_keys(None) == []
    assert parse_allowed_keys("a, b ,c") == ["a", "b", "c"]
    assert parse_allowed_keys(" , ,") == []


# ---------------------------------------------------------------------------
# Connector path: search_knowledge_base
# ---------------------------------------------------------------------------

def _make_search_client(mock_config, extra_cfg, patch_dependencies):
    from connectors.search import SearchClient

    base = {
        "SEARCH_SERVICE_QUERY_ENDPOINT": "https://fake-search.search.windows.net",
        "SEARCH_RAG_INDEX_NAME": "ragindex",
        "SEARCH_APPROACH": "term",
        "ALLOW_ANONYMOUS": "true",
    }
    base.update(extra_cfg)
    mock_config.get.side_effect = lambda key, default=None, type=str: base.get(key, default)
    with patch("connectors.search.get_config", return_value=mock_config):
        client = SearchClient()
    client._get_search_user_token_for_trimming = AsyncMock(return_value=None)
    return client


_DOC = {
    "title": "Doc A",
    "content": "Body text",
    "filepath": "doc-a.pdf",
    "custom_metadata": [
        {"key": "author", "value": "Jane"},
        {"key": "department", "value": "Finance"},
    ],
}


@pytest.mark.asyncio
async def test_flag_off_does_not_select_metadata_and_output_unchanged(patch_dependencies, mock_config):
    client = _make_search_client(mock_config, {}, patch_dependencies)

    search_mock = AsyncMock(return_value={"value": [dict(_DOC)]})
    with patch.object(client, "search", search_mock):
        result_json = await client.search_knowledge_base("hello")

    # Select must be byte-for-byte the original string (no custom_metadata).
    body = search_mock.call_args.kwargs["body"]
    assert body["select"] == "title,content,url,filepath,chunk_id"
    assert "custom_metadata" not in body["select"]

    # Content is unchanged: no metadata block prepended, and no extra JSON key.
    results = json.loads(result_json)["results"]
    assert results[0]["content"] == "Body text"
    assert "custom_metadata" not in results[0]
    assert METADATA_HEADER not in result_json


@pytest.mark.asyncio
async def test_flag_on_selects_metadata_and_prepends_block(patch_dependencies, mock_config):
    client = _make_search_client(
        mock_config,
        {
            "SEARCH_INCLUDE_METADATA_IN_CONTEXT": True,
            "SEARCH_METADATA_MAX_CHARS": 500,
            "SEARCH_METADATA_ALLOWED_KEYS": "",
        },
        patch_dependencies,
    )

    search_mock = AsyncMock(return_value={"value": [dict(_DOC)]})
    with patch.object(client, "search", search_mock):
        result_json = await client.search_knowledge_base("hello")

    body = search_mock.call_args.kwargs["body"]
    assert body["select"] == "title,content,url,filepath,chunk_id,custom_metadata"

    content = json.loads(result_json)["results"][0]["content"]
    expected_block = f"{METADATA_HEADER}\nauthor: Jane\ndepartment: Finance"
    assert content == f"{expected_block}\n\nBody text"


# ---------------------------------------------------------------------------
# Provider parity: SearchContextProvider
# ---------------------------------------------------------------------------

class _AsyncDocs:
    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        async def gen():
            for d in self._docs:
                yield d
        return gen()


class _FakeAzureSearchClient:
    """Stands in for azure.search.documents.aio.SearchClient."""

    last_select = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def search(self, **kwargs):
        type(self).last_select = kwargs.get("select")
        return _AsyncDocs([dict(_DOC)])


@pytest.mark.asyncio
async def test_provider_flag_on_prepends_block(mock_config):
    from strategies.search_context_provider import SearchContextProvider
    from agent_framework import ChatMessage, Role

    mock_config.get.side_effect = lambda key, default=None, type=str: {
        "SEARCH_INCLUDE_METADATA_IN_CONTEXT": True,
        "SEARCH_METADATA_MAX_CHARS": 500,
        "SEARCH_METADATA_ALLOWED_KEYS": "",
    }.get(key, default)

    provider = SearchContextProvider(
        endpoint="https://fake.search.windows.net",
        index_name="ragindex",
        credential=AsyncMock(),
    )

    with (
        patch("strategies.search_context_provider.get_config", return_value=mock_config),
        patch("strategies.search_context_provider.SearchClient", _FakeAzureSearchClient),
    ):
        ctx = await provider.invoking([ChatMessage(role=Role.USER, text="hello")])

    assert "custom_metadata" in _FakeAzureSearchClient.last_select
    text = ctx.messages[0].text
    assert METADATA_HEADER in text
    assert "author: Jane" in text
    assert "department: Finance" in text
