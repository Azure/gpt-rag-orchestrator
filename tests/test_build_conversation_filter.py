"""Regression tests for ``build_conversation_filter``.

The orchestrator scopes AI Search retrieval to the current chat plus shared
(global) corpora. Global chunks have historically been ingested with two
different sentinels for "no specific conversation":

- ``conversationId == 'NaN'`` (string sentinel), and
- ``conversationId == null`` (unset field).

The original filter only matched ``'NaN'`` for the shared scope, so documents
ingested into the global corpus with a null ``conversationId`` were never
returned by any RAG strategy (they all funnel through this single helper).
These tests pin the behaviour that both representations count as shared.
"""

from connectors.search import build_conversation_filter


def test_no_conversation_id_matches_both_shared_sentinels():
    f = build_conversation_filter(None)
    assert f == "(conversationId eq 'NaN' or conversationId eq null)"


def test_empty_conversation_id_is_treated_as_no_id():
    assert build_conversation_filter("") == build_conversation_filter(None)
    assert build_conversation_filter("   ") == build_conversation_filter(None)


def test_with_conversation_id_includes_chat_and_shared_scope():
    f = build_conversation_filter("abc123")
    assert f == (
        "conversationId eq 'abc123' "
        "or (conversationId eq 'NaN' or conversationId eq null)"
    )


def test_null_clause_is_present_so_global_null_docs_are_visible():
    # Core of the bug fix: globally-ingested docs stored with null must match.
    assert "conversationId eq null" in build_conversation_filter(None)
    assert "conversationId eq null" in build_conversation_filter("conv-xyz")


def test_custom_field_name_is_honoured():
    f = build_conversation_filter("c1", field_name="convId")
    assert f == "convId eq 'c1' or (convId eq 'NaN' or convId eq null)"


def test_blank_field_name_falls_back_to_default():
    f = build_conversation_filter(None, field_name="   ")
    assert f == "(conversationId eq 'NaN' or conversationId eq null)"


def test_conversation_id_with_single_quote_is_escaped():
    f = build_conversation_filter("o'brien")
    assert f.startswith("conversationId eq 'o''brien'")
