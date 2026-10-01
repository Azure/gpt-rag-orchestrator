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
from util.conversation_scope import (
    build_conversation_owner_clause,
    resolve_conversation_owner_id,
)

SHARED = "(conversationId eq 'NaN' or conversationId eq null)"


def test_no_conversation_id_matches_both_shared_sentinels():
    f = build_conversation_filter(None)
    assert f == "(conversationId eq 'NaN' or conversationId eq null)"


def test_empty_conversation_id_is_treated_as_no_id():
    assert build_conversation_filter("") == build_conversation_filter(None)
    assert build_conversation_filter("   ") == build_conversation_filter(None)


def test_conversation_id_without_owner_returns_shared_scope_only():
    # SECURITY: a client-chosen conversation id alone is not proof of ownership.
    assert build_conversation_filter("abc123") == SHARED


def test_with_owner_includes_owned_chat_and_shared_scope():
    f = build_conversation_filter("abc123", owner_id="user-a")
    assert f == (
        "(conversationId eq 'abc123' and metadata_security_user_ids/any(u: u eq 'user-a')) "
        f"or {SHARED}"
    )


def test_other_user_filter_does_not_match_first_users_uploads():
    owner_a = build_conversation_filter("abc123", owner_id="user-a")
    owner_b = build_conversation_filter("abc123", owner_id="user-b")
    assert "u eq 'user-a'" in owner_a
    assert "user-a" not in owner_b
    assert "u eq 'user-b'" in owner_b


def test_anonymous_owner_only_matches_chunks_without_acl():
    f = build_conversation_filter("abc123", owner_id="anonymous")
    assert f == (
        "(conversationId eq 'abc123' and not metadata_security_user_ids/any()) "
        f"or {SHARED}"
    )


def test_owner_id_with_single_quote_is_escaped():
    f = build_conversation_filter("c1", owner_id="x' or true or 'y")
    assert "u eq 'x'' or true or ''y'" in f


def test_owner_clause_requires_conversation_and_owner():
    assert build_conversation_owner_clause(None, "user-a") is None
    assert build_conversation_owner_clause("c1", None) is None
    assert build_conversation_owner_clause("c1", "   ") is None


def test_resolve_conversation_owner_id_prefers_principal_id():
    assert resolve_conversation_owner_id({"principal_id": "p1", "oid": "o1"}) == "p1"
    assert resolve_conversation_owner_id({"oid": "o1"}) == "o1"
    assert resolve_conversation_owner_id({"principal_id": "  "}) is None
    assert resolve_conversation_owner_id(None) is None
    assert resolve_conversation_owner_id("not-a-mapping") is None


def test_null_clause_is_present_so_global_null_docs_are_visible():
    # Core of the bug fix: globally-ingested docs stored with null must match.
    assert "conversationId eq null" in build_conversation_filter(None)
    assert "conversationId eq null" in build_conversation_filter("conv-xyz", owner_id="u1")


def test_custom_field_name_is_honoured():
    f = build_conversation_filter("c1", owner_id="u1", field_name="convId")
    assert f == (
        "(convId eq 'c1' and metadata_security_user_ids/any(u: u eq 'u1')) "
        "or (convId eq 'NaN' or convId eq null)"
    )


def test_blank_field_name_falls_back_to_default():
    f = build_conversation_filter(None, field_name="   ")
    assert f == "(conversationId eq 'NaN' or conversationId eq null)"


def test_conversation_id_with_single_quote_is_escaped():
    f = build_conversation_filter("o'brien", owner_id="u1")
    assert f.startswith("(conversationId eq 'o''brien' and")
