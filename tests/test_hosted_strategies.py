"""Tests for src/strategies/hosted_strategies.py.

Covers the stateless hosted-runtime helpers: request-local conversation
construction (no service-managed thread/backend tagging) and the ordered
history replay used to keep the underlying model call fully stateless.
"""

from agent_framework import Role

from strategies.hosted_strategies import (
    build_hosted_conversation,
    build_stateless_messages,
)


class TestBuildHostedConversation:
    def test_never_tags_thread_id_or_agent_backend(self):
        """Regardless of strategy key, the hosted conversation dict must
        never carry a service-managed thread/backend identity -- the hosted
        runtime performs zero managed-Conversations data-plane operations."""
        for strategy_key in ("single_agent_rag", "maf_agent_service", "maf_lite", "mcp"):
            conv = build_hosted_conversation(
                strategy_key,
                "conv-id",
                [{"role": "user", "text": "hi"}],
            )
            assert conv == {
                "id": "conv-id",
                "messages": [{"role": "user", "text": "hi"}],
            }
            assert "thread_id" not in conv
            assert "agent_backend" not in conv

    def test_copies_messages_so_mutation_cannot_leak(self):
        source = [{"role": "user", "text": "hi"}]
        conv = build_hosted_conversation("maf_lite", "conv-id", source)

        conv["messages"].append({"role": "assistant", "text": "leaked?"})

        assert source == [{"role": "user", "text": "hi"}]


class TestBuildStatelessMessages:
    def test_replays_ordered_history_then_current_ask(self):
        history = [
            {"role": "user", "text": "first question"},
            {"role": "assistant", "text": "first answer"},
        ]
        messages = build_stateless_messages(history, "follow-up")

        assert [(m.role, m.text) for m in messages] == [
            (Role.USER, "first question"),
            (Role.ASSISTANT, "first answer"),
            (Role.USER, "follow-up"),
        ]

    def test_maps_system_role(self):
        history = [{"role": "system", "text": "be concise"}]
        messages = build_stateless_messages(history, "hi")

        assert messages[0].role == Role.SYSTEM

    def test_skips_unknown_role_and_empty_text(self):
        history = [
            {"role": "tool", "text": "ignored"},
            {"role": "assistant", "text": ""},
            {"role": "assistant", "text": None},
            {"role": "user", "text": "kept"},
        ]
        messages = build_stateless_messages(history, "current ask")

        assert [(m.role, m.text) for m in messages] == [
            (Role.USER, "kept"),
            (Role.USER, "current ask"),
        ]

    def test_empty_history_still_yields_current_ask(self):
        messages = build_stateless_messages([], "only turn")

        assert [(m.role, m.text) for m in messages] == [(Role.USER, "only turn")]
