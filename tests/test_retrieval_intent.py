"""Tests for retrieval-needed intent helpers."""

from strategies.retrieval_intent import (
    GREETING,
    NO_RETRIEVAL,
    QUESTION,
    build_retrieval_intent_messages,
    format_recent_history,
    parse_retrieval_intent,
)


def test_parse_retrieval_intent_labels():
    assert parse_retrieval_intent("GREETING") == GREETING
    assert parse_retrieval_intent("NO_RETRIEVAL_FOLLOWUP") == NO_RETRIEVAL
    assert parse_retrieval_intent("RETRIEVAL_NEEDED") == QUESTION
    assert parse_retrieval_intent("QUESTION") == QUESTION
    assert parse_retrieval_intent("") == QUESTION


def test_no_retrieval_label_can_be_disabled():
    assert (
        parse_retrieval_intent("NO_RETRIEVAL_FOLLOWUP", enable_no_retrieval=False)
        == QUESTION
    )


def test_parse_retrieval_intent_is_conservative_for_verbose_negative_labels():
    assert parse_retrieval_intent("This is not a greeting. RETRIEVAL_NEEDED.") == QUESTION


def test_format_recent_history_is_bounded_and_keeps_recent_messages():
    history = [
        {"role": "user", "text": "old question"},
        {"role": "assistant", "text": "old answer"},
        {"role": "user", "text": "new question"},
        {"role": "assistant", "text": "new answer"},
    ]

    formatted = format_recent_history(history, max_messages=2, max_chars=1_000)

    assert "old question" not in formatted
    assert "old answer" not in formatted
    assert "new question" in formatted
    assert "new answer" in formatted


def test_format_recent_history_char_limit_prioritizes_most_recent_messages():
    history = [
        {"role": "user", "text": "x" * 500},
        {"role": "assistant", "text": "recent answer"},
    ]

    formatted = format_recent_history(history, max_messages=2, max_chars=80)

    assert "recent answer" in formatted
    assert "x" * 100 not in formatted


def test_build_retrieval_intent_messages_includes_bounded_history_and_current_message():
    history = [
        {"role": "assistant", "text": "Previous answer about refund policy."},
    ]

    messages = build_retrieval_intent_messages(
        user_message="Format that answer as a table.",
        history=history,
        max_history_messages=4,
        max_history_chars=1_000,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "NO_RETRIEVAL_FOLLOWUP" in messages[0]["content"]
    assert "Previous answer about refund policy." in messages[1]["content"]
    assert "Format that answer as a table." in messages[1]["content"]
