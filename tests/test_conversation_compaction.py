"""Tests for conversation document compaction before Cosmos persistence."""

from orchestration.conversation_compaction import (
    ConversationCompactionConfig,
    compact_conversation_for_persistence,
    serialized_size_bytes,
)


def _message(index: int, text: str = "hello") -> dict:
    return {"role": "user" if index % 2 == 0 else "assistant", "text": f"{text}-{index}"}


def test_compaction_leaves_small_document_unchanged():
    doc = {"id": "conv-1", "messages": [_message(0), _message(1)]}

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(max_bytes=100_000, max_messages=10),
    )

    assert compacted == doc
    assert stats["compacted"] is False
    assert compacted is not doc


def test_compaction_prunes_old_messages_by_count_and_preserves_metadata():
    doc = {
        "id": "conv-1",
        "name": "Important chat",
        "principal_id": "user-1",
        "thread_id": "thread-1",
        "messages": [_message(i) for i in range(6)],
    }

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(max_bytes=100_000, max_messages=4),
    )

    assert [m["text"] for m in compacted["messages"]] == [
        "hello-2",
        "hello-3",
        "hello-4",
        "hello-5",
    ]
    assert compacted["id"] == "conv-1"
    assert compacted["name"] == "Important chat"
    assert compacted["principal_id"] == "user-1"
    assert compacted["thread_id"] == "thread-1"
    assert compacted["history_truncated"] is True
    assert compacted["history_pruned_message_count"] == 2
    assert stats["pruned_messages"] == 2


def test_compaction_prunes_by_serialized_bytes():
    doc = {
        "id": "conv-1",
        "messages": [_message(i, text="x" * 2_000) for i in range(10)],
    }

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(max_bytes=5_500, max_messages=20),
    )

    assert serialized_size_bytes(compacted) <= 5_500
    assert len(compacted["messages"]) < len(doc["messages"])
    assert stats["pruned_messages"] > 0
    assert compacted["history_pruned_byte_count"] > 0


def test_compaction_prunes_questions_with_messages():
    doc = {
        "id": "conv-1",
        "messages": [_message(i) for i in range(8)],
        "questions": [
            {"question_id": f"q{i}", "text": f"question {i}"}
            for i in range(8)
        ],
    }

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(max_bytes=100_000, max_messages=4),
    )

    assert [q["question_id"] for q in compacted["questions"]] == ["q6", "q7"]
    assert compacted["history_pruned_question_count"] == 6
    assert stats["pruned_questions"] == 6


def test_compaction_prunes_questions_when_message_limit_allows_no_questions():
    doc = {
        "id": "conv-1",
        "messages": [_message(i) for i in range(2)],
        "questions": [{"question_id": "q1", "text": "question"}],
    }

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(max_bytes=100_000, max_messages=1),
    )

    assert len(compacted["messages"]) == 1
    assert compacted["questions"] == []
    assert stats["pruned_questions"] == 1


def test_compaction_can_be_disabled():
    doc = {"id": "conv-1", "messages": [_message(i) for i in range(10)]}

    compacted, stats = compact_conversation_for_persistence(
        doc,
        ConversationCompactionConfig(enabled=False, max_bytes=1, max_messages=1),
    )

    assert compacted is doc
    assert stats["compacted"] is False
