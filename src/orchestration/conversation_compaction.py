"""Conversation document compaction helpers."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


DEFAULT_MAX_BYTES = 1_500_000
DEFAULT_MAX_PERSISTED_MESSAGES = 200


@dataclass(frozen=True)
class ConversationCompactionConfig:
    enabled: bool = True
    max_bytes: int = DEFAULT_MAX_BYTES
    max_messages: int = DEFAULT_MAX_PERSISTED_MESSAGES


def load_conversation_compaction_config(cfg: Any) -> ConversationCompactionConfig:
    """Load compaction settings from App Configuration-compatible objects."""
    return ConversationCompactionConfig(
        enabled=cfg.get("CONVERSATION_HISTORY_COMPACTION_ENABLED", True, type=bool),
        max_bytes=cfg.get("CONVERSATION_HISTORY_MAX_BYTES", DEFAULT_MAX_BYTES, type=int),
        max_messages=cfg.get(
            "CONVERSATION_HISTORY_MAX_PERSISTED_MESSAGES",
            DEFAULT_MAX_PERSISTED_MESSAGES,
            type=int,
        ),
    )


def serialized_size_bytes(document: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _drop_oldest(items: list[Any], count: int) -> int:
    if count <= 0:
        return 0
    del items[:count]
    return count


def compact_conversation_for_persistence(
    conversation: dict[str, Any],
    config: ConversationCompactionConfig,
) -> tuple[dict[str, Any], dict[str, int | bool]]:
    """Return a copy of a conversation document sized safely for Cosmos DB.

    The helper keeps the latest messages/questions and prunes oldest entries
    first. Metadata such as id, name, principal_id, thread_id, feedback, and
    timestamps is preserved.
    """
    if not config.enabled or not isinstance(conversation, dict):
        return conversation, {
            "compacted": False,
            "pruned_messages": 0,
            "pruned_questions": 0,
            "original_bytes": 0,
            "final_bytes": 0,
        }

    compacted = dict(conversation)
    messages = compacted.get("messages")
    questions = compacted.get("questions")

    if isinstance(messages, list):
        compacted["messages"] = list(messages)
    if isinstance(questions, list):
        compacted["questions"] = list(questions)

    original_bytes = serialized_size_bytes(compacted)
    original_message_count = len(compacted.get("messages", [])) if isinstance(compacted.get("messages"), list) else 0
    original_question_count = len(compacted.get("questions", [])) if isinstance(compacted.get("questions"), list) else 0

    pruned_messages = 0
    pruned_questions = 0
    max_messages = max(0, config.max_messages)
    max_questions = max(0, max_messages // 2)
    max_bytes = max(0, config.max_bytes)

    if max_messages > 0 and isinstance(compacted.get("messages"), list):
        pruned_messages += _drop_oldest(
            compacted["messages"],
            max(0, len(compacted["messages"]) - max_messages),
        )

    if max_messages > 0 and isinstance(compacted.get("questions"), list):
        pruned_questions += _drop_oldest(
            compacted["questions"],
            max(0, len(compacted["questions"]) - max_questions),
        )

    if max_bytes:
        while (
            isinstance(compacted.get("messages"), list)
            and compacted["messages"]
            and serialized_size_bytes(compacted) > max_bytes
        ):
            pruned_messages += _drop_oldest(compacted["messages"], 1)

        while (
            isinstance(compacted.get("questions"), list)
            and compacted["questions"]
            and serialized_size_bytes(compacted) > max_bytes
        ):
            pruned_questions += _drop_oldest(compacted["questions"], 1)

    if pruned_messages or pruned_questions:
        final_bytes_before_metadata = serialized_size_bytes(compacted)
        compacted["history_truncated"] = True
        compacted["history_pruned_message_count"] = (
            _safe_int(compacted.get("history_pruned_message_count")) + pruned_messages
        )
        compacted["history_pruned_question_count"] = (
            _safe_int(compacted.get("history_pruned_question_count")) + pruned_questions
        )
        compacted["history_pruned_byte_count"] = (
            _safe_int(compacted.get("history_pruned_byte_count"))
            + max(0, original_bytes - final_bytes_before_metadata)
        )
        compacted["history_last_pruned_at"] = datetime.now(timezone.utc).isoformat()

        if max_bytes:
            while (
                isinstance(compacted.get("messages"), list)
                and compacted["messages"]
                and serialized_size_bytes(compacted) > max_bytes
            ):
                pruned_messages += _drop_oldest(compacted["messages"], 1)
                compacted["history_pruned_message_count"] += 1

            while (
                isinstance(compacted.get("questions"), list)
                and compacted["questions"]
                and serialized_size_bytes(compacted) > max_bytes
            ):
                pruned_questions += _drop_oldest(compacted["questions"], 1)
                compacted["history_pruned_question_count"] += 1

    final_bytes = serialized_size_bytes(compacted)
    if max_bytes and final_bytes > max_bytes:
        logging.warning(
            "[ConversationCompaction] document still exceeds configured byte limit "
            "after pruning (id=%s final_bytes=%d max_bytes=%d messages=%d/%d questions=%d/%d)",
            compacted.get("id"),
            final_bytes,
            max_bytes,
            len(compacted.get("messages", [])) if isinstance(compacted.get("messages"), list) else 0,
            original_message_count,
            len(compacted.get("questions", [])) if isinstance(compacted.get("questions"), list) else 0,
            original_question_count,
        )
    elif pruned_messages or pruned_questions:
        logging.info(
            "[ConversationCompaction] pruned conversation before persistence "
            "(id=%s messages=%d questions=%d bytes=%d->%d)",
            compacted.get("id"),
            pruned_messages,
            pruned_questions,
            original_bytes,
            final_bytes,
        )

    return compacted, {
        "compacted": bool(pruned_messages or pruned_questions),
        "pruned_messages": pruned_messages,
        "pruned_questions": pruned_questions,
        "original_bytes": original_bytes,
        "final_bytes": final_bytes,
    }
