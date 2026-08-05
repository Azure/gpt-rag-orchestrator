"""Hosted runtime strategy guard.

Only the strategies listed in ``HOSTED_ELIGIBLE_STRATEGIES`` are approved for
the shared hosted runtime (Foundry hosted agents, no-panel mode).  Any other
strategy must fail explicitly rather than silently falling back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from strategies.agent_provider_v2 import AGENT_BACKEND_TAG

# ADR-eligible strategies for the hosted runtime.
# multimodal and nl2sql are excluded pending further ADR investigation.
HOSTED_ELIGIBLE_STRATEGIES: frozenset[str] = frozenset({
    "maf_lite",
    "maf_agent_service",
    "single_agent_rag",
    "mcp",
})

HOSTED_SERVER_THREAD_STRATEGIES: frozenset[str] = frozenset({
    "maf_agent_service",
    "single_agent_rag",
})

# Strategies that call out to the Toolbox MCP server for retrieval and must
# therefore carry a validated Foundry call id (ADR-0001, Azure/GPT-RAG#591).
# Other hosted-eligible strategies use the classic Foundry IQ / OBO retrieval
# path and are out of scope for this passthrough.
HOSTED_TOOLBOX_STRATEGIES: frozenset[str] = frozenset({
    "mcp",
})


class HostedConversationMessage(TypedDict):
    role: str
    text: str


def build_hosted_conversation(
    strategy_key: str,
    conversation_id: str,
    messages: Sequence[HostedConversationMessage],
) -> dict:
    """Build request-local strategy state from a managed Conversation.

    The complete ordered prior history is copied so strategy mutations cannot
    leak into another request. Foundry Responses-backed strategies reuse the
    managed conversation id as their stable server-side thread id.
    """
    conversation = {
        "id": conversation_id,
        "messages": [dict(message) for message in messages],
    }
    if strategy_key in HOSTED_SERVER_THREAD_STRATEGIES:
        conversation["thread_id"] = conversation_id
        conversation["agent_backend"] = AGENT_BACKEND_TAG
    return conversation


def guard_hosted_strategy(key: str) -> None:
    """Raise :class:`ValueError` explicitly when *key* is not hosted-eligible.

    The hosted runtime must never silently fall back to an unsupported strategy.
    Callers that need a user-friendly HTTP error should catch this and map it to
    a 422 or 400 response.
    """
    if key not in HOSTED_ELIGIBLE_STRATEGIES:
        raise ValueError(
            f"Strategy '{key}' is not supported in the hosted runtime. "
            f"Eligible strategies: {sorted(HOSTED_ELIGIBLE_STRATEGIES)}"
        )
