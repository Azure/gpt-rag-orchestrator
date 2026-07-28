"""Hosted runtime strategy guard.

Only the strategies listed in ``HOSTED_ELIGIBLE_STRATEGIES`` are approved for
the shared hosted runtime (Foundry hosted agents, no-panel mode).  Any other
strategy must fail explicitly rather than silently falling back.
"""

from __future__ import annotations

# ADR-eligible strategies for the hosted runtime.
# multimodal and nl2sql are excluded pending further ADR investigation.
HOSTED_ELIGIBLE_STRATEGIES: frozenset[str] = frozenset({
    "maf_lite",
    "maf_agent_service",
    "single_agent_rag",
    "mcp",
})


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
