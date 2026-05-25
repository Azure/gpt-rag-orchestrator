"""Startup warmup helpers for strategy-specific dependencies."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any


SINGLE_AGENT_RAG = "single_agent_rag"
MAF_AGENT_SERVICE = "maf_agent_service"
MAF_LITE = "maf_lite"
NL2SQL = "nl2sql"


AGENT_SERVICE_PREWARM_STRATEGIES = {
    SINGLE_AGENT_RAG,
    MAF_AGENT_SERVICE,
    NL2SQL,
}

REUSABLE_AGENT_PREWARM_STRATEGIES = {
    SINGLE_AGENT_RAG,
}

PrewarmAgentsClient = Callable[..., Awaitable[None]]


def resolve_agent_strategy(cfg: Any) -> str:
    """Resolve the active strategy using the same default as Orchestrator."""
    value = cfg.get("AGENT_STRATEGY", MAF_LITE)
    return str(value or MAF_LITE).strip().lower()


def should_prewarm_agent_service(strategy_name: str) -> bool:
    return strategy_name in AGENT_SERVICE_PREWARM_STRATEGIES


def should_create_reusable_agent(strategy_name: str) -> bool:
    return strategy_name in REUSABLE_AGENT_PREWARM_STRATEGIES


async def prewarm_agents_for_strategy(
    cfg: Any,
    prewarm_agents_client_func: PrewarmAgentsClient | None = None,
) -> bool:
    """Pre-warm Agent Service only when the active strategy uses it."""
    strategy_name = resolve_agent_strategy(cfg)
    if not should_prewarm_agent_service(strategy_name):
        logging.info(
            "[Startup] Skipping AgentsClient pre-warm for AGENT_STRATEGY=%s",
            strategy_name,
        )
        return False

    if prewarm_agents_client_func is None:
        from strategies.single_agent_rag_strategy_v2 import prewarm_agents_client

        prewarm_agents_client_func = prewarm_agents_client

    create_reusable_agent = should_create_reusable_agent(strategy_name)
    await prewarm_agents_client_func(create_reusable_agent=create_reusable_agent)
    logging.info(
        "[Startup] AgentsClient pre-warm completed for AGENT_STRATEGY=%s "
        "(create_reusable_agent=%s)",
        strategy_name,
        create_reusable_agent,
    )
    return True
