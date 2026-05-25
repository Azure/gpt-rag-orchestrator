"""Tests for strategy-aware startup warmup behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from startup_warmup import (
    prewarm_agents_for_strategy,
    resolve_agent_strategy,
    should_create_reusable_agent,
    should_prewarm_agent_service,
)


def _cfg(strategy=None):
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None, type=str: {
        "AGENT_STRATEGY": strategy,
    }.get(key, default)
    return cfg


def test_resolve_agent_strategy_defaults_to_maf_lite():
    assert resolve_agent_strategy(_cfg(None)) == "maf_lite"


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("single_agent_rag", True),
        ("maf_agent_service", True),
        ("nl2sql", True),
        ("maf_lite", False),
        ("mcp", False),
        ("multimodal", False),
        ("unknown", False),
    ],
)
def test_should_prewarm_agent_service(strategy, expected):
    assert should_prewarm_agent_service(strategy) is expected


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("single_agent_rag", True),
        ("maf_agent_service", False),
        ("nl2sql", False),
        ("maf_lite", False),
    ],
)
def test_should_create_reusable_agent(strategy, expected):
    assert should_create_reusable_agent(strategy) is expected


@pytest.mark.asyncio
async def test_maf_lite_skips_agents_client_prewarm():
    prewarm = AsyncMock()
    assert await prewarm_agents_for_strategy(_cfg("maf_lite"), prewarm) is False

    prewarm.assert_not_called()


@pytest.mark.asyncio
async def test_single_agent_rag_prewarms_and_creates_reusable_agent():
    prewarm = AsyncMock()
    assert await prewarm_agents_for_strategy(_cfg("single_agent_rag"), prewarm) is True

    prewarm.assert_awaited_once_with(create_reusable_agent=True)


@pytest.mark.asyncio
async def test_agent_service_strategy_prewarms_transport_only():
    prewarm = AsyncMock()
    assert await prewarm_agents_for_strategy(_cfg("maf_agent_service"), prewarm) is True

    prewarm.assert_awaited_once_with(create_reusable_agent=False)
