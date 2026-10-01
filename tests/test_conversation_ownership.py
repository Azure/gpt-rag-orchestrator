"""A client-supplied conversation_id owned by another principal is never adopted."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestration import orchestrator as orchestration
from test_audit_lifecycle import Strategy

FOREIGN_ID = "foreign-conversation"


@pytest.fixture
async def pending_writes():
    before = set(orchestration._persistence_tasks)
    yield
    await asyncio.gather(
        *(orchestration._persistence_tasks - before), return_exceptions=True
    )


@pytest.fixture
def make_orchestrator(patch_dependencies, mock_config, mock_cosmos, monkeypatch):
    async def make(*, conversation_id=FOREIGN_ID, principal="attacker"):
        async def flow():
            yield "answer"

        value = Strategy("mcp", flow)
        value.set_context = MagicMock()
        monkeypatch.setattr(orchestration, "get_config", lambda: mock_config)
        monkeypatch.setattr(orchestration, "get_cosmosdb_client", lambda: mock_cosmos)
        monkeypatch.setattr(
            orchestration.AgentStrategyFactory, "get_strategy",
            AsyncMock(return_value=value),
        )
        mock_cosmos.update_document = AsyncMock(return_value={"id": "x"})
        mock_cosmos.create_document = AsyncMock(return_value={"id": "x"})
        return await orchestration.Orchestrator.create(
            conversation_id=conversation_id,
            user_context={"principal_id": principal},
        )
    return make


async def test_create_does_not_scope_retrieval_to_client_supplied_id(
    make_orchestrator,
):
    instance = await make_orchestrator()
    instance.agentic_strategy.set_context.assert_called_once_with(None)


@pytest.mark.parametrize("principal", ["attacker", "anonymous"])
async def test_foreign_conversation_id_is_replaced(
    make_orchestrator, mock_cosmos, pending_writes, principal,
):
    mock_cosmos.document_id_exists = AsyncMock(return_value=True)
    instance = await make_orchestrator(principal=principal)

    chunks = [c async for c in instance.stream_response("question")]

    assert instance.conversation_id != FOREIGN_ID
    assert chunks[0] == f"{instance.conversation_id} "
    assert FOREIGN_ID not in "".join(chunks)
    instance.agentic_strategy.set_context.assert_called_with(instance.conversation_id)
    await asyncio.gather(*orchestration._persistence_tasks, return_exceptions=True)
    created = mock_cosmos.create_document.await_args
    assert created.args[1] == instance.conversation_id
    expected_partition = (
        f"anonymous-{instance.conversation_id}" if principal == "anonymous" else principal
    )
    assert created.kwargs["partition_key"] == expected_partition


async def test_unclaimed_conversation_id_is_kept(
    make_orchestrator, mock_cosmos, pending_writes,
):
    instance = await make_orchestrator()

    chunks = [c async for c in instance.stream_response("question")]

    mock_cosmos.document_id_exists.assert_awaited_once_with(
        instance.database_container, FOREIGN_ID
    )
    assert instance.conversation_id == FOREIGN_ID
    assert chunks[0] == f"{FOREIGN_ID} "
    instance.agentic_strategy.set_context.assert_called_with(FOREIGN_ID)


async def test_owned_conversation_skips_cross_partition_lookup(
    make_orchestrator, mock_cosmos, pending_writes,
):
    mock_cosmos.get_document = AsyncMock(
        return_value={"id": FOREIGN_ID, "principal_id": "attacker"}
    )
    instance = await make_orchestrator()

    [c async for c in instance.stream_response("question")]

    mock_cosmos.document_id_exists.assert_not_awaited()
    assert instance.conversation_id == FOREIGN_ID


async def test_ownership_lookup_failure_fails_closed(
    make_orchestrator, mock_cosmos, pending_writes,
):
    mock_cosmos.document_id_exists = AsyncMock(side_effect=RuntimeError("cosmos down"))
    instance = await make_orchestrator()

    [c async for c in instance.stream_response("question")]

    assert instance.conversation_id != FOREIGN_ID
