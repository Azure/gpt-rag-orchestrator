"""Cosmos SDK failures retain existing results without masking programming errors."""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from azure.core.exceptions import ServiceRequestError
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from connectors import cosmosdb


@pytest.fixture
def cosmos_boundary(mock_config):
    mock_config.get.side_effect = lambda key, default=None, **_kwargs: {
        "DATABASE_ACCOUNT_NAME": "example",
        "DATABASE_NAME": "test",
    }.get(key, default)
    container = MagicMock()
    sdk = MagicMock()
    sdk.get_database_client.return_value.get_container_client.return_value = container
    with (
        patch.object(cosmosdb, "get_config", return_value=mock_config),
        patch.object(cosmosdb, "CosmosClient", return_value=sdk),
    ):
        client = cosmosdb.CosmosDBClient()
    with patch.object(cosmosdb, "get_cosmosdb_client", return_value=client):
        yield client, container


def _failure(kind):
    marker = "synthetic-private-cosmos-detail"
    if kind == "missing":
        return CosmosResourceNotFoundError(status_code=404, message=marker)
    if kind == "forbidden":
        return CosmosHttpResponseError(status_code=403, message=marker)
    if kind == "transport":
        return ServiceRequestError(marker)
    if kind == "cancelled":
        return asyncio.CancelledError(marker)
    return RuntimeError(marker)


@pytest.mark.parametrize("operation", ["get", "create", "update"])
@pytest.mark.parametrize("kind", ["missing", "forbidden", "transport", "unexpected", "cancelled"])
async def test_cosmos_document_failures_preserve_sdk_result_and_unexpected_propagation(
    cosmos_boundary, operation, kind, caplog,
):
    caplog.set_level("DEBUG")
    client, container = cosmos_boundary
    failure = _failure(kind)
    body = {"id": "conversation-1", "principal_id": "principal-1"}
    method = {"get": "read_item", "create": "create_item", "update": "replace_item"}[operation]
    sdk_call = AsyncMock(side_effect=failure)
    setattr(container, method, sdk_call)
    if operation == "get":
        call = client.get_document("conversations", body["id"], partition_key=body["principal_id"])
    elif operation == "create":
        call = client.create_document("conversations", body["id"], body, partition_key=body["principal_id"])
    else:
        call = client.update_document("conversations", body)
    if kind in {"unexpected", "cancelled"}:
        with pytest.raises(type(failure)) as raised:
            await call
        assert raised.value is failure
    else:
        assert await call is None
    sdk_call.assert_awaited_once()
    if operation == "get":
        sdk_call.assert_awaited_once_with(item=body["id"], partition_key="principal-1")
    else:
        sent = sdk_call.await_args.kwargs["body"]
        assert sent["id"] == "conversation-1"
        assert sent["principal_id"] == "principal-1"
        assert "lastUpdated" in sent
    assert "synthetic-private-cosmos-detail" not in caplog.text


@pytest.mark.parametrize("operation,stage", [
    ("read_user_conversation", "read"),
    ("update_conversation_name", "read"),
    ("update_conversation_name", "replace"),
    ("soft_delete_conversation", "read"),
    ("soft_delete_conversation", "replace"),
])
@pytest.mark.parametrize("kind", ["missing", "forbidden", "transport", "unexpected", "cancelled"])
async def test_user_conversation_failure_keeps_partition_and_sdk_unavailable_result(
    cosmos_boundary, operation, stage, kind, caplog,
):
    caplog.set_level("DEBUG")
    _, container = cosmos_boundary
    failure = _failure(kind)
    body = {"id": "conversation-1", "principal_id": "principal-1"}
    container.read_item = AsyncMock(
        return_value=body, side_effect=failure if stage == "read" else None)
    container.replace_item = AsyncMock(side_effect=failure)
    args = ("conversation-1", "principal-1")
    if operation == "update_conversation_name":
        args += ("renamed",)
    call = getattr(cosmosdb, operation)(*args)
    if kind in {"unexpected", "cancelled"}:
        with pytest.raises(type(failure)) as raised:
            await call
        assert raised.value is failure
    else:
        assert await call is None
    container.read_item.assert_awaited_once_with(
        item="conversation-1", partition_key="principal-1")
    if stage == "read":
        container.replace_item.assert_not_awaited()
    else:
        container.replace_item.assert_awaited_once_with(item="conversation-1", body=body)
        assert body["principal_id"] == "principal-1"
        assert "lastUpdated" in body
        if operation == "soft_delete_conversation":
            assert body["isDeleted"] is True
            assert body["deletedAt"] == body["lastUpdated"]
        else:
            assert body["name"] == "renamed"
    assert "synthetic-private-cosmos-detail" not in caplog.text


@pytest.mark.parametrize("partition_key", [None, "principal-1"])
async def test_cosmos_read_and_create_keep_partition_compatibility(cosmos_boundary, partition_key):
    client, container = cosmos_boundary
    returned = {"id": "conversation-1"}
    container.read_item = AsyncMock(return_value=returned)
    container.create_item = AsyncMock(return_value=returned)
    assert await client.get_document("conversations", returned["id"], partition_key) is returned
    container.read_item.assert_awaited_once_with(
        item="conversation-1", partition_key=partition_key or "conversation-1")
    assert await client.create_document(
        "conversations", returned["id"], partition_key=partition_key) is returned
    body = container.create_item.await_args.kwargs["body"]
    assert body["id"] == "conversation-1"
    assert body.get("principal_id") == partition_key
    assert "lastUpdated" in body


@pytest.mark.parametrize("operation", ["read_user_conversation", "update_conversation_name"])
async def test_soft_deleted_conversation_is_not_read_or_renamed(cosmos_boundary, operation):
    _, container = cosmos_boundary
    container.read_item = AsyncMock(return_value={"id": "conversation-1", "isDeleted": True})
    container.replace_item = AsyncMock()
    args = ("conversation-1", "principal-1")
    if operation == "update_conversation_name":
        args += ("renamed",)
    assert await getattr(cosmosdb, operation)(*args) is None
    container.replace_item.assert_not_awaited()


@pytest.fixture
def conversation_api(patch_dependencies, cosmos_boundary):
    previous = sys.modules.pop("main", None)
    try:
        with (
            patch("dotenv.load_dotenv", return_value=False),
            patch("telemetry.Telemetry.configure_basic"),
            patch("telemetry.Telemetry.log_log_level_diagnostics"),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"),
            patch("opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor.instrument"),
        ):
            main = importlib.import_module("main")
        main.app.dependency_overrides[main.validate_auth] = lambda: None
        with patch.object(main, "validate_user_access", new=AsyncMock(return_value="principal-1")):
            yield main.app, cosmos_boundary[1]
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


@pytest.mark.parametrize("operation", ["list", "read", "rename", "delete"])
@pytest.mark.parametrize("kind", ["missing", "forbidden", "transport", "unexpected", "cancelled"])
async def test_real_cosmos_to_http_failure_translation(conversation_api, operation, kind, caplog):
    app, container = conversation_api
    failure = _failure(kind)
    container.read_item = AsyncMock(side_effect=failure)
    container.replace_item = AsyncMock()

    async def failed_query():
        raise failure
        yield

    container.query_items.return_value = failed_query()
    method = {"list": "GET", "read": "GET", "rename": "PATCH", "delete": "DELETE"}[operation]
    url = "/conversations" if operation == "list" else "/conversations/conversation-1"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        request = client.request(method, url, json={"name": "renamed"} if operation == "rename" else None)
        if kind == "cancelled":
            with pytest.raises(asyncio.CancelledError) as raised:
                await request
            assert raised.value is failure
        else:
            response = await request
            expected = 500 if operation == "list" or kind == "unexpected" else 404
            assert response.status_code == expected
            assert "synthetic-private-cosmos-detail" not in response.text
    container.replace_item.assert_not_awaited()
    if operation == "list":
        assert container.query_items.call_args.kwargs["partition_key"] == "principal-1"
    else:
        container.read_item.assert_awaited_once_with(
            item="conversation-1", partition_key="principal-1")
    assert "synthetic-private-cosmos-detail" not in caplog.text


@pytest.mark.parametrize("operation", ["read", "rename", "delete"])
async def test_real_conversation_http_rejects_foreign_principal(conversation_api, operation):
    app, container = conversation_api
    container.read_item = AsyncMock(return_value={
        "id": "conversation-1", "principal_id": "different-principal",
    })
    container.replace_item = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.request(
            {"read": "GET", "rename": "PATCH", "delete": "DELETE"}[operation],
            "/conversations/conversation-1",
            json={"name": "renamed"} if operation == "rename" else None,
        )
    assert response.status_code == 403
    container.replace_item.assert_not_awaited()


@pytest.mark.parametrize("operation", ["rename", "delete"])
@pytest.mark.parametrize("kind", ["success", "forbidden", "unexpected", "cancelled"])
async def test_real_conversation_http_mutation_outcome(conversation_api, operation, kind, caplog):
    app, container = conversation_api
    body = {"id": "conversation-1", "principal_id": "principal-1", "name": "original"}
    failure = None if kind == "success" else _failure(kind)
    container.read_item = AsyncMock(return_value=body)
    container.replace_item = AsyncMock(return_value=body, side_effect=failure)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        request = client.request(
            "PATCH" if operation == "rename" else "DELETE",
            "/conversations/conversation-1",
            json={"name": "renamed"} if operation == "rename" else None,
        )
        if kind == "cancelled":
            with pytest.raises(asyncio.CancelledError) as raised:
                await request
            assert raised.value is failure
        else:
            response = await request
            assert response.status_code == (200 if kind == "success" else 500)
            assert "synthetic-private-cosmos-detail" not in response.text
            if kind == "success":
                if operation == "rename":
                    assert response.json() == {
                        "id": "conversation-1", "name": "renamed",
                        "last_updated": body["lastUpdated"].replace("+00:00", "Z"),
                    }
                else:
                    assert response.json() == {
                        "status": "success", "message": "Conversation deleted successfully",
                    }
    assert container.read_item.await_count == 2
    assert all(
        call.kwargs["partition_key"] == "principal-1"
        for call in container.read_item.await_args_list
    )
    container.replace_item.assert_awaited_once_with(item="conversation-1", body=body)
    assert body["principal_id"] == "principal-1"
    assert "synthetic-private-cosmos-detail" not in caplog.text
