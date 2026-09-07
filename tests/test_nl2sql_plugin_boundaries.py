"""Typed NL2SQL tool failures preserve results, scope and cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.nl2sql.plugin import NL2SQLPlugin


@pytest.fixture
def plugin(mock_config):
    cosmos = SimpleNamespace(get_document=AsyncMock())
    search = SimpleNamespace(search=AsyncMock())
    model = SimpleNamespace(get_embeddings=AsyncMock(return_value=[0.1, 0.2]))
    with (
        patch("plugins.nl2sql.plugin.get_config", return_value=mock_config),
        patch("plugins.nl2sql.plugin.get_cosmosdb_client", return_value=cosmos),
        patch("plugins.nl2sql.plugin.get_search_client", return_value=search),
        patch("plugins.nl2sql.plugin.get_genai_client", return_value=model),
    ):
        yield NL2SQLPlugin()


@pytest.mark.parametrize("method,args,collection", [
    ("get_all_tables_info", ("source'quoted",), "tables"),
    ("get_schema_info", ("source'quoted", "table'quoted"), "columns"),
    ("tables_retrieval", ("question", "source'quoted"), "tables"),
    ("measures_retrieval", ("source'quoted",), "measures"),
    ("queries_retrieval", ("question", "source'quoted"), "queries"),
], ids=["get_all_tables_info", "get_schema_info", "tables_retrieval", "measures_retrieval", "queries_retrieval"])
@pytest.mark.parametrize("outcome", ["provider", "malformed", "cancelled", "success"])
async def test_metadata_tools_keep_typed_partial_or_unavailable_results(
    plugin, method, args, collection, outcome, caplog,
):
    marker = "synthetic-private-metadata-error"
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    valid = {
        "table": "table-1", "description": "description", "datasource": "source'quoted",
        "name": "measure-1", "type": "external",
        "question": "sample", "query": "SELECT 1", "reasoning": "example",
        "columns": [{"name": "value", "description": "column"}],
    }
    invalid = {
        **valid, "table": {"detail": marker}, "name": {"detail": marker},
        "question": {"detail": marker},
        "columns": [{"name": "value", "description": {"detail": marker}}],
    }
    if outcome in {"provider", "cancelled"}:
        plugin.search.search.side_effect = failure
    else:
        docs = [valid]
        if outcome == "malformed":
            docs = [invalid] if method == "get_schema_info" else [valid, invalid]
        plugin.search.search.return_value = {"value": docs}
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await getattr(plugin, method)(*args)
        assert raised.value is failure
    else:
        result = await getattr(plugin, method)(*args)
        data = result.model_dump(mode="json")
        if method == "get_schema_info":
            # SchemaInfo has no error field; retain its legacy unavailable shape.
            assert set(data) == {"datasource", "table", "description", "columns"}
            assert data["columns"] == ({"value": "column"} if outcome == "success" else None)
        else:
            assert len(data[collection]) == (0 if outcome == "provider" else 1)
            assert bool(data["error"]) is (outcome != "success")
            if outcome == "provider":
                assert "RuntimeError" in data["error"]
        assert marker not in result.model_dump_json()
    body = plugin.search.search.await_args.kwargs["body"]
    assert "datasource eq 'source''quoted'" in body["filter"]
    if method == "get_schema_info":
        assert "table eq 'table''quoted'" in body["filter"]
    assert marker not in caplog.text


@pytest.mark.parametrize("method", ["tables_retrieval", "queries_retrieval"])
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_metadata_embedding_failure_still_propagates_before_search(plugin, method, failure_type):
    failure = failure_type("synthetic embedding failure")
    model = SimpleNamespace(get_embeddings=AsyncMock(side_effect=failure))
    with patch("plugins.nl2sql.plugin.get_genai_client", return_value=model):
        with pytest.raises(failure_type) as raised:
            await getattr(plugin, method)("question", "source")
    assert raised.value is failure
    plugin.search.search.assert_not_awaited()


@pytest.mark.parametrize("stage", ["configuration", "connect", "execute", "fetch", "cancelled", "success"])
async def test_sql_tool_keeps_explicit_error_result_and_read_only_driver_inputs(
    plugin, stage, caplog,
):
    marker = "synthetic-private-sql-error"
    failure = asyncio.CancelledError(marker) if stage == "cancelled" else RuntimeError(marker)
    plugin.cosmos.get_document.return_value = {
        "id": "source", "description": "test", "type": "sql_database",
        "server": "db.example.invalid", "database": "test",
    }
    cursor = MagicMock()
    cursor.description = [("value",)]
    cursor.fetchall.return_value = [(1,)]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    client = SimpleNamespace(create_connection=AsyncMock(return_value=connection))
    if stage == "configuration":
        plugin.cosmos.get_document.side_effect = failure
    elif stage in {"connect", "cancelled"}:
        client.create_connection.side_effect = failure
    elif stage == "execute":
        cursor.execute.side_effect = failure
    elif stage == "fetch":
        cursor.fetchall.side_effect = failure
    with patch("plugins.nl2sql.plugin.SQLDBClient", return_value=client):
        if stage == "cancelled":
            with pytest.raises(asyncio.CancelledError) as raised:
                await plugin.execute_sql_query("source", "SELECT 1")
            assert raised.value is failure
        else:
            result = await plugin.execute_sql_query("source", "SELECT 1")
            assert result.model_dump() == (
                {"results": [{"value": 1}], "error": None}
                if stage == "success" else {"results": None, "error": "RuntimeError"}
            )
    plugin.cosmos.get_document.assert_awaited_once_with(plugin.container_name, "source")
    if stage not in {"configuration", "connect", "cancelled"}:
        cursor.execute.assert_called_once_with("SELECT 1")
    else:
        cursor.execute.assert_not_called()
    assert marker not in caplog.text


@pytest.mark.parametrize("outcome", ["configuration", "request", "cancelled", "success"])
async def test_dax_tool_retains_user_token_and_explicit_error_result(plugin, outcome, caplog):
    marker = "synthetic-private-dax-error"
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    plugin.cosmos.get_document.return_value = {
        "id": "source", "description": "test", "type": "semantic_model",
        "organization": "test", "dataset": "dataset", "workspace": "workspace",
        "tenant_id": "tenant", "client_id": "client",
    }
    client = SimpleNamespace(execute_restapi_dax_query=AsyncMock(return_value=[{"value": 1}]))
    if outcome == "configuration":
        plugin.cosmos.get_document.side_effect = failure
    elif outcome in {"request", "cancelled"}:
        client.execute_restapi_dax_query.side_effect = failure
    with patch("plugins.nl2sql.plugin.SemanticModelClient", return_value=client):
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError) as raised:
                await plugin.execute_dax_query("source", "EVALUATE table", "synthetic-user-token")
            assert raised.value is failure
        else:
            result = await plugin.execute_dax_query("source", "EVALUATE table", "synthetic-user-token")
            assert result.model_dump() == (
                {"results": [{"value": 1}], "error": None}
                if outcome == "success" else {"results": None, "error": "RuntimeError"}
            )
    if outcome == "configuration":
        client.execute_restapi_dax_query.assert_not_awaited()
    else:
        client.execute_restapi_dax_query.assert_awaited_once_with(
            dax_query="EVALUATE table", user_token="synthetic-user-token")
    assert marker not in caplog.text
    assert "synthetic-user-token" not in caplog.text
