import inspect
from unittest.mock import AsyncMock, patch

import pytest


class FakeNL2SQLPlugin:
    def __init__(self):
        self.get_all_datasources_info = AsyncMock(
            return_value={
                "datasources": [
                    {"name": "sales", "description": "Sales orders", "type": "sql_database"}
                ]
            }
        )
        self.tables_retrieval = AsyncMock(
            return_value={"tables": [{"table": "Orders", "description": "Sales orders"}]}
        )
        self.get_all_tables_info = AsyncMock(
            return_value={"tables": [{"table": "Orders", "description": "Sales orders"}]}
        )
        self.get_schema_info = AsyncMock(
            return_value={
                "table": "Orders",
                "columns": {"OrderId": "Order id", "Amount": "Order amount"},
            }
        )
        self.queries_retrieval = AsyncMock(return_value={"queries": []})
        self.validate_sql_query = AsyncMock(return_value={"is_valid": True})
        self.execute_sql_query = AsyncMock(return_value={"results": [{"total": 2}]})


def test_nl2sql_no_longer_imports_semantic_kernel():
    from plugins.nl2sql import plugin
    from strategies import nl2sql_strategy

    assert "semantic_kernel" not in inspect.getsource(nl2sql_strategy)
    assert "semantic_kernel" not in inspect.getsource(plugin)


@pytest.mark.asyncio
async def test_nl2sql_uses_local_tools_without_agent_service_creation(patch_dependencies):
    fake_plugin = FakeNL2SQLPlugin()

    with patch("strategies.nl2sql_strategy.NL2SQLPlugin", return_value=fake_plugin):
        from strategies.nl2sql_strategy import NL2SQLStrategy

        strategy = NL2SQLStrategy()
        strategy.conversation = {"messages": []}

    async def fake_run_agent(instructions, message, *, max_tokens=1200):
        if "choose exactly one datasource" in instructions:
            return '{"datasource_name":"sales","datasource_type":"sql_database"}'
        return '{"sql_query":"SELECT COUNT(*) AS total FROM Orders","reasoning":"Count orders."}'

    async def fake_stream_agent(instructions, message, *, max_tokens=None):
        yield "There are 2 orders."

    strategy._run_agent = fake_run_agent
    strategy._stream_agent = fake_stream_agent

    chunks = [chunk async for chunk in strategy.initiate_agent_flow("How many orders are there?")]

    assert "".join(chunks) == "There are 2 orders."
    fake_plugin.execute_sql_query.assert_awaited_once_with(
        "sales",
        "SELECT COUNT(*) AS total FROM Orders",
    )
    assert len(strategy.conversation["messages"]) == 2
