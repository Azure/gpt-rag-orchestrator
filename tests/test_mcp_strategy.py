import asyncio
import ast
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import h11
import pytest
from agent_framework import AgentResponseUpdate, TextContent, UsageContent, UsageDetails


async def _create_strategy(mock_config, *, overrides=None):
    values = {
        "AGENT_ID": "configured-agent",
        "MCP_APP_ENDPOINT": "https://mcp.example.test",
        "MCP_CLIENT_TIMEOUT": 90,
        "MCP_APP_APIKEY": "api-key",
        "MCP_SERVER_TRANSPORT": "sse",
    }
    values.update(overrides or {})
    original_get = mock_config.get.side_effect

    def get_value(key, default=None, type=str):  # noqa: A002
        if key in values:
            return values[key]
        return original_get(key, default, type)

    mock_config.get.side_effect = get_value

    from strategies.mcp_strategy import McpStrategy

    with (
        patch(
            "strategies.mcp_strategy.is_azure_environment",
            return_value=True,
        ),
        patch.object(
            McpStrategy,
            "_get_model",
            return_value={
                "name": "gpt-4o",
                "endpoint": "https://models.example.test",
                "version": "2025-04-01-preview",
            },
        ),
        patch(
            "strategies.mcp_strategy.get_bearer_token_provider",
            return_value=MagicMock(name="token_provider"),
        ),
    ):
        return await McpStrategy.create()


class _FakeAgent:
    def __init__(self, updates=None, *, error=None, block=False):
        self.id = "configured-agent"
        self.updates = updates or []
        self.error = error
        self.block = block
        self.entered = False
        self.closed = False
        self.thread = object()
        self.blocking = asyncio.Event()

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def get_new_thread(self):
        return self.thread

    async def run_stream(self, messages, *, thread):
        assert thread is self.thread
        if self.error:
            raise self.error
        for update in self.updates:
            yield update
        if self.block:
            self.blocking.set()
            await asyncio.Event().wait()


def _fake_mcp_context(state):
    @asynccontextmanager
    async def open_tool(**kwargs):
        state["open_kwargs"] = kwargs
        state["entered"] = True
        try:
            yield MagicMock(functions=[object(), object()])
        finally:
            state["closed"] = True

    return open_tool


@pytest.mark.asyncio
async def test_mcp_strategy_streams_assistant_text_and_sums_usage(
    patch_dependencies,
    mock_config,
):
    strategy = await _create_strategy(mock_config)
    strategy.conversation = {"id": "conversation"}
    strategy.user_context = {"principal_id": "user-one"}
    state = {}
    fake_agent = _FakeAgent(
        [
            AgentResponseUpdate(role="assistant", text="Hello "),
            AgentResponseUpdate(role="tool", text="private tool result"),
            AgentResponseUpdate(
                role="assistant",
                contents=[
                    UsageContent(
                        UsageDetails(
                            input_token_count=10,
                            output_token_count=4,
                        )
                    )
                ],
            ),
            AgentResponseUpdate(role=None, text="not assistant"),
            AgentResponseUpdate(role="assistant", text="world"),
            AgentResponseUpdate(
                role="assistant",
                contents=[
                    UsageContent(
                        UsageDetails(
                            input_token_count=7,
                            output_token_count=3,
                        )
                    )
                ],
            ),
        ]
    )

    with (
        patch(
            "strategies.mcp_strategy.open_mcp_tool",
            _fake_mcp_context(state),
        ),
        patch(
            "strategies.mcp_strategy.ChatAgent",
            return_value=fake_agent,
        ) as chat_agent,
        patch.object(
            strategy,
            "_create_chat_client",
            return_value=MagicMock(),
        ),
    ):
        chunks = [
            chunk
            async for chunk in strategy.initiate_agent_flow("private prompt")
        ]

    assert chunks == ["Hello ", "world"]
    assert strategy.conversation == {
        "id": "conversation",
        "agent_id": "configured-agent",
        "messages": [{"role": "system", "text": "Hello world"}],
        "completion_tokens": 7,
        "prompt_tokens": 17,
        "user_context": {"principal_id": "user-one"},
    }
    assert state["closed"] is True
    assert fake_agent.closed is True
    assert state["open_kwargs"]["user_context"] == {
        "principal_id": "user-one"
    }
    chat_agent.assert_called_once()
    assert chat_agent.call_args.kwargs["id"] == "configured-agent"
    assert chat_agent.call_args.kwargs["name"] == "MultiPluginAgent"


@pytest.mark.asyncio
async def test_mcp_strategy_cleans_up_after_stream_error(
    patch_dependencies,
    mock_config,
):
    strategy = await _create_strategy(mock_config)
    strategy.conversation = {}
    state = {}
    fake_agent = _FakeAgent(error=RuntimeError("model failed"))

    with (
        patch(
            "strategies.mcp_strategy.open_mcp_tool",
            _fake_mcp_context(state),
        ),
        patch(
            "strategies.mcp_strategy.ChatAgent",
            return_value=fake_agent,
        ),
        patch.object(
            strategy,
            "_create_chat_client",
            return_value=MagicMock(),
        ),
    ):
        with pytest.raises(RuntimeError, match="model failed"):
            _ = [
                chunk
                async for chunk in strategy.initiate_agent_flow("prompt")
            ]

    assert state["closed"] is True
    assert fake_agent.closed is True
    assert "messages" not in strategy.conversation


@pytest.mark.asyncio
async def test_mcp_strategy_cleans_up_after_cancellation(
    patch_dependencies,
    mock_config,
):
    strategy = await _create_strategy(mock_config)
    strategy.conversation = {}
    state = {}
    fake_agent = _FakeAgent(
        [AgentResponseUpdate(role="assistant", text="started")],
        block=True,
    )

    async def consume():
        async for _ in strategy.initiate_agent_flow("prompt"):
            pass

    with (
        patch(
            "strategies.mcp_strategy.open_mcp_tool",
            _fake_mcp_context(state),
        ),
        patch(
            "strategies.mcp_strategy.ChatAgent",
            return_value=fake_agent,
        ),
        patch.object(
            strategy,
            "_create_chat_client",
            return_value=MagicMock(),
        ),
    ):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(fake_agent.blocking.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert state["closed"] is True
    assert fake_agent.closed is True
    assert "messages" not in strategy.conversation


@pytest.mark.asyncio
async def test_mcp_strategy_preserves_azure_openai_configuration(
    patch_dependencies,
    mock_config,
):
    strategy = await _create_strategy(mock_config)

    with patch(
        "strategies.mcp_strategy.AzureOpenAIChatClient",
    ) as chat_client:
        strategy._create_chat_client()

    chat_client.assert_called_once_with(
        deployment_name="gpt-4o",
        endpoint="https://models.example.test",
        api_version="2025-04-01-preview",
        ad_token_provider=strategy._token_provider,
    )


@pytest.mark.asyncio
async def test_mcp_strategy_rejects_unsupported_transport(
    patch_dependencies,
    mock_config,
):
    with pytest.raises(ValueError, match="Supported values"):
        await _create_strategy(
            mock_config,
            overrides={"MCP_SERVER_TRANSPORT": "websocket"},
        )


@pytest.mark.asyncio
async def test_mcp_strategy_does_not_mutate_h11(
    patch_dependencies,
    mock_config,
):
    original_write_headers = h11._writers.write_headers

    await _create_strategy(mock_config)

    assert h11._writers.write_headers is original_write_headers


def test_migrated_surfaces_do_not_import_semantic_kernel():
    root = Path(__file__).parents[1]
    migrated_files = [
        root / "src" / "strategies" / "mcp_strategy.py",
        root / "src" / "strategies" / "base_agent_strategy.py",
        root / "src" / "plugins" / "common" / "plugin.py",
        root / "src" / "plugins" / "retrieval" / "plugin.py",
        root / "requirements.txt",
        root / "pyproject.toml",
    ]

    for path in migrated_files:
        assert "semantic_kernel" not in path.read_text(encoding="utf-8")


def test_plugin_decorators_preserve_public_contracts():
    from plugins.common.plugin import CommonPlugin

    assert CommonPlugin.get_today_date.name == "GetTodayDate"
    assert CommonPlugin.get_time.name == "GetTime"

    retrieval_path = (
        Path(__file__).parents[1]
        / "src"
        / "plugins"
        / "retrieval"
        / "plugin.py"
    )
    tree = ast.parse(retrieval_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected_names = {
        "vector_index_retrieve": "VectorIndexRetrieve",
        "multimodal_vector_index_retrieve": "MultimodalVectorIndexRetrieve",
        "get_data_points_from_chat_log": "GetDataPointsFromChatLog",
    }

    for function_name, public_name in expected_names.items():
        decorator = functions[function_name].decorator_list[0]
        assert isinstance(decorator, ast.Call)
        assert isinstance(decorator.func, ast.Name)
        assert decorator.func.id == "ai_function"
        name_argument = next(
            keyword.value
            for keyword in decorator.keywords
            if keyword.arg == "name"
        )
        assert isinstance(name_argument, ast.Constant)
        assert name_argument.value == public_name

    security_default = functions["vector_index_retrieve"].args.defaults[0]
    assert isinstance(security_default, ast.Constant)
    assert security_default.value == "anonymous"


def test_stream_filter_only_returns_assistant_text():
    from strategies.mcp_strategy import McpStrategy

    assistant = AgentResponseUpdate(
        role="assistant",
        contents=[TextContent("visible")],
    )
    tool = AgentResponseUpdate(
        role="tool",
        contents=[TextContent("hidden")],
    )

    assert McpStrategy._assistant_text(assistant) == "visible"
    assert McpStrategy._assistant_text(tool) == ""
    assert "semantic_kernel" not in inspect.getsource(McpStrategy)
