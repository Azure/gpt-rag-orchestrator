"""Optional adapter absence must not mask provider initialization defects."""

import builtins
import runpy
from unittest.mock import AsyncMock, MagicMock

import pytest

from strategies import agent_provider_v2


def _load_provider(monkeypatch, error=None):
    original_import = builtins.__import__

    def import_adapter(name, *args, **kwargs):
        if name == "agent_framework_azure_ai._shared" and error is not None:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_adapter)
    return runpy.run_path(agent_provider_v2.__file__)


@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
async def test_absent_optional_converter_keeps_explicit_creation_failure(monkeypatch, error_type):
    namespace = _load_provider(monkeypatch, error_type("synthetic adapter absence"))
    provider = MagicMock(create_agent=AsyncMock())
    client = MagicMock()
    client.agents.create_version = AsyncMock()
    create = namespace["_create_versioned_agent"]
    arguments = dict(provider=provider, client=client, name="test-agent", model="test-model",
                     instructions="test instructions", reasoning_effort="low")
    with pytest.raises(RuntimeError, match="unavailable in this build but tools were requested"):
        await create(**arguments, tools=[object()])
    client.agents.create_version.assert_not_awaited()
    provider.create_agent.assert_not_awaited()
    await create(**arguments, tools=None)
    client.agents.create_version.assert_awaited_once()


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_unexpected_converter_import_failure_propagates(monkeypatch, error_type):
    error = error_type("synthetic adapter initialization failure")
    with pytest.raises(error_type) as raised:
        _load_provider(monkeypatch, error)
    assert raised.value is error


def test_available_converter_is_retained(monkeypatch):
    namespace = _load_provider(monkeypatch)
    assert namespace["_HAS_TOOL_CONVERTER"] is True
    assert namespace["to_azure_ai_tools"] is agent_provider_v2.to_azure_ai_tools
