import importlib
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response
from starlette.requests import Request


class Config:
    auth_failed = False
    disabled = True

    def get(self, key, default=None, **_kwargs):
        values = {
            "ALLOW_ANONYMOUS": True,
            "LOG_LEVEL": "WARNING",
            "AZURE_LOG_LEVEL": "WARNING",
        }
        return values.get(key, default)

    def get_value(self, _key, default=None, **_kwargs):
        return default


@pytest.mark.asyncio
async def test_endpoint_replaces_inbound_correlation_id_and_returns_server_id():
    sys.modules.pop("main", None)
    with patch("dependencies.get_config", return_value=Config()):
        main = importlib.import_module("main")

    async def stream_response(_ask, _question_id):
        yield "answer"

    orchestrator = SimpleNamespace(
        stream_response=stream_response,
        save_feedback=AsyncMock(),
    )
    body = SimpleNamespace(
        type="ask",
        ask="question",
        question=None,
        conversation_id=None,
        question_id="question-id",
        user_context={},
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/orchestrator",
            "headers": [(b"x-correlation-id", b"req_attacker")],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )
    response = Response()

    with patch.object(
        main.Orchestrator,
        "create",
        new=AsyncMock(return_value=orchestrator),
    ) as create:
        streaming = await main.orchestrator_endpoint(
            request=request,
            response=response,
            body=body,
            x_api_key=None,
            dapr_api_token=None,
            authorization=None,
        )
        chunks = [chunk async for chunk in streaming.body_iterator]

    correlation_id = streaming.headers["X-Correlation-ID"]
    assert re.fullmatch(r"req_[0-9a-f]{32}", correlation_id)
    assert correlation_id != "req_attacker"
    assert response.headers["X-Correlation-ID"] == correlation_id
    assert create.await_args.kwargs["correlation_id"] == correlation_id
    assert chunks == ["answer"]
