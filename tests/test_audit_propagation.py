from unittest.mock import patch

import pytest
import httpx
from opentelemetry import baggage, context
from opentelemetry.sdk.trace import TracerProvider

from connectors.mcp_client import (
    _TraceContextOnlyAsyncTransport,
    open_mcp_tool,
)


class FakeSSETool:
    headers = None
    httpx_client_factory = None

    def __init__(self, **kwargs):
        type(self).headers = kwargs["headers"]
        type(self).httpx_client_factory = kwargs["httpx_client_factory"]
        self.functions = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_sse_injects_only_w3c_trace_context_without_baggage():
    provider = TracerProvider()
    baggage_context = baggage.set_baggage("identity", "must-not-propagate")
    baggage_token = context.attach(baggage_context)
    try:
        with provider.get_tracer("test").start_as_current_span("request"):
            with patch(
                "connectors.mcp_client.LegacySSEMCPTool",
                FakeSSETool,
            ):
                async with open_mcp_tool(
                    endpoint="https://mcp.example.test",
                    transport="sse",
                    timeout=10,
                    user_context={},
                    api_key=None,
                ):
                    pass
    finally:
        context.detach(baggage_token)

    assert FakeSSETool.headers["traceparent"].startswith("00-")
    assert "baggage" not in FakeSSETool.headers
    client = FakeSSETool.httpx_client_factory()
    try:
        assert isinstance(client._transport, _TraceContextOnlyAsyncTransport)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_streamable_http_transport_strips_baggage_and_injects_trace():
    class RecordingTransport(httpx.AsyncBaseTransport):
        request = None

        async def handle_async_request(self, request):
            type(self).request = request
            return httpx.Response(200, request=request)

        async def aclose(self):
            return None

    transport = _TraceContextOnlyAsyncTransport(RecordingTransport())
    baggage_context = baggage.set_baggage("identity", "must-not-propagate")
    baggage_token = context.attach(baggage_context)
    try:
        with TracerProvider().get_tracer("test").start_as_current_span(
            "request"
        ):
            await transport.handle_async_request(
                httpx.Request(
                    "POST",
                    "https://mcp.example.test/mcp",
                    headers={"baggage": "identity=private-user"},
                )
            )
    finally:
        context.detach(baggage_token)
        await transport.aclose()

    assert RecordingTransport.request.headers["traceparent"].startswith("00-")
    assert "baggage" not in RecordingTransport.request.headers
