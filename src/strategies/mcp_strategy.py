import asyncio
import logging
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from agent_framework import ChatAgent, Role, TextContent, UsageContent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import get_bearer_token_provider
from opentelemetry.trace import SpanKind

from connectors.mcp_client import (
    normalize_mcp_transport,
    open_mcp_tool,
    resolve_mcp_endpoint,
)
from telemetry import Telemetry, wrap_ai_functions
from util.tools import is_azure_environment

from .agent_strategies import AgentStrategies
from .base_agent_strategy import BaseAgentStrategy

tracer = Telemetry.get_tracer(__name__)


class McpStrategy(BaseAgentStrategy):
    """Run a request-scoped local MAF agent with tools from an MCP server."""

    def __init__(self) -> None:
        super().__init__()
        self.strategy_type = AgentStrategies.MCP
        self.existing_agent_id = self.cfg.get("AGENT_ID", "") or None
        self.mcp_server_timeout = self.cfg.get(
            "MCP_CLIENT_TIMEOUT",
            default=600,
            type=int,
        )
        self.mcp_server_api_key = self.cfg.get("MCP_APP_APIKEY", default=None)
        self.mcp_server_transport = normalize_mcp_transport(
            self.cfg.get("MCP_SERVER_TRANSPORT", default="sse")
        )

        endpoint = self.cfg.get(
            "MCP_APP_ENDPOINT",
            default="http://localhost:80",
        )
        if not is_azure_environment():
            endpoint = "http://localhost:5000"
        self.mcp_server_url = resolve_mcp_endpoint(
            endpoint,
            self.mcp_server_transport,
        )

        self.model: dict[str, Any] | None = None
        self._token_provider = None

    @classmethod
    async def create(cls) -> "McpStrategy":
        instance = cls()
        instance.model = instance._get_model()
        if not instance.model:
            raise ValueError(
                "MODEL_DEPLOYMENTS must contain the configured chat deployment."
            )
        instance._token_provider = get_bearer_token_provider(
            instance.cfg.credential,
            "https://ai.azure.com/.default",
        )
        return instance

    def _create_chat_client(self) -> AzureOpenAIChatClient:
        if self.model is None or self._token_provider is None:
            raise RuntimeError("MCP strategy has not been initialized.")
        return AzureOpenAIChatClient(
            deployment_name=self.model["name"],
            endpoint=self.model["endpoint"],
            api_version=self.model["version"],
            ad_token_provider=self._token_provider,
        )

    @staticmethod
    def _assistant_text(update: Any) -> str:
        role = getattr(update.role, "value", update.role)
        if role != Role.ASSISTANT.value:
            return ""
        return "".join(
            content.text
            for content in update.contents
            if isinstance(content, TextContent)
        )

    @staticmethod
    def _usage(update: Any) -> tuple[int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        for content in update.contents:
            if not isinstance(content, UsageContent):
                continue
            prompt_tokens += content.details.input_token_count or 0
            completion_tokens += content.details.output_token_count or 0
        return prompt_tokens, completion_tokens

    async def initiate_agent_flow(self, user_message: str):
        """Stream assistant text and persist exactly the emitted response."""

        conv = self.conversation
        full_response_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        request_status = "error"
        connection_started = time.monotonic()
        stream_started: float | None = None
        connection_latency_ms = 0.0
        stream_latency_ms = 0.0
        tool_count = 0
        mcp_host = urlsplit(self.mcp_server_url).hostname or ""
        chat_client: AzureOpenAIChatClient | None = None

        with tracer.start_as_current_span(
            "initiate_agent_flow",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("agent.strategy", self.strategy_type.value)
            span.set_attribute("agent.id", self.existing_agent_id or "generated")
            span.set_attribute("mcp.transport", self.mcp_server_transport)
            span.set_attribute("mcp.host", mcp_host)

            try:
                async with open_mcp_tool(
                    endpoint=self.mcp_server_url,
                    transport=self.mcp_server_transport,
                    timeout=self.mcp_server_timeout,
                    user_context=self.user_context,
                    api_key=self.mcp_server_api_key,
                ) as mcp_tool:
                    connection_latency_ms = (
                        time.monotonic() - connection_started
                    ) * 1000
                    audited_functions = wrap_ai_functions(
                        mcp_tool.functions,
                        tool_kind="mcp",
                    )
                    tool_count = len(audited_functions)
                    span.set_attribute("mcp.tool_count", tool_count)

                    chat_client = self._create_chat_client()
                    async with ChatAgent(
                        chat_client=chat_client,
                        id=self.existing_agent_id,
                        name="MultiPluginAgent",
                        tools=audited_functions,
                    ) as agent:
                        conv["agent_id"] = agent.id
                        thread = agent.get_new_thread()
                        stream_started = time.monotonic()

                        async for update in agent.run_stream(
                            user_message,
                            thread=thread,
                        ):
                            update_prompt_tokens, update_completion_tokens = (
                                self._usage(update)
                            )
                            prompt_tokens += update_prompt_tokens
                            completion_tokens += update_completion_tokens

                            text = self._assistant_text(update)
                            if not text:
                                continue
                            full_response_parts.append(text)
                            yield text

                        stream_latency_ms = (
                            time.monotonic() - stream_started
                        ) * 1000

                full_response = "".join(full_response_parts)
                conv["messages"] = [
                    {
                        "role": "system",
                        "text": full_response,
                    }
                ]
                conv["completion_tokens"] = completion_tokens
                conv["prompt_tokens"] = prompt_tokens
                if self.user_context:
                    conv["user_context"] = self.user_context
                request_status = "success"
            except asyncio.CancelledError:
                request_status = "cancelled"
                raise
            finally:
                primary_exception = sys.exception()
                cleanup_error: BaseException | None = None
                if chat_client is not None:
                    try:
                        await chat_client.client.close()
                    except BaseException:
                        if primary_exception is not None:
                            logging.exception(
                                "[McpStrategy] Failed to close chat client while "
                                "preserving the primary exception."
                            )
                        else:
                            request_status = "error"
                            cleanup_error = sys.exception()

                if stream_started is not None and stream_latency_ms == 0.0:
                    stream_latency_ms = (
                        time.monotonic() - stream_started
                    ) * 1000
                span.set_attribute(
                    "mcp.connection_latency_ms",
                    connection_latency_ms,
                )
                span.set_attribute("agent.stream_latency_ms", stream_latency_ms)
                span.set_attribute("agent.request_status", request_status)
                logging.info(
                    "[McpStrategy] Request finished transport=%s host=%s "
                    "agent_id=%s status=%s tools=%s connection_ms=%.2f "
                    "stream_ms=%.2f",
                    self.mcp_server_transport,
                    mcp_host,
                    conv.get("agent_id", self.existing_agent_id or "generated"),
                    request_status,
                    tool_count,
                    connection_latency_ms,
                    stream_latency_ms,
                )
                if cleanup_error is not None:
                    raise cleanup_error
