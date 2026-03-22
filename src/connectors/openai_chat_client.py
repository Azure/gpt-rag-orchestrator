"""
OpenAI Chat Client Adapter for Microsoft Agent Framework.

Implements ChatClientProtocol using AsyncAzureOpenAI, enabling ChatAgent
to call Azure OpenAI / Foundry model deployments directly — without going
through Azure AI Foundry Agent Service V2.
"""

import logging
from collections.abc import AsyncIterable, Sequence
from typing import Any

from azure.identity import get_bearer_token_provider
from openai import AsyncAzureOpenAI

from agent_framework import (
    ChatClientProtocol,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
)

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    """Chat client that wraps AsyncAzureOpenAI and satisfies ChatClientProtocol.

    This allows ``ChatAgent`` to use an Azure OpenAI model deployment directly,
    with no dependency on Azure AI Foundry Agent Service V2.
    """

    additional_properties: dict[str, Any]

    def __init__(
        self,
        *,
        azure_endpoint: str,
        model_deployment_name: str,
        credential: Any,
        api_version: str = "2025-04-01-preview",
        max_retries: int = 10,
    ) -> None:
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        self._client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            max_retries=max_retries,
        )
        self._model = model_deployment_name
        self.additional_properties = {}

    # ------------------------------------------------------------------
    # ChatClientProtocol.get_response
    # ------------------------------------------------------------------
    async def get_response(
        self,
        messages: str | ChatMessage | Sequence[str | ChatMessage],
        *,
        options: Any | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        oai_messages = self._to_openai_messages(messages)
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            stream=False,
        )
        choice = completion.choices[0]
        return ChatResponse(
            text=choice.message.content or "",
            model_id=completion.model,
            response_id=completion.id,
        )

    # ------------------------------------------------------------------
    # ChatClientProtocol.get_streaming_response
    # ------------------------------------------------------------------
    def get_streaming_response(
        self,
        messages: str | ChatMessage | Sequence[str | ChatMessage],
        *,
        options: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        oai_messages = self._to_openai_messages(messages)
        return self._stream(oai_messages)

    async def _stream(self, oai_messages: list[dict]) -> AsyncIterable[ChatResponseUpdate]:
        t0 = __import__("time").time()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            stream=True,
            max_completion_tokens=1024,
        )
        logger.info("[OpenAIChatClient] completions.create returned in %.2fs", __import__("time").time() - t0)
        first_token = True
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content if delta and delta.content else None
            if text:
                if first_token:
                    logger.info("[OpenAIChatClient] first_token: %.2fs", __import__("time").time() - t0)
                    first_token = False
                yield ChatResponseUpdate(
                    text=text,
                    role="assistant",
                    response_id=chunk.id,
                    model_id=chunk.model,
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_openai_messages(
        messages: str | ChatMessage | Sequence[str | ChatMessage],
    ) -> list[dict]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, ChatMessage):
            return [{"role": messages.role.value, "content": messages.text or ""}]
        result: list[dict] = []
        for m in messages:
            if isinstance(m, str):
                result.append({"role": "user", "content": m})
            else:
                result.append({"role": m.role.value, "content": m.text or ""})
        return result

    # ------------------------------------------------------------------
    # async context manager (optional, for parity with AzureAIAgentClient)
    # ------------------------------------------------------------------
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.close()
