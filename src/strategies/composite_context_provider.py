"""Composite Context Provider for Microsoft Agent Framework.

Wraps multiple :class:`ContextProvider` instances into a single provider
so they can be passed to ``ChatAgent(context_provider=...)`` which only
accepts one provider.
"""

import asyncio
import logging
from collections.abc import MutableSequence, Sequence
from typing import Any

from agent_framework import ChatMessage, Context, ContextProvider

logger = logging.getLogger(__name__)


class CompositeContextProvider(ContextProvider):
    """Delegates to multiple child providers and merges their contexts."""

    def __init__(self, providers: Sequence[ContextProvider]) -> None:
        self._providers = list(providers)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self):
        for p in self._providers:
            await p.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for p in self._providers:
            await p.__aexit__(exc_type, exc_val, exc_tb)

    async def thread_created(self, thread_id: str | None) -> None:
        for p in self._providers:
            await p.thread_created(thread_id)

    # -- core hook ---------------------------------------------------------

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        all_messages: list[ChatMessage] = []
        all_tools: list = []
        all_instructions: list[str] = []

        # Run all providers in parallel for lower latency
        contexts = await asyncio.gather(
            *(p.invoking(messages, **kwargs) for p in self._providers),
            return_exceptions=True,
        )

        for i, ctx in enumerate(contexts):
            if isinstance(ctx, Exception):
                logger.warning("[CompositeContextProvider] Provider %d failed: %s", i, ctx)
                continue
            if ctx:
                if ctx.messages:
                    all_messages.extend(ctx.messages)
                if ctx.tools:
                    all_tools.extend(ctx.tools)
                if ctx.instructions:
                    all_instructions.append(ctx.instructions)

        return Context(
            messages=all_messages or None,
            tools=all_tools or None,
            instructions="\n".join(all_instructions) if all_instructions else None,
        )

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        # Run all providers in parallel (mirrors invoking behaviour)
        results = await asyncio.gather(
            *(p.invoked(request_messages, response_messages, invoke_exception, **kwargs)
              for p in self._providers),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("[CompositeContextProvider] Provider %d invoked() failed: %s", i, result)
