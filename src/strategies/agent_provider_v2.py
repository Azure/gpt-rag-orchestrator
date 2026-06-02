"""
Shared helpers for the Foundry declarative Agent API (Agent Service v2).

This module centralises the *new* Azure AI Foundry agent pattern:

* a process-wide ``AIProjectClient`` + ``AzureAIProjectAgentProvider`` singleton,
* a get-or-create-by-name routine that materialises a *versioned prompt agent*
  (``AIProjectClient.agents.create_version`` + ``PromptAgentDefinition``) exactly
  once and reuses it across every request.

It replaces the legacy Assistants-style ``AgentsClient.create_agent()`` /
``delete_agent()`` ephemeral-per-request pattern used by the older strategies.

Strategies are instantiated per request by ``AgentStrategyFactory``, so any state
that must survive across requests (the provider, the project client and the
resolved agent definitions) lives at module scope here.
"""

import asyncio
import hashlib
import logging
from typing import Any, Optional, Sequence

from azure.core.exceptions import HttpResponseError, ResourceExistsError, ResourceNotFoundError
from azure.ai.projects.aio import AIProjectClient
from agent_framework.azure import AzureAIProjectAgentProvider

# Conversations whose server-side ``thread_id`` was created by the new
# Responses-based prompt agents are tagged with this value. A conversation
# carrying a different/absent tag holds a legacy Assistants thread id that is
# NOT a valid Responses conversation id and must be reset before reuse.
AGENT_BACKEND_TAG = "foundry_responses_v1"

_provider: Optional[AzureAIProjectAgentProvider] = None
_project_client: Optional[AIProjectClient] = None
_provider_lock = asyncio.Lock()

# name -> AgentVersionDetails, cached so per-request agent construction is a
# pure in-process call (no HTTP) once the agent has been resolved once.
_agent_details_cache: dict[str, Any] = {}
_agent_locks: dict[str, asyncio.Lock] = {}


async def get_provider(endpoint: str, credential: Any) -> AzureAIProjectAgentProvider:
    """Return the process-wide provider, creating it (and its ``AIProjectClient``)
    once. The client is intentionally never closed: it lives for the lifetime of
    the worker process and is shared by all requests."""
    global _provider, _project_client
    if _provider is not None:
        return _provider
    async with _provider_lock:
        if _provider is None:
            _project_client = AIProjectClient(endpoint=endpoint, credential=credential)
            _provider = AzureAIProjectAgentProvider(project_client=_project_client)
            logging.info("[AgentProviderV2] Initialized AIProjectClient + provider (endpoint=%s)", endpoint)
    return _provider


def compute_agent_name(
    base_name: str,
    *,
    model: str,
    instructions: str,
    tool_names: Sequence[str],
    extra: Optional[dict] = None,
) -> str:
    """Derive a stable, definition-aware agent name.

    A short fingerprint of the definition (model + instructions + tool names +
    extra flags) is embedded in the name so that:

    * identical definitions resolve to the *same* persistent agent — created
      once, reused on every subsequent request and across restarts;
    * a changed definition (e.g. a new release with updated instructions or
      tools) deterministically maps to a *new* agent name, avoiding stale-version
      reuse during rolling deployments where multiple app versions coexist.
    """
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8"))
    h.update(b"\0")
    h.update((instructions or "").encode("utf-8"))
    h.update(b"\0")
    h.update(",".join(sorted(tool_names)).encode("utf-8"))
    if extra:
        for key in sorted(extra):
            h.update(f"\0{key}={extra[key]}".encode("utf-8"))
    fingerprint = h.hexdigest()[:10]
    return f"{base_name}-{fingerprint}"


async def get_or_create_agent_details(
    *,
    provider: AzureAIProjectAgentProvider,
    name: str,
    model: str,
    instructions: str,
    tools: Optional[Sequence[Any]] = None,
) -> Any:
    """Return cached ``AgentVersionDetails`` for ``name``, creating the versioned
    prompt agent exactly once when it does not yet exist.

    Existence is checked with ``AIProjectClient.agents.get(name)``; creation goes
    through ``AzureAIProjectAgentProvider.create_agent`` which calls
    ``agents.create_version`` with a ``PromptAgentDefinition`` under the hood.

    There is no ephemeral per-request creation and no ``delete_agent``: the
    resolved definition is cached in-process and reused.

    Note: ``create_version`` is not idempotent, so creation is guarded by an
    existence check and a per-name lock. Concurrent cold replicas may still each
    create an (identical) version; that is acceptable because every process
    always reuses ``versions.latest`` for the same definition-fingerprinted name.
    """
    cached = _agent_details_cache.get(name)
    if cached is not None:
        return cached

    lock = _agent_locks.setdefault(name, asyncio.Lock())
    async with lock:
        cached = _agent_details_cache.get(name)
        if cached is not None:
            return cached

        client = _project_client
        try:
            existing = await client.agents.get(agent_name=name)
            details = existing.versions.latest
            logging.info(
                "[AgentProviderV2] Reusing existing prompt agent '%s' (version=%s)",
                name, getattr(details, "version", "?"),
            )
        except ResourceNotFoundError:
            logging.info(
                "[AgentProviderV2] Prompt agent '%s' not found; creating once via create_version",
                name,
            )
            try:
                await provider.create_agent(
                    name=name,
                    model=model,
                    instructions=instructions,
                    tools=tools,
                )
            except (ResourceExistsError, HttpResponseError) as create_err:
                # Another cold replica created the same definition-fingerprinted
                # agent first (create_version is not idempotent). Fall back to
                # reusing whatever version now exists for this name.
                status = getattr(create_err, "status_code", None)
                if isinstance(create_err, HttpResponseError) and status not in (409, None):
                    raise
                logging.info(
                    "[AgentProviderV2] Prompt agent '%s' was created concurrently "
                    "(status=%s); reusing existing version",
                    name, status,
                )
            existing = await client.agents.get(agent_name=name)
            details = existing.versions.latest
            logging.info(
                "[AgentProviderV2] Created prompt agent '%s' (version=%s)",
                name, getattr(details, "version", "?"),
            )

        _agent_details_cache[name] = details
        return details


def reset_legacy_thread(conv: dict) -> None:
    """Drop a server ``thread_id`` created by a different agent backend.

    Legacy Assistants-API thread ids are not valid Responses conversation ids, so
    they are cleared whenever the stored backend tag does not match the current
    one. Fresh conversations are simply tagged with the current backend.
    """
    if conv.get("agent_backend") != AGENT_BACKEND_TAG:
        if conv.get("thread_id"):
            logging.info(
                "[AgentProviderV2] Resetting incompatible thread_id (stored backend=%s)",
                conv.get("agent_backend"),
            )
        conv["thread_id"] = None
        conv["agent_backend"] = AGENT_BACKEND_TAG
