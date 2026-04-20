# Changelog

All notable changes to this project will be documented in this file.  
This format follows [Keep a Changelog](https://keepachangelog.com/) and adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [v2.6.3] - 2026-04-20

### Fixed
- **Missing `regex` dependency in `requirements.txt`:** Added `regex>=2022.1.18` as an explicit dependency. The `tiktoken==0.8.0` package requires `regex` at runtime, but it was not listed in `requirements.txt`. This caused pip dependency resolver warnings when installing additional packages (such as `agentops-toolkit`) on top of the project dependencies, since `tiktoken` would report an unsatisfied requirement. Users installing into a fresh virtual environment could also encounter import errors from `tiktoken`.

## [v2.6.2] - 2026-04-18
### Fixed
- **OpenTelemetry version pinning:** Pinned `azure-monitor-opentelemetry==1.8.7`, `azure-monitor-opentelemetry-exporter==1.0.0b49`, `opentelemetry-instrumentation-httpx==0.61b0`, and `opentelemetry-instrumentation-fastapi==0.61b0` in `requirements.txt`. Unpinned versions caused non-deterministic Docker builds where an older exporter (referencing the removed `LogData` class) could be paired with `opentelemetry-sdk>=1.39.0`, crashing the container on startup with `ImportError: cannot import name 'LogData' from 'opentelemetry.sdk._logs'`. ([#445](https://github.com/Azure/GPT-RAG/issues/445))
- **Permission trimming header format:** Removed erroneous `Bearer ` prefix from the `x-ms-query-source-authorization` header value in both the REST API path (`search.py`) and the SDK path (`search_context_provider.py`). Azure AI Search expects the raw OBO token without the prefix; including it caused `400 Invalid header` errors when `permissionFilterOption` was enabled on the search index. ([#447](https://github.com/Azure/GPT-RAG/issues/447))

## [v2.6.1] - 2026-04-01
### Fixed
- **Multimodal image rendering:** Fixed images not appearing in multimodal responses caused by `reasoning_effort` parameter being rejected by gpt-5-nano in the image classifier, validator, and intent classification calls. Removed the unsupported parameter from all three LLM classification calls.
- **Image validation guardrail:** Fixed the post-response image validator returning verbose 200-token descriptions instead of "VALID"/"INVALID" classifications. Strengthened the validation prompt, reframed the user message to prevent Q&A-style responses, and reduced `max_completion_tokens` from 200 to 10.
- **Inline image placement:** Fixed images being grouped at the bottom of responses in a "Visual Guides" section instead of appearing inline. Added a few-shot example to the system prompt demonstrating correct inline placement with introductory sentences, and reinforced inline rules in both the main prompt and the search context provider prompt.
- **Search retry without OBO:** Added fallback logic in `MultimodalSearchContextProvider` to retry search without the `x_ms_query_source_authorization` header when the OBO token exchange fails, preventing total search failure when admin consent is not granted.
- **Assistant history few-shot poisoning:** Stripped `![](path)` markdown from old assistant messages in conversation history to prevent them from acting as few-shot examples that teach the model to omit image embedding.

## [v2.6.0] - 2026-03-31
### Added
- **Conversation History REST API:** New CRUD endpoints (`GET /conversations`, `GET /conversations/{id}`, `PATCH /conversations/{id}`, `DELETE /conversations/{id}`) enabling front-end conversation list, detail view, rename, and soft-delete operations.
- **Pydantic response schemas:** Added `ConversationMetadata`, `ConversationListResponse`, and `ConversationDetail` models for typed API responses.
- **CosmosDB conversation query helpers:** Added `query_user_conversations`, `read_user_conversation`, `update_conversation_name`, and `soft_delete_conversation` module-level functions with parameterized queries.
- **Conversation history unit tests** (`test_conversation_history.py`).

### Changed
- **Multi-tenant partition key support:** `CosmosDBClient.get_document` and `create_document` now accept an optional `partition_key` parameter, enabling per-user partition isolation (`principal_id` for authenticated users, `anonymous-{conversation_id}` for unauthenticated users).
- **Message persistence in strategies:** `SingleAgentRAGStrategyV2` now captures the full streamed response and appends `{"role": "user", "text": ...}` / `{"role": "assistant", "text": ...}` messages to the conversation document for both direct-LLM and Agent SDK paths.
- **Orchestrator user identity plumbing:** `Orchestrator.__init__` now receives `principal_id` from `user_context`, populates partition key, and stores `name`, `principal_id`, and `lastUpdated` on new conversation documents.
- **Automatic `lastUpdated` timestamps:** `CosmosDBClient.create_document` and `update_document` now set `lastUpdated` automatically.
- **Conversation not-found handling:** When a conversation ID is provided but the document does not exist, the orchestrator now creates a new conversation instead of raising an error.

### Improved
- **Citation prompt rules:** Strengthened citation instructions in `single_agent_rag/main.jinja2` and added `_CITATION_RULES` constant in `SingleAgentRAGStrategyV2` to enforce `[title](link)` format, prevent link omission, and limit citation repetition.

### Fixed
- Normalized line endings from CRLF to LF across 32 files for cross-platform consistency.

## [v2.5.0] - 2026-03-24
### Added
- **MafLiteStrategy (`maf_lite`):** New orchestration strategy using the Microsoft Agent Framework with direct Azure OpenAI model access. Provides the same MAF features (ChatAgent, UserProfileMemory, AzureAISearchContextProvider, retrieval plugins) without requiring Azure AI Foundry Agent Service v2, offering a lighter-weight deployment option.
- **OpenAIChatClient:** Custom `ChatClientProtocol` adapter wrapping `AsyncAzureOpenAI` for direct model access, used by the `maf_lite` strategy.
- **Citation utilities module (`src/util/citations.py`):** Extracted citation processing functions into a shared utility module.
- **Unit test suite:** Added comprehensive unit tests for strategies, factory, citations, and OpenAI chat client (46 tests).

### Changed
- **Renamed `MafStrategy` to `MafAgentServiceStrategy` (`maf_agent_service`):** Clarifies that this strategy uses the Microsoft Agent Framework **with** Azure AI Foundry Agent Service v2. The App Configuration key changed from `maf` to `maf_agent_service`.
- **Strategy lineup table in README:** Added a quick-reference table listing all available strategies with their configuration keys.
- **Default strategy changed to `maf_lite`:** The fallback strategy when `AGENT_STRATEGY` is not set in App Configuration is now `maf_lite` instead of `single_agent_rag`.

### Fixed
- **MAF Agent Service lifecycle cleanup:** Updated `maf_agent_service` to use request-scoped Agent Service client ownership, preventing leaked async HTTP sessions that produced `Unclosed client session`, `Unclosed connector`, and `SSL shutdown timed out` errors in runtime logs.
- **User profile extraction warnings in Agent Service path:** Guarded unsupported extraction path in `UserProfileMemory` for `AzureAIAgentClient`, preventing noisy `'dict' object has no attribute 'conversation_id'` warnings in normal operation.
- **Conversation persist teardown path:** Updated orchestrator stream teardown to persist conversation with awaited completion at end-of-stream, reducing detached-task/span lifecycle noise during request finalization.
- **`azure-search-documents` version conflict:** Updated from `11.7.0b1` to `11.7.0b2` to align with `agent-framework-azure-ai-search` dependency.
- **`opentelemetry-semantic-conventions-ai` version incompatibility:** Pinned to `>=0.4.13,<0.5.0` to prevent `SpanAttributes.LLM_SYSTEM` missing error in `agent-framework-core`.
- **`opentelemetry-instrumentation-httpx` version conflict:** Unpinned from `==0.52b1` to allow pip to resolve a compatible version with `agent-framework-core` and `semantic-kernel`.

### Removed
- **`SingleAgentRAGStrategyV1` (`single_agent_rag_v1`):** Removed the legacy V1 RAG strategy and its alias module. The `single_agent_rag` key now exclusively maps to V2.

## [v2.4.2] - 2026-02-28
### Added
- **Microsoft Agent Framework (MAF) + Azure AI Foundry Agent Service v2:** Upgraded the orchestration platform from `azure-ai-projects==1.1.0b3` to `azure-ai-projects==2.0.0b3`, introducing the new decoupled `AgentsClient` (`azure.ai.agents.aio.AgentsClient`) with native event-handler streaming, replacing the legacy `AIProjectClient` polling-based approach.
- **SingleAgentRAGStrategyV2:** New orchestration strategy optimized for latency built on top of MAF v2. Incorporates dynamic routing that cuts Time To First Byte (TTFB) from multiple seconds to sub-milliseconds by bypassing the cloud Agent backend when the index is empty, routing to direct in-memory LLM streaming.
- **Granular MAF Latency Telemetry:** Added detailed millisecond-level latency tracking (`[Agent Flow V2][Telemetry]`) to measure Thread initialization, Agent Creation, Message Insertion, and Time To First Token (TTFB) when interacting with the V2 SDK.
- **Identity Singleton Pattern:** Implemented a new `IdentityManager` to centralize Azure Entra ID credential instantiation across plugins and connections, significantly minimizing repetitive token acquisition overhead (`~0.5s` savings per external request).
- **FastAPI Startup Pre-warming:** Injected `lifespan` routines in `main.py` to pre-fetch Azure tokens and pre-warm Azure AI Search indices upon application boot, successfully eliminating Cold-Start latencies on the first user query.

### Changed
- **Asynchronous CosmosDB Persistence (Latency Reduction):** Conversation history saving now runs in the background using `asyncio.create_task` within `orchestrator.py`, decoupling database I/O from the HTTP response and reducing total request latency by ~4 seconds.
- **Index Validation Cache (Latency Bypass):** The `is_index_empty` check in Azure AI Search now utilizes a local in-memory cache. This prevents repetitive network penalties (~3-5 seconds) caused by constantly probing the index; validation now takes `0.00s`.
- **Native Streaming Integration (MAF V2):** Replaced the old asynchronous polling loop for Azure AI Foundry with Event Handlers from the new `AgentsClient` API to dispatch the response stream directly to the user at ultra-fast speeds.

### Performance
- **CosmosDBClient Persistent Connection + Singleton:** Replaced per-operation `CosmosClient` creation (`async with CosmosClient(...)` on every CRUD call) with a single persistent client instance reused across the application lifetime. Conversation persist latency reduced from `~4.38s` to `~0.88s` (`-80%`).
- **SearchClient Singleton + Shared aiohttp Session:** Eliminated per-request `aiohttp.ClientSession()` creation by introducing a lazily-initialized shared session (`_get_session()`) and a module-level `get_search_client()` singleton. Removes redundant TCP/TLS handshakes on every search call.
- **AgentsClient Module-Level Singleton:** Replaced per-stream `AgentsClient` instantiation with a module-level singleton (`_agents_client`), eliminating repeated TCP/TLS negotiation and token acquisition on every user request.
- **AgentsClient Startup Pre-warming (`prewarm_agents_client`):** Added a startup routine that creates the `AgentsClient` singleton and forces the first HTTP round-trip (`list_agents(limit=1)`) during FastAPI `lifespan`, so the initial user request avoids the `~5-6s` cold-start penalty for connection setup and token acquisition.
- **GenAIModelClient Singleton Reuse:** Strategy V2 now uses `get_genai_client()` instead of creating a new `GenAIModelClient()` per instantiation, sharing the pre-warmed embedding and LLM client across all strategies and plugins.
- **Parallel Thread + Agent Creation (`asyncio.gather`):** Thread initialization and Agent creation API calls to Azure AI Foundry now execute concurrently via `asyncio.gather()`, saving `~1-2s` by overlapping two independent network round-trips that were previously sequential.
- **Agent Pre-Creation at Startup (`prewarm_agents_client`):** The reusable MAF agent (with tools and instructions) is now created once during FastAPI startup and cached at module level (`_cached_agent`). Every request resolves the agent in `~0ms` instead of issuing a `create_agent` POST + `delete_agent` per request. Thread + Agent setup reduced from `~4.34s` to `~2.60s` (`-40%`), where the remaining time is purely thread creation (irreducible Azure API latency). Also supports pre-fetching when `AGENT_ID` is configured externally.
- **Orchestrator.create Overhead Reduction:** All connector initializations in `Orchestrator.__init__` and strategy constructors now resolve to pre-warmed singletons, reducing `Orchestrator.create` latency from `~0.62s` to `~0.005s` (`-99%`).
- **Overall Controllable Overhead:** Total application-side controllable overhead (excluding irreducible Azure service latency) reduced from `~12.2s` to `~6.2s` (`-49%`).

### Fixed
- Fixed `AttributeError: enable_auto_function_calls` due to a change in the tool registration API in SDK v2 compared to V1.
- Fixed `AttributeError: threads` on `AIProjectClient` by refactoring the orchestration to use the new decoupled 2.0 API `azure.ai.agents.aio.AgentsClient`.
- Resolved Jinja2 error (`'aisearch_enabled' is undefined`) ensuring the `aisearch` and `bing` contexts are properly added when injecting base prompts.
- Reduced Cosmos DB update error log verbosity by emitting concise status-based warnings and moving detailed diagnostics to debug logs.
- Fixed V2 direct LLM bypass streaming call to use the correct OpenAI chat client API (`openai_client.chat.completions.create`) instead of an invalid `model_client` attribute.
- Removed explicit `temperature/top_p` in V2 direct LLM bypass streaming to stay compatible with models that only support default sampling values.
- Reinforced single-agent prompt language policy so fallback/uncertainty responses stay in the user's language instead of defaulting to English.
- Fixed stale Azure AI Search empty-index cache in V2 routing by adding TTL-based cache revalidation and cache self-healing when retrieval returns documents.

## [v2.4.1] – 2026-02-04
### Fixed
- Updated the Docker image to install Microsoft's current public signing key, fixing build failures caused by SHA-1 signature rejection in newer Debian/apt verification policies.
- Fixed Docker builds on ARM-based machines by explicitly setting the target platform to `linux/amd64`, preventing Azure Container Apps deployment failures.
### Changed
- Pinned the Docker base image to `mcr.microsoft.com/devcontainers/python:3.12-slim` to ensure stable package verification behavior across environments.
- Bumped `aiohttp` to `3.13.3`.
- Standardized on the container best practice of using a non-privileged port (`8080`) instead of a privileged port (`80`), reducing the risk of runtime/permission friction and improving stability of long-running ingestion workloads.

## [v2.4.0] – 2026-01-15
### Added
- End-to-end document-level security: added Microsoft Entra ID authentication in the UI and end-user access token validation in the orchestrator to establish user identity and authorization context.
  Retrieval enforcement uses Azure AI Search native ACL/RBAC trimming with end-user identity propagation via `x-ms-query-source-authorization`, plus permission-aware indexing metadata (userIds/groupIds/rbacScope), safe-by-default behavior when no valid user token is present, and optional elevated-read debugging support.

## [v2.3.0] – 2025-12-15
### Added
- Refactored Single Agent Strategy to simplify citation handling. [#161](https://github.com/Azure/gpt-rag-orchestrator/pull/161)
- Simplified MCP Strategy. [#159](https://github.com/Azure/gpt-rag-orchestrator/pull/159)
### Tested
- Compatibility with Azure direct models for inference

## [v2.2.1] – 2025-10-21
### Added
- Added more troubleshooting logs.
### Fixed
- Citations [387](https://github.com/Azure/GPT-RAG/issues/387)

## [v2.2.0] – 2025-10-16
### Added
- Added [Agentic Retrieval](https://github.com/Azure/GPT-RAG/issues/359) to the Single Agent Strategy.

## [v2.1.0] – 2025-08-31
### Added
- User Feedback Loop. [#358](https://github.com/Azure/GPT-RAG/issues/358) 
### Changed
- Standardized resource group variable as `AZURE_RESOURCE_GROUP`. [#365](https://github.com/Azure/GPT-RAG/issues/365)

## [v2.0.3] – 2025-08-20
### Added
- NL2SQL docs and improved settings checks.

## [v2.0.2] – 2025-08-18
### Fixed
- Corrected v2.0.1 deployment issues.

## [v2.0.1] – 2025-08-08
### Fixed
- Corrected v2.0.0 deployment issues.

## [v2.0.0] – 2025-07-22
### Changed
- Major architecture refactor to support the vNext architecture.

## [v1.0.0] 
- Original version.
