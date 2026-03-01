# Changelog

All notable changes to this project will be documented in this file.  
This format follows [Keep a Changelog](https://keepachangelog.com/) and adheres to [Semantic Versioning](https://semver.org/).

## [v2.4.2] - 2026-02-28
### Added
- **SingleAgentRAGStrategyV2:** New orchestration strategy optimized for latency and based on the new Microsoft Agent Framework (MAF) v2 (`azure-ai-projects==2.0.0b3`). Incorporates dynamic routing that cuts Time To First Byte (TTFB) from multiple seconds to sub-milliseconds by bypassing the cloud Agent backend when the index is empty, routing to direct in-memory LLM streaming.
- **Granular MAF Latency Telemetry:** Added detailed millisecond-level latency tracking (`[Agent Flow V2][Telemetry]`) to measure Thread initialization, Agent Creation, Message Insertion, and Time To First Token (TTFB) when interacting with the V2 SDK.
- **Identity Singleton Pattern:** Implemented a new `IdentityManager` to centralize Azure Entra ID credential instantiation across plugins and connections, significantly minimizing repetitive token acquisition overhead (`~0.5s` savings per external request).
- **FastAPI Startup Pre-warming:** Injected `lifespan` routines in `main.py` to pre-fetch Azure tokens and pre-warm Azure AI Search indices upon application boot, successfully eliminating Cold-Start latencies on the first user query.

### Changed
- **Asynchronous CosmosDB Persistence (Latency Reduction):** Conversation history saving now runs in the background using `asyncio.create_task` within `orchestrator.py`, decoupling database I/O from the HTTP response and reducing total request latency by ~4 seconds.
- **Index Validation Cache (Latency Bypass):** The `is_index_empty` check in Azure AI Search now utilizes a local in-memory cache. This prevents repetitive network penalties (~3-5 seconds) caused by constantly probing the index; validation now takes `0.00s`.
- **Native Streaming Integration (MAF V2):** Replaced the old asynchronous polling loop for Azure AI Foundry with Event Handlers from the new `AgentsClient` API to dispatch the response stream directly to the user at ultra-fast speeds.

### Fixed
- Fixed `AttributeError: enable_auto_function_calls` due to a change in the tool registration API in SDK v2 compared to V1.
- Fixed `AttributeError: threads` on `AIProjectClient` by refactoring the orchestration to use the new decoupled 2.0 API `azure.ai.agents.aio.AgentsClient`.
- Resolved Jinja2 error (`'aisearch_enabled' is undefined`) ensuring the `aisearch` and `bing` contexts are properly added when injecting base prompts.

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
