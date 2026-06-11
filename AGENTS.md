## Engineering Standards

### Clean Code and Modularity

All implementations in this repository should follow clean code best practices.
The orchestrator is a Python 3.12 FastAPI service whose value is its agentic
orchestration layer; keep that layer modular and easy to evolve, and avoid
letting any single module become a catch-all for unrelated behavior.

- Keep each module and file focused on a single, clear responsibility.
- Extract non-trivial logic into the right layer under `src/` instead of
  growing the FastAPI entrypoint or a single strategy file:
  - `src/strategies/` — agent strategies (one class per strategy, subclass
    `BaseAgentStrategy`).
  - `src/orchestration/` — orchestration flow and agent wiring.
  - `src/connectors/` — Azure / external service clients (Cosmos, AI Search,
    Foundry, App Config, identity).
  - `src/plugins/` — Semantic Kernel / tool plugins.
  - `src/prompts/` — prompt templates (no hardcoded prompts in code paths).
  - `src/telemetry/` and `src/util/` — cross-cutting helpers.
- Keep FastAPI route handlers thin: validate input, delegate to a
  strategy/orchestration/connector, shape the response. No business logic in
  the route function.
- Prefer small, cohesive `async` functions and classes over large procedural
  blocks. Respect async correctness — never block the event loop with
  synchronous I/O.
- Reuse existing connectors, plugins, prompts, and `util` helpers before
  adding new ones. Avoid duplication and speculative abstractions; extract
  only when code is genuinely repeated or a file is mixing concerns.
- Use clear, intent-revealing names so the code reads without excessive
  comments. Comment only non-obvious decisions.

### New Strategies Are Extension Points

Add a new orchestration behavior by subclassing `BaseAgentStrategy` and
registering it in the `AgentStrategies` enum and `AgentStrategyFactory` — do
not branch existing strategies with conditionals. The active strategy is
selected at runtime via the App Config key `AGENT_STRATEGY`, never a code
constant.

### Configuration, Secrets, and Contracts

- Read runtime settings from **Azure App Configuration** (label `gpt-rag`) via
  the existing config provider; resolve secrets through **Key Vault**
  references. Never hardcode endpoints, deployment names, model names, or
  feature flags in code.
- Prefer typed, explicit data contracts (type hints, dataclasses, or Pydantic
  models) at boundaries — API request/response shapes, inter-service payloads,
  and agent inputs/outputs.
- Surface errors clearly and consistently through the telemetry/logging
  helpers. Do not swallow exceptions or add silent fallbacks that hide a
  broken model, connector, or config state. Never use `print` for
  diagnostics — use the configured logger.

### Testing

This is the only service with a maintained `tests/` suite (`pytest`,
`asyncio_mode = "auto"`).

- Add or update tests when behavior changes or when extracted logic becomes
  independently testable.
- Use the fixtures in `tests/conftest.py` (`mock_config`, `mock_cosmos`,
  `mock_identity_manager`, `patch_dependencies`) instead of reaching for live
  Azure clients.

```powershell
pip install -e ".[test]"
pytest
```
