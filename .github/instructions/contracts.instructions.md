---
applyTo: "contracts/**,src/schemas.py"
---

# API and shared contracts

- Treat Pydantic and JSON schemas as versioned compatibility boundaries.
- Preserve existing consumers by default; prefer additive optional fields
  when semantics remain compatible.
- Update schema versions when interpretation changes.
- Keep logical and Application Insights wire schemas aligned.
- Regenerate `contracts/audit-event-v1.sha256` from the exact committed schema
  bytes whenever the protected schemas change.
- Consumers must ignore unknown optional fields unless the contract states
  otherwise.
- Coordinate orchestrator, ingestion, and platform changes when a shared
  contract changes.
- Add or update schema, serialization, and consumer fixtures and tests.
- Do not claim legal or regulatory compliance from technical audit evidence.
- Load `architecture-decision` for breaking or cross-repository contract
  changes.
