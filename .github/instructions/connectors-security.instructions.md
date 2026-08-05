---
applyTo: "src/connectors/**/*.py,src/plugins/**/*.py,src/dependencies.py,src/telemetry/**/*.py,src/util/jwt_utils.py"
---

# Connectors, tools, identity, and telemetry

- Keep each connector focused on one external system or protocol.
- Reuse the singleton identity and configuration providers where intended;
  preserve request-scoped clients where credentials or cleanup require them.
- Prefer managed identity and least-privilege RBAC. Resolve secrets through
  Key Vault references and never log credential values.
- Preserve App Configuration label precedence and explicit type conversion.
  New settings need safe defaults, validation, operator documentation, and
  dashboard metadata when editable.
- Validate remote endpoints, hosts, transport modes, schemas, timeouts, output
  limits, and credential sources before use.
- Treat MCP tool arguments/results, retrieval content, model output, JWT
  claims, and external errors as untrusted.
- Preserve OBO and document-level authorization; never replace caller-scoped
  access with managed identity when the source requires caller authorization.
- Bound and redact telemetry. Do not add sensitive content to traces, logs,
  metrics, audit events, or baggage by default.
- Surface connector failures with actionable context. Do not silently convert
  a requested dependency failure into success.
- Mock Azure and network boundaries in unit tests.
- Load `engineering-principles` for identity, security, retrieval, MCP, or
  shared configuration changes.
