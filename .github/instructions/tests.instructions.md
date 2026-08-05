---
applyTo: "tests/**/*.py"
---

# Python tests

- Use pytest with the repository's `asyncio_mode = "auto"` configuration.
- Reuse `mock_config`, `mock_cosmos`, `mock_identity_manager`, and
  `patch_dependencies` from `tests/conftest.py`.
- Unit tests must not require Azure credentials, deployed resources, network
  access, or mutable global cloud state.
- Test observable behavior, contracts, authorization failures, lifecycle, and
  cleanup rather than private implementation details.
- Use `AsyncMock` for awaited boundaries and keep async assertions explicit.
- Add negative tests for invalid configuration, unauthorized access, unsafe
  SQL, untrusted MCP inputs, and sensitive telemetry where applicable.
- Keep fixtures deterministic and free of real endpoints, secrets, tokens,
  and personal data.
- Run the narrowest affected test files first and the full suite when the
  changed boundary or risk warrants it.
- Load the testing reference from `engineering-principles` for contract,
  security, integration, or cross-repository validation.
