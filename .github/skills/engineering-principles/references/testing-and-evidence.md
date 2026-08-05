# Testing and evidence

Choose validation according to the changed boundary:

- Local logic: focused pytest unit tests.
- Async orchestration or strategy lifecycle: streaming, cancellation,
  request-scope, and cleanup tests.
- FastAPI routes: request/response tests with dependency overrides.
- Connectors and plugins: integration-style tests with mocked Azure and
  network boundaries.
- Pydantic or JSON schemas: contract, serialization, and fixture tests.
- Identity and document authorization: negative tests proving an
  unauthorized principal cannot access protected content.
- Dashboard: existing frontend lint/build plus backend route tests.
- PowerShell or shell deployment assets: syntax and behavioral parity checks.
- Cross-repository changes: validate the exact compatible commits or tags.

Use `tests/conftest.py` fixtures instead of live Azure clients. The full Python
suite is run with `pytest`; the repository's pytest configuration enables
asyncio automatically and adds `src/` to `PYTHONPATH`.

For every change, capture:

1. acceptance criterion and observable result;
2. commands run and results;
3. relevant configuration, dependency, and component versions;
4. API, persistence, security, deployment, and rollback impact; and
5. validation that could not run and the resulting risk.

Do not treat a successful deployment as sufficient evidence when a change
affects authorization, data correctness, retrieval quality, streaming,
cleanup, or upgrade behavior.
