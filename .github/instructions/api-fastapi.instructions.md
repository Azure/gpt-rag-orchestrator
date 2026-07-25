---
applyTo: "src/main.py,src/api/**/*.py,src/schemas.py"
---

# FastAPI and API contracts

- Keep route handlers thin: validate HTTP input, resolve dependencies,
  delegate to orchestration or a focused service, and shape the response.
- Put new routers under `src/api/` rather than expanding `src/main.py` unless
  the route is part of the existing top-level orchestration contract.
- Preserve SSE framing, status codes, response headers, authentication
  dependencies, and documented request/response schemas by default.
- Use Pydantic models and explicit type hints at HTTP boundaries.
- Never log authorization headers, API keys, cookies, tokens, prompts, or
  response content unless an existing approved bounded/redacted contract
  explicitly permits it.
- Keep synchronous I/O out of async request handlers.
- Map domain or connector failures to actionable HTTP errors without exposing
  secrets or provider internals.
- Add focused route tests with dependency overrides; do not contact live Azure
  services.
- Load `documentation-consistency` for any user-visible API or dashboard
  behavior change.
