# Python and FastAPI

- Target the Python version declared in `pyproject.toml` and deployment
  workflows.
- Keep modules cohesive and names intent-revealing. Extract logic when a file
  mixes responsibilities or behavior is genuinely reused, not to create
  speculative abstraction.
- Use type hints and Pydantic models at API, connector, plugin, strategy,
  persistence, and configuration boundaries.
- Keep FastAPI handlers limited to HTTP concerns and delegation.
- Prefer small async functions and context-managed resource lifecycles.
- Never perform blocking network calls or unbounded CPU work on the event
  loop.
- Avoid mutable default arguments and process-global request state.
- Propagate cancellation and close request-scoped async clients reliably.
- Catch only exceptions that can be handled meaningfully. Preserve actionable
  context and do not return a success-shaped fallback for a failed requested
  operation.
- Use the configured logging and telemetry paths rather than `print`.
- Keep prompts in `src/prompts/` and configuration in App Configuration.
  Hardcoded runtime instructions, endpoints, deployment names, and feature
  flags do not belong in code paths.
- Reuse existing fixtures and dependency seams in tests.
