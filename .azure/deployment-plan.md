# Azure Deployment Preparation Plan

## Status

Ready for Validation

Status history for this pass: `Approved` (plan drafted and scope confirmed)
-> `Ready for Validation` (preparation checks completed, handoff proof
recorded below).

## Scope

Prepare and record validation evidence for the `gpt-rag-orchestrator`
component of [Azure/GPT-RAG#571](https://github.com/Azure/GPT-RAG/issues/571)
("Establish a practical governance baseline and auditable AI activity
trail"). This supersedes the stale plan previously recorded in this file for
`Azure/gpt-rag-orchestrator#260` (dashboard MSAL sign-in), which was handled
separately and no longer applies.

This plan does not authorize deployment, merge, release, deletion, or
Azure/Entra mutation. It records a MODIFY-mode preparation and validation pass
over application configuration that ships inside the existing orchestrator
container image; it does not add, remove, or resize any Azure resource.

## Preparation finding

The orchestrator-side implementation for issue #571 (the shared v1 audit
event contract, disabled-by-default instrumentation, and admin-safe
configuration defaults) was already implemented, reviewed, merged, and
released before this preparation pass started:

- Pull request: [Azure/gpt-rag-orchestrator#277](https://github.com/Azure/gpt-rag-orchestrator/pull/277)
  (`feature/audit-events-v1` -> `develop`, merged).
- Release: [`v3.8.0`](https://github.com/Azure/gpt-rag-orchestrator/releases/tag/v3.8.0)
  (published; `VERSION` file synchronized to `3.8.0` on `develop` and `main`).
- This preparation pass therefore does not introduce new orchestrator
  application code. It validates the already-released state and finalizes
  this deployment-plan artifact, which had not been updated since the
  unrelated `#260` preparation.

## MODIFY mode and recipe

- Mode: MODIFY. No new Azure resource type is introduced. Audit configuration
  is delivered through existing Azure App Configuration key-values (label
  `gpt-rag-orchestrator` or shared `gpt-rag`) and the existing Key Vault
  secret-reference pattern already used for other orchestrator secrets
  (for example `MCP_APP_APIKEY`, `OAUTH_AZURE_AD_CLIENT_SECRET`).
- Recipe: this repository's `infra/` is a documented pointer, not a
  standalone Bicep environment (`infra/README.md`: "The infrastructure for
  this project is defined in: https://github.com/Azure/GPT-RAG"; `infra/main.bicep`
  is a one-line `targetScope = 'resourceGroup'` stub). The existing AZD
  workflow (`azure.yaml` -> `scripts/preProvision.*` / `scripts/deploy.*`)
  builds and deploys the orchestrator container image into the Container App
  provisioned by the main `Azure/GPT-RAG` environment. No changes to that
  recipe are required for this feature; it is pure application configuration
  that ships inside the existing image.

## Architecture

- Audit events reuse the existing OpenTelemetry + Application Insights export
  path (`src/telemetry/telemetry.py`, `src/telemetry/audit.py`) under the
  `gptrag.audit` logger namespace; no new telemetry backend is introduced.
- Configuration is read through the existing App Configuration provider
  (`src/api/config_settings.py`), the same mechanism used for every other
  runtime setting.
- The shared v1 contract (`contracts/audit-event-v1.schema.json` and
  `contracts/audit-event-v1.application-insights.schema.json`) is versioned
  and additive; the ingestion component reserves its own event names in the
  same schema but the orchestrator does not emit them.
- Cross-repo alignment: `gpt-rag-orchestrator` v3.8.0 is the paired release
  that carries this shared contract; `gpt-rag-ingestion` v2.5.0 is the
  expected companion release on the ingestion side for the remaining
  `Azure/GPT-RAG#571` scope (ingestion audit events, ingestion governance
  defaults, and additive Azure AI Search index fields). Ingestion-side and
  Azure/GPT-RAG umbrella-repo changes are **out of scope of this workspace**
  (a different repository/session owns them); see "Cross-repo scope" below.

## Pinned contract identifiers

The shared v1 contract ships with a committed SHA-256 manifest
(`contracts/audit-event-v1.sha256`) so any consumer (including
`gpt-rag-ingestion`) can detect drift before trusting the schema:

| Artifact | SHA-256 (raw bytes) |
| --- | --- |
| `contracts/audit-event-v1.schema.json` | `825db8ef40a81e2c19e5d80d37c565b6b47fc9a6540e9881d35cc12b8fde5aab` |
| `contracts/audit-event-v1.application-insights.schema.json` | `066c8f5408610ab839d5121d06ca5bc59e8797e551d5c47c875c5ba52f7e0588` |

Both hashes were independently reproduced in this pass (raw file bytes,
`\r\n` normalized to `\n`) and match the committed manifest and the values
required by `Azure/GPT-RAG#571`. `tests/test_audit_contract.py::test_published_contract_hashes_match_artifacts`
enforces this on every test run, so any future edit to either schema file
without updating `contracts/audit-event-v1.sha256` fails CI instead of
silently drifting from what `gpt-rag-ingestion` expects.

## Security

- `AUDIT_HMAC_KEY` (the 256-bit pseudonymization key) is read only through the
  existing Key Vault secret-reference pattern in App Configuration, exactly
  like other orchestrator secrets. It is denylisted in
  `src/api/config_settings.py::DENYLIST`, so it can never be read or written
  through the admin dashboard, and it is never logged, echoed in an audit
  event, or placed in documentation as a literal value.
- Enabling `AUDIT_EVENTS_ENABLED=true` without a valid 256-bit key (base64,
  base64url, or hex) fails startup (`AuditConfigurationError`) instead of
  silently degrading to an unkeyed or predictable pseudonym.
- Sensitive-content capture (`AUDIT_SENSITIVE_CONTENT_ENABLED`) stays
  disabled by default and requires a non-empty allowlist
  (`AUDIT_SENSITIVE_CONTENT_FIELDS`) drawn from a fixed, small set of fields.
  Tokens, credentials, authorization headers, cookies, connection strings,
  SAS values, and private keys are filtered before export even when
  sensitive capture is enabled (defense in depth, not a DLP boundary).
- Generation guidance (README "Audit event configuration") uses
  `secrets.token_bytes(32)` locally, writes directly to Key Vault, and clears
  the shell variable; no hardcoded secret exists anywhere in the repository.

## Migration and rollback

- All seven audit configuration keys are additive App Configuration
  key-values with safe, documented defaults
  (`AUDIT_EVENTS_ENABLED=false`, `AUDIT_SENSITIVE_CONTENT_ENABLED=false`,
  `AUDIT_SENSITIVE_CONTENT_FIELDS=` (empty), `AUDIT_ACTOR_PSEUDONYM_ENABLED=false`,
  `AUDIT_SOURCE_EVENT_LIMIT=25`, `AUDIT_HMAC_KEY_ID=v1`,
  `AUDIT_ADDITIONAL_REDACTED_KEYS=` (empty)). Existing deployments upgrade to
  the new image with no configuration changes and preserve current behavior
  (`AUDIT_EVENTS_ENABLED` absent is treated the same as `false`).
- Rollback: set `AUDIT_EVENTS_ENABLED=false` and restart/re-revision the
  Container App, or roll back to the previous image. Neither action deletes
  already-exported telemetry or requires any resource or index change.
- No destructive schema/index recreation is part of this change; the
  additive Azure AI Search index fields described in `Azure/GPT-RAG#571`
  belong to `gpt-rag-ingestion` and are not implemented in this repository.

## No destructive deployment

- This pass performs no `azd provision`, `azd deploy`, resource creation, or
  resource deletion.
- No resource group is created, modified, or deleted by this plan.
- Before any future deletion consideration in any environment, the safety
  rules from the prior `#260` plan remain in force: check `tags` for
  `keep=true`, never delete by `gpt-rag`/`gptrag` substring match, and only
  delete an exact, explicitly approved resource group name.

## Cross-repo scope (explicitly not implemented here)

`Azure/GPT-RAG#571` lists both `gpt-rag-orchestrator` and `gpt-rag-ingestion`
as in-scope components. This workspace is the `gpt-rag-orchestrator`
repository only. The following remain open and belong to other
repositories/sessions:

- `gpt-rag-ingestion` v2.5.0: ingestion-side audit events, ingestion
  governance defaults (`INGESTION_PROVENANCE_ENABLED`,
  `INGESTION_REQUIRE_GOVERNANCE_METADATA`, `INGESTION_DEFAULT_CLASSIFICATION`,
  `INGESTION_DEFAULT_RIGHT_TO_USE`), and the additive Azure AI Search index
  fields (`provenance_id`, `source_uri_id`, `source_version_id`,
  `content_checksum_sha256`, `ingested_at`, `ingest_run_id`,
  `data_classification`, `right_to_use`, `retention_class`, `delete_after`).
  `delete_after` must be documented there as policy intent (a value read by
  operator tooling/reporting), not an automatic purge mechanism.
- `Azure/GPT-RAG` (umbrella repository): pinning `gpt-rag-orchestrator`
  v3.8.0 and `gpt-rag-ingestion` v2.5.0 together for a future minor GPT-RAG
  release, and the umbrella `CHANGELOG.md`/release notes for that release.
- Stage 1, 4, and 5 governance/documentation deliverables from the issue
  (governance statement, responsibility guidance, reconstruction query
  examples, retention/deletion/access guidance) that are not specific to the
  orchestrator's own README.

## Validation performed in this pass

| Check | Command | Result |
| --- | --- | --- |
| Full unit test suite | `python -m pytest -q` | 533 passed, 0 failed (109.82s) |
| Audit-contract lint scope | `python -m ruff check src/telemetry/audit*.py src/telemetry/__init__.py src/telemetry/telemetry.py src/main.py src/api/config_settings.py scripts/benchmark_audit.py tests/test_audit_*.py` | All checks passed |
| Byte compile | `python -m compileall -q src tests scripts` | Exit code 0 |
| Dependency consistency | `python -m pip check` | No broken requirements found |
| Bicep stub build | `az bicep build --file infra/main.bicep --stdout` | Builds to an empty-resources ARM template (confirms the stub is syntactically valid; there is no other infra in this repo to validate) |
| Docker image build | `docker build -t gpt-rag-orchestrator:validate-571 .` | Builds successfully (multi-stage Node 20 dashboard build + Python 3.12 runtime); image removed after validation |
| Contract hash reproduction | raw SHA-256 of both `contracts/audit-event-v1*.schema.json` files | Matches `contracts/audit-event-v1.sha256` and the values required by `Azure/GPT-RAG#571` (see table above) |

Repo-wide `ruff check src tests scripts` (unscoped) reports 78 pre-existing
findings in files unrelated to the audit contract (for example unused
imports in `tests/test_conversation_history.py` and
`tests/test_maf_agent_service_strategy.py`). These predate this pass, are
outside the `Azure/GPT-RAG#571` scope, and are not modified here to keep this
change focused.

Not run / explicit blocker:

- `azd provision --preview` (or equivalent non-deploy preflight): this
  repository's `infra/` intentionally has no standalone environment to
  preview against (see "MODIFY mode and recipe" above); a meaningful preview
  requires the main `Azure/GPT-RAG` environment/workspace, which is outside
  this repository and this session. No preview was attempted against any
  shared/real GPT-RAG environment to avoid unintended state changes to
  infrastructure this workspace does not own. This is recorded as the
  explicit blocker for that specific validation step; nothing was deployed.

## Handoff proof

- Test/lint/build evidence: recorded in the table above, produced in this
  session against `origin/develop` at commit `0fccab6` (`chore: sync main to
  develop for v3.8.0`).
- Release evidence: PR [#277](https://github.com/Azure/gpt-rag-orchestrator/pull/277)
  merged; release [v3.8.0](https://github.com/Azure/gpt-rag-orchestrator/releases/tag/v3.8.0)
  published; `VERSION` = `3.8.0` on `develop`/`main`.
- Contract evidence: `contracts/audit-event-v1.sha256` hashes independently
  reproduced and matched in this pass.

## Planning checklist

- [x] Inspect the existing (stale) deployment plan and identify it as
      unrelated to `#571`.
- [x] Confirm MODIFY mode, existing AZD/Bicep recipe, and that no new Azure
      resource is introduced.
- [x] Document architecture, security, migration, and rollback for the
      audit-event feature.
- [x] Reproduce and record the pinned contract hash identifiers.
- [x] Identify and record ingestion-repo and umbrella-repo scope that is not
      implemented here.
- [x] Run full test suite, scoped lint, compileall, pip check, Bicep stub
      build, and Docker build; record results.
- [x] Record the `azd provision --preview` blocker without attempting a live
      preview or deployment.
- [x] Finalize this plan and move status to Ready for Validation with
      handoff proof recorded.
