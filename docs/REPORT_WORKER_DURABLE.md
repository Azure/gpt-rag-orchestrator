# Durable Report Worker

This document explains how the report worker operates using Azure Durable Functions. It intentionally ignores the legacy queue-trigger path and focuses solely on the Durable Functions implementation.

## Overview

- Scheduled or ad-hoc jobs are orchestrated with Azure Durable Functions.
- Jobs are grouped by tenant and processed with a Durable Entity rate limiter (default global concurrency: 1).
- Each job is executed by a deterministic activity that transitions status, generates a report (markdown → PDF), and writes results to Blob Storage and Cosmos DB.
- Idempotency and duplicate suppression rely on Cosmos DB ETag checks plus a Blob lease lock.

High-level sequence:
1. Load due jobs and start `MainOrchestrator`.
2. `MainOrchestrator` groups jobs by `tenant_id` and fans out sub-orchestrators in waves.
3. `TenantOrchestrator` serializes or limits concurrency per tenant and calls `GenerateReportActivity` with retry.
4. `GenerateReportActivity` performs ETag-based status transition, runs the processor, and returns structured results.
5. The processor generates markdown, converts to PDF, stores the PDF in Blob Storage, and updates the job in Cosmos.

## Entrypoints

- Timer (weekly, Durable): `scheduler/report_scheduler_timer.py`
  - Cron: `0 0 2 * * 0` (Sundays 02:00 UTC)
  - Loads due jobs from Cosmos DB and starts `MainOrchestrator` with a unique `instance_id`.

- HTTP (manual start): `function_app.py` (`POST /api/start-orch`)
  - Starts an orchestration (`OneShotOrchestrator` or `MainOrchestrator`) with provided input payload.

## Orchestrations

### MainOrchestrator
File: `orchestrators/main_orchestrator.py`

- Groups input jobs by `tenant_id` (per-tenant fairness) and starts `TenantOrchestrator` for each group.
- Runs sub-orchestrators in bounded waves (current `WAVE_SIZE = 1`) to limit global fan-out. Tune to match RateLimiter `global_limit`.

### TenantOrchestrator
File: `orchestrators/tenant_orchestrator.py`

- Enforces fairness/throttling with a Durable Entity (`RateLimiter`).
- For each job:
  - Repeatedly calls `RateLimiter.acquire` until capacity is granted (optionally waiting `wait_ms`).
  - Invokes `GenerateReportActivity` with retry (`first_retry_interval≈30s`, `max_number_of_attempts=5`).
  - Releases capacity after each job by calling `RateLimiter.release`.

### OneShotOrchestrator (Test)
File: `orchestrators/oneshot_orchestrator.py`

- Convenience orchestrator for single-job tests; directly calls `GenerateReportActivity`.

## Durable Entity (Rate Limiter)

File: `entities/rate_limiter_entity.py`

- State: `global_limit` (default 1), `global_inflight`, `inflight_by_tenant`, `per_tenant_limit` (default=1).
- Operations:
  - `acquire({tenant_id})` → `{granted: bool, wait_ms?: int}`
  - `release({tenant_id})` → `{ok: true}`
  - `configure({global_limit?, per_tenant_limit?})` → `{ok: true}`

This entity ensures work can be distributed fairly across tenants and provides backpressure. Increase `global_limit` and/or per-tenant limits for parallelism.

## Activities

### GenerateReportActivity
File: `report_worker/activities.py`

- Input: `{ job_id, organization_id, tenant_id?, etag?, attempt? }`.
- If `etag` is present, attempts `QUEUED → RUNNING` with Cosmos If-Match:
  - On ETag mismatch (taken by another worker), returns `{status: "SKIPPED", reason: ...}`.
- Executes with a 30-minute timeout (`asyncio.timeout(1800)`).
- Calls the processor (`process_report_job`) and records success/failure in Cosmos.
- Returns a normalized result: `{ job_id, organization_id, status: SUCCEEDED|FAILED|SKIPPED, ... }`.

Note: The activity expects the job document to already exist in Cosmos DB (status `QUEUED`). If it does not exist, processing will fail with a not-found error. Include `etag` from the job document to enable strict claim (idempotency); omitting `etag` relies on status checks and the blob lease lock.

### LoadScheduledJobsActivity
File: `report_worker/activities.py`

- Optional helper to create a deterministic job list (brands/products/competitors) outside the orchestrator code.
- The weekly timer currently loads due jobs directly via `shared/cosmos_jobs`.

## Processor (Job Execution)

File: `report_worker/processor.py`

Steps per job:
1. Read the job from Cosmos DB. If not found, fail.
2. If status is not `QUEUED`, skip (idempotent re-entry).
3. Update status to `RUNNING` and set `started_at`.
4. Acquire a distributed lock using a Blob lease (`report-locks/{job_id}.lock`).
   - If another worker completes the job during wait, return early.
   - If no lock after 30 minutes, mark as `FAILED` (transient) and raise `TimeoutError`.
5. Select a markdown generator via `report_worker/registry.get_generator(report_key)`.
   - Generators return markdown only; conversion and storage are centralized.
6. Convert markdown → HTML → PDF using `shared/markdown_to_pdf.dict_to_pdf`.
7. Store PDF in Blob Storage: `documents/organization_files/{organization_id}/report_{job_id}_{timestamp}.pdf`.
   - Metadata: `{organization_id, report_id, timestamp, report_key}`.
8. Update job in Cosmos to `SUCCEEDED` with `result` metadata.
9. Always release the Blob lease in `finally`.

On errors:
- Deterministic errors (e.g., missing `report_key` or generator) set `FAILED` with `error_type: "deterministic"`.
- Transient errors set `FAILED` with `error_type: "transient"` and re-raise to respect Durable retry.
- Timeout errors set `FAILED` with `error_type: "timeout"`.

## Report Generators

File: `report_worker/registry.py`

- Base: `ReportGeneratorBase` enforces that generators return markdown.
- Built-in generators:
  - `sample`
  - `brand_analysis`
  - `competitor_analysis`
  - `product_analysis`
- Generators rely on `reports/report_generator.py` to produce the markdown (LLM-based) and are deliberately kept deterministic at the activity boundary.

To add a new type:
1. Create a new subclass of `ReportGeneratorBase` with a `generate(...) -> str` method.
2. Add the key to `VALID_REPORT_TYPES` and an instance to `_REPORT_GENERATORS`.
3. Ensure jobs include the new `report_key` and expected `params`.

## Scheduling

### Weekly Batch (Durable)
File: `scheduler/report_scheduler_timer.py`

- Loads due jobs via `shared/cosmos_jobs.load_scheduled_jobs()` (status `QUEUED` and `schedule_time <= now`).
- Starts `MainOrchestrator` with a date-based `instance_id` (one orchestration per weekly run).

### Manual Start (HTTP)
File: `function_app.py`

- Endpoint: `POST /api/start-orch`
- Request examples:

One-shot single job (job must already exist in Cosmos DB):
```json
{
  "orchestrator": "OneShotOrchestrator",
  "input": {
    "job_id": "test-sample-001",
    "organization_id": "<org>",
    "report_key": "sample",
    "params": {},
    "attempt": 1
  }
}
```

Multi-tenant batch (each job must already exist in Cosmos DB):
```json
{
  "orchestrator": "MainOrchestrator",
  "input": [
    {"job_id": "job-1", "organization_id": "org-1", "tenant_id": "A", "report_key": "sample", "params": {}, "attempt": 1},
    {"job_id": "job-2", "organization_id": "org-2", "tenant_id": "B", "report_key": "sample", "params": {}, "attempt": 1}
  ]
}
```

Durable status endpoints (local):
- `GET /runtime/webhooks/durabletask/instances/{instanceId}`
- `GET /runtime/webhooks/durabletask/instances`
- `POST /runtime/webhooks/durabletask/instances/{instanceId}/terminate`

You can also use the Postman collection at `tests/durable-functions-test.postman_collection.json`.

## Data Model & State Transitions

Cosmos container: default `reportJobs` (configurable via env vars).

Representative fields:
- `job_id` (or `id`), `organization_id` (partition key), `tenant_id`
- `report_key`, `params`, `status` ∈ {`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`}
- `schedule_time`, `_etag` (for optimistic concurrency) # IMPORTANT
- Worker-managed timestamps: `started_at`, `updated_at`, `completed_at`/`failed_at`
- On success: `result` metadata (Blob info)
- On failure: `error` payload (`error_type`, message, attempt/timestamps)

Transitions:
1. `QUEUED` → `RUNNING` (ETag If-Match in `GenerateReportActivity`)
2. `RUNNING` → `SUCCEEDED` or `FAILED`

## Storage

### Blob Storage
- Container: `documents`
- Path: `organization_files/{organization_id}/report_{job_id}_{YYYYMMDD_HHMMSS}.pdf`
- Metadata: `{organization_id, report_id, timestamp, report_key}`
- Client: `shared/blob_client_async.get_blob_service_client()`

### Cosmos DB
- Helpers: `shared/cosmos_jobs.py` (ETag transition, result marking), `shared/util.py` (status CRUD)
- Container name defaults:
  - `reportJobs` via `AZURE_REPORT_JOBS_CONTAINER` (util) or `COSMOS_CONTAINER` (cosmos_jobs)

## Idempotency & Concurrency Controls

- ETag Guard: `try_mark_job_running` performs If-Match replace of the job document to claim processing. Prevents duplicate runners for the same `QUEUED` job.
- Blob Lease Lock: Processor acquires a short lease on `report-locks/{job_id}.lock` to prevent concurrent work across workers/functions.
- Durable Retries: Transient failures bubble to Durable which retries per `RetryOptions`.
- Global Concurrency: Defaults to 1 via `RateLimiter.global_limit` and `WAVE_SIZE`. Increase for parallelism as needed.
- Activity Timeout: 30 minutes per job (prevents indefinite execution).

## Configuration

Environment variables of interest:
- Cosmos: `AZURE_DB_ID`, `AZURE_DB_NAME`, `AZURE_REPORT_JOBS_CONTAINER` (or `COSMOS_CONTAINER`)
- Storage: `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCOUNT` (uses `DefaultAzureCredential`)
- Durable hub: uses the configured Storage account (Azurite locally)

## Local Development

Prerequisites:
- Python 3.10+
- Azurite running (Tables/Blobs/Queues) for Durable Functions

Commands:
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Start Functions: `func start` (or VS Code F5)

Test flows:
- Use `POST /api/start-orch` with `OneShotOrchestrator` for fast validation.
- Check orchestration status via Durable runtime endpoints.
- Find generated PDFs in the `documents` container under `organization_files/{organization_id}/`.

## Extensibility Checklist

When adding a new report type:
1. Implement a generator in `report_worker/registry.py` returning markdown.
2. Register the generator and key in `_REPORT_GENERATORS` and `VALID_REPORT_TYPES`.
3. Ensure job creation populates `report_key` and appropriate `params`.
4. Keep activity deterministic; perform I/O in the activity/processor only.

## File Map (Key References)

- Orchestrators: `orchestrators/main_orchestrator.py`, `orchestrators/tenant_orchestrator.py`, `orchestrators/oneshot_orchestrator.py`
- Entity: `entities/rate_limiter_entity.py`
- Activities: `report_worker/activities.py`
- Processor: `report_worker/processor.py`
- Registry: `report_worker/registry.py`
- Scheduler (Durable): `scheduler/report_scheduler_timer.py`
- Cosmos helpers: `shared/cosmos_jobs.py`, `shared/util.py`
- PDF: `shared/markdown_to_pdf.py`
- Blob client: `shared/blob_client_async.py`

---

Notes:
- This document intentionally disregards the legacy queue worker. Keep `ENABLE_LEGACY_QUEUE_WORKER` unset or not equal to `"1"` to avoid legacy behavior.
