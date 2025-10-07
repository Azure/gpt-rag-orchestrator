from function_app import app
import azure.durable_functions as df
from datetime import timedelta
import logging


@app.orchestration_trigger(context_name="context")
def TenantOrchestrator(context: df.DurableOrchestrationContext):
    """
    Runs a list of jobs for ONE tenant, sequentially (or you can extend to small batches).
    Expects input: {"tenant_id": "...", "jobs": [ {job}, ... ]}
    """
    payload = context.get_input() or {}
    tenant_id = payload["tenant_id"]
    jobs = list(payload["jobs"])

    limiter = df.EntityId("RateLimiter", "global")
    results = []
    retry_opts = df.RetryOptions(first_retry_interval_in_milliseconds=30000, max_number_of_attempts=5)

    for job in jobs:
        # Acquire capacity
        while True:
            grant = yield context.call_entity(limiter, "acquire", {"tenant_id": tenant_id})
            if grant.get("granted"):
                break
            wait_ms = grant.get("wait_ms", 1500)
            yield context.create_timer(
                context.current_utc_datetime + timedelta(milliseconds=wait_ms)
            )

        # Run the activity (with retry)
        try:
            res = yield context.call_activity_with_retry(
                "GenerateReportActivity", retry_opts, job
            )
            results.append(res)
        except Exception as e:
            # capture failures and continue
            logging.error(f"[TenantOrchestrator] job failed: {e}")
            results.append({"job_id": job.get("job_id"), "status": "FAILED", "error": str(e)})

        # Always release after processing (outside finally to avoid generator cleanup issues)
        yield context.call_entity(limiter, "release", {"tenant_id": tenant_id})

    return results
