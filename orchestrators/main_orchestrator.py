from function_app import app
import azure.durable_functions as df
from collections import defaultdict

WAVE_SIZE = 1  # Set to 1 to match global_limit=1 for sequential execution


@app.orchestration_trigger(context_name="context")
def MainOrchestrator(context: df.DurableOrchestrationContext):
    """
    Groups jobs by tenant and kicks TenantOrchestrator in bounded waves.
    Input: [ {job with tenant_id, organization_id, job_id, etag}, ...]
    """
    all_jobs = context.get_input() or []

    grouped = defaultdict(list)
    for j in all_jobs:
        grouped[j["tenant_id"]].append(j)

    results = []
    wave = []
    for tenant_id, jobs in grouped.items():
        wave.append(
            context.call_sub_orchestrator(
                "TenantOrchestrator", {"tenant_id": tenant_id, "jobs": jobs}
            )
        )
        if len(wave) >= WAVE_SIZE:
            batch_res = yield context.task_all(wave)
            results.extend(batch_res)
            wave = []

    if wave:
        batch_res = yield context.task_all(wave)
        results.extend(batch_res)

    return results
