from function_app import app
import azure.durable_functions as df


@app.orchestration_trigger(context_name="context")
def OneShotOrchestrator(context: df.DurableOrchestrationContext):
    """
    Simple orchestrator for testing a single job.
    """
    job = context.get_input() or {}
    result = yield context.call_activity("GenerateReportActivity", job)
    return result
