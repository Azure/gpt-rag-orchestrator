from function_app import app
import azure.functions as func
import azure.durable_functions as df
import logging
from datetime import datetime
from shared.cosmos_jobs import load_scheduled_jobs


# @app.timer_trigger(schedule="0 0 2 * * 0", arg_name="mytimer", run_on_startup=False)
# @app.durable_client_input(client_name="client")
# async def report_scheduler_timer(mytimer: func.TimerRequest, client: df.DurableOrchestrationClient):
#     """
#     Kicks off MainOrchestrator every Sunday 02:00 UTC.
#     Uses a unique instance id per date to avoid collision.
#     """
#     logging.info("[Scheduler] Loading due jobs…")
#     jobs = await load_scheduled_jobs()
#     instance_id = f"weekly-scheduler-{datetime.utcnow().strftime('%Y%m%d')}"
#     logging.info(f"[Scheduler] Starting orchestration {instance_id} with {len(jobs)} jobs")
#     await client.start_new("MainOrchestrator", instance_id=instance_id, client_input=jobs)
