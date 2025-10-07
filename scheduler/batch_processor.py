"""
Helper functions for batch job processing via HTTP endpoint.
"""
import asyncio
import logging
from shared.cosmos_jobs import load_scheduled_jobs

logger = logging.getLogger(__name__)


async def load_and_process_jobs(client):
    """
    Load scheduled jobs from Cosmos DB, wait 10 seconds, then start orchestration.

    Args:
        client: DurableOrchestrationClient instance

    Returns:
        dict: Result with instance_id and job count, or error
    """
    logger.info("Loading scheduled jobs from Cosmos DB...")

    # Load jobs
    jobs = await load_scheduled_jobs()

    if not jobs:
        logger.warning("No jobs found to process")
        return {
            "success": False,
            "message": "No jobs found (check schedule_time <= now and status = QUEUED)",
            "job_count": 0
        }

    logger.info(f"Loaded {len(jobs)} jobs")

    logger.info("Waiting 10 seconds before processing...")
    await asyncio.sleep(10)

    logger.info("Starting MainOrchestrator...")
    instance_id = await client.start_new("MainOrchestrator", client_input=jobs)

    logger.info(f"Orchestration started: {instance_id}")

    return {
        "success": True,
        "instance_id": instance_id,
        "job_count": len(jobs),
        "message": f"Started processing {len(jobs)} jobs"
    }
