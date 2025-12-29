"""
Durable Functions Activities for report generation.

Activities are the actual work units that get executed by the orchestrator.
Each activity should be idempotent and deterministic.
"""

import logging
import traceback
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
import azure.functions as func

from function_app import app
from report_worker.processor import process_report_job
from shared.util import update_report_job_status, get_report_job

from shared.cosmos_jobs import (
    cosmos_container,
    try_mark_job_running,
    mark_job_result
)

logger = logging.getLogger(__name__)


@app.activity_trigger(input_name="job")
async def GenerateReportActivity(job: dict) -> dict:
    """
    Durable Activity: Process a single report generation job.

    This activity wraps the existing process_report_job logic with:
    - Timeout protection (30 minutes)
    - Comprehensive error handling
    - Status tracking
    - Idempotency support

    Args:
        job: Dictionary containing:
            - job_id: Unique job identifier
            - organization_id: Organization/tenant ID
            - tenant_id: Tenant identifier (optional)
            - etag: Cosmos DB _etag for idempotency (optional)
            - attempt: Attempt number (for retries)

    Returns:
        Dictionary with job result:
            - job_id: Job identifier
            - organization_id: Organization ID
            - status: "SUCCEEDED", "FAILED", or "SKIPPED"
            - completed_at: ISO timestamp (if succeeded)
            - error: Error message (if failed)
            - reason: Skip reason (if skipped)
    """
    job_id = job["job_id"]
    organization_id = job["organization_id"]
    tenant_id = job.get("tenant_id")
    etag = job.get("etag")
    attempt = job.get("attempt", 1)

    logger.info(f"[GenerateReportActivity] Starting job {job_id} for org {organization_id} (attempt {attempt})")

    try:
        if etag:
            container = cosmos_container()
            if not try_mark_job_running(container, job_id, organization_id, etag):
                logger.info(f"[GenerateReportActivity] Skip {job_id}, another worker took it.")
                return {"job_id": job_id,
                        "organization_id": organization_id,
                        "status": "SKIPPED",
                        "reason": "Job already taken by another worker"}
        # Add timeout protection (30 minutes max per job)
        async with asyncio.timeout(1800):  # 1800 seconds = 30 minutes
            await process_report_job(job_id, organization_id, attempt)

        completion_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"[GenerateReportActivity] Successfully completed job {job_id}")

        # mark success
        mark_job_result(cosmos_container(), job_id, organization_id, status="SUCCEEDED")

        return {
            "job_id": job_id,
            "organization_id": organization_id,
            "status": "SUCCEEDED",
            "completed_at": completion_time
        }

    except asyncio.TimeoutError:
        error_msg = f"Job {job_id} timed out after 30 minutes"
        logger.error(f"[GenerateReportActivity] {error_msg}")

        # Update job status in Cosmos DB
        error_payload = {
            "error_type": "timeout",
            "error_message": error_msg,
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        update_report_job_status(job_id, organization_id, "FAILED", error_payload=error_payload)
        mark_job_result(cosmos_container(), job_id, organization_id, status="FAILED", error=error_msg)

        return {
            "job_id": job_id,
            "organization_id": organization_id,
            "status": "FAILED",
            "error": error_msg,
            "error_type": "timeout"
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"[GenerateReportActivity] Error for job {job_id}: {error_msg}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        # Update job status in Cosmos DB
        error_payload = {
            "error_type": "unexpected",
            "error_message": error_msg,
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        update_report_job_status(job_id, organization_id, "FAILED", error_payload=error_payload)
        mark_job_result(cosmos_container(), job_id, organization_id, status="FAILED", error=error_msg)

        return {
            "job_id": job_id,
            "organization_id": organization_id,
            "status": "FAILED",
            "error": error_msg,
            "error_type": "unexpected"
        }
