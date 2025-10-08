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
from report_scheduler import get_all_organizations, get_brands, get_products, get_competitors
from report_scheduler import create_brands_payload, create_products_payload, create_competitors_payload

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


@app.activity_trigger(input_name="batch_date")
async def LoadScheduledJobsActivity(batch_date: str) -> List[Dict[str, Any]]:
    """
    Durable Activity: Load all scheduled jobs for a batch run.

    This activity queries Cosmos DB and external systems to build the list
    of all report jobs that need to be processed. It's implemented as an
    Activity (not in orchestrator) to maintain orchestrator determinism.

    Args:
        batch_date: ISO timestamp of when the batch was scheduled

    Returns:
        List of job dictionaries, each containing:
            - job_id: Created from Cosmos DB job document
            - organization_id: Organization identifier
            - report_key: Type of report
            - params: Report-specific parameters
            - attempt: Always 1 for scheduled jobs
    """
    logger.info(f"[LoadScheduledJobsActivity] Loading jobs for batch: {batch_date}")

    all_jobs = []

    try:
        # Get all organizations
        organizations = get_all_organizations()
        logger.info(f"[LoadScheduledJobsActivity] Found {len(organizations)} organizations")

        for org in organizations:
            org_id = org.get("id")
            industry_description = org.get("industry_description", "")

            if not org_id:
                logger.warning(f"[LoadScheduledJobsActivity] Organization missing ID, skipping: {org}")
                continue

            logger.info(f"[LoadScheduledJobsActivity] Processing organization: {org_id}")

            # Get brands for this organization
            try:
                brands = get_brands(org_id)
                logger.info(f"[LoadScheduledJobsActivity] Found {len(brands)} brands for {org_id}")

                for brand in brands:
                    brand_name = brand.get("name")
                    if not brand_name:
                        continue

                    payload = create_brands_payload(brand_name, industry_description)
                    all_jobs.append({
                        "job_id": f"{org_id}-brand-{brand_name}",  # Simple ID for now
                        "organization_id": org_id,
                        "report_key": payload["report_key"],
                        "params": payload["params"],
                        "attempt": 1
                    })

            except Exception as e:
                logger.error(f"[LoadScheduledJobsActivity] Error fetching brands for {org_id}: {e}")

            # Get products for this organization
            try:
                products = get_products(org_id)
                logger.info(f"[LoadScheduledJobsActivity] Found {len(products)} products for {org_id}")

                # Group products by category
                category_map = {}
                for product in products:
                    product_name = product.get("name")
                    product_category = product.get("category")
                    if not product_name or not product_category:
                        continue

                    if product_category not in category_map:
                        category_map[product_category] = []
                    category_map[product_category].append(product_name)

                # Create one job per category
                for category, product_names in category_map.items():
                    payload = create_products_payload(product_names, category)
                    all_jobs.append({
                        "job_id": f"{org_id}-product-{category}",
                        "organization_id": org_id,
                        "report_key": payload["report_key"],
                        "params": payload["params"],
                        "attempt": 1
                    })

            except Exception as e:
                logger.error(f"[LoadScheduledJobsActivity] Error fetching products for {org_id}: {e}")

            # Get competitors for this organization
            try:
                competitors = get_competitors(org_id)
                logger.info(f"[LoadScheduledJobsActivity] Found {len(competitors)} competitors for {org_id}")

                # Get brand names for context
                brand_names = [brand.get("name") for brand in brands if brand.get("name")]

                for competitor in competitors:
                    competitor_name = competitor.get("name")
                    if not competitor_name:
                        continue

                    payload = create_competitors_payload(competitor_name, brand_names, industry_description)
                    all_jobs.append({
                        "job_id": f"{org_id}-competitor-{competitor_name}",
                        "organization_id": org_id,
                        "report_key": payload["report_key"],
                        "params": payload["params"],
                        "attempt": 1
                    })

            except Exception as e:
                logger.error(f"[LoadScheduledJobsActivity] Error fetching competitors for {org_id}: {e}")

        logger.info(f"[LoadScheduledJobsActivity] Loaded {len(all_jobs)} total jobs")
        return all_jobs

    except Exception as e:
        logger.error(f"[LoadScheduledJobsActivity] Error loading jobs: {e}\n{traceback.format_exc()}")
        # Return empty list on error - orchestrator can handle this
        return []
