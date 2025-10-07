import os
import logging
from datetime import datetime, UTC
from typing import List, Dict, Any
from azure.cosmos import CosmosClient, exceptions
from azure.core import MatchConditions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
load_dotenv()

def _client():
    db_id = os.getenv("AZURE_DB_ID")
    if not db_id:
        raise ValueError("AZURE_DB_ID environment variable not set")
    return CosmosClient(
        url=f"https://{db_id}.documents.azure.com:443/",
        credential=DefaultAzureCredential()
    )


def cosmos_container():
    db = os.getenv("AZURE_DB_NAME")
    if not db:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    cont = os.getenv("COSMOS_CONTAINER", "reportJobs")
    return _client().get_database_client(db).get_container_client(cont)


async def load_scheduled_jobs() -> List[Dict[str, Any]]:
    """
    Returns jobs that are due now.
    MUST include _etag -> mapped to 'etag' for the activity.
    """
    c = cosmos_container()
    query = "SELECT c.id, c.job_id, c.organization_id, c.tenant_id, c.schedule_time, c.status, c._etag FROM c WHERE c.status = 'QUEUED' AND c.schedule_time <= @now"
    params = [{"name": "@now", "value": datetime.now(UTC).isoformat()}]
    items = list(c.query_items(query=query, parameters=params, enable_cross_partition_query=True))

    # Normalize fields the orchestrators/activities expect
    jobs = []
    for it in items:
        jobs.append({
            "job_id": it.get("job_id") or it.get("id"),
            "organization_id": it["organization_id"],
            "tenant_id": it["tenant_id"],
            "etag": it.get("_etag")
        })
    return jobs


def try_mark_job_running(container, job_id: str, organization_id: str, etag: str) -> bool:
    """
    Optimistic transition QUEUED -> RUNNING using ETag to avoid duplicates.
    """
    try:
        # Load existing doc to preserve its body (replace only status)
        existing = container.read_item(item=job_id, partition_key=organization_id)
        existing["status"] = "RUNNING"
        container.replace_item(item=job_id, body=existing, etag=etag, match_condition=MatchConditions.IfNotModified)
        return True
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 412:
            logging.info(f"[Idempotency] Job {job_id} already taken (etag mismatch).")
            return False
        elif e.status_code == 404:
            logging.error(f"[Cosmos] Job {job_id} not found for organization {organization_id}")
            raise ValueError(f"Job {job_id} not found in database")
        raise


def mark_job_result(container, job_id: str, organization_id: str, status: str, error: str = None):
    try:
        existing = container.read_item(item=job_id, partition_key=organization_id)
        existing["status"] = status
        existing["completed_at"] = datetime.now(UTC).isoformat()
        if error:
            existing["error"] = error
        container.replace_item(item=job_id, body=existing)
    except exceptions.CosmosHttpResponseError as e:
        if e.status_code == 404:
            logging.warning(f"[Cosmos] Job {job_id} not found for organization {organization_id}, skipping mark_job_result")
        else:
            raise
    except Exception as e:
        logging.error(f"[Cosmos] mark_job_result failed for {job_id}: {e}")
