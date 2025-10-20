"""
Report Processing Logic

This module contains the main logic for processing report generation jobs.
Separated from function_app.py for better organization and readability.
"""

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
from azure.storage.blob.aio import BlobLeaseClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from shared.util import get_report_job, update_report_job_status
from shared.markdown_to_pdf import dict_to_pdf
from shared.blob_client_async import get_blob_service_client
from .registry import get_generator


async def acquire_job_lock(job_id: str, max_wait: int = 1800) -> Tuple[bool, Optional[BlobLeaseClient]]:
    """
    Try to acquire an exclusive lock for job processing using Azure Blob lease.

    Args:
        job_id: Unique identifier for the job
        max_wait: Maximum time to wait in seconds (default 30 minutes)

    Returns:
        Tuple of (lock_acquired: bool, lease_client: Optional[BlobLeaseClient])
    """
    blob_service_client = await get_blob_service_client()
    container_name = "report-locks"
    blob_name = f"{job_id}.lock"

    # Ensure container exists
    container_client = blob_service_client.get_container_client(container_name)
    try:
        await container_client.get_container_properties()
    except ResourceNotFoundError:
        try:
            await container_client.create_container()
            logging.info(f"[LockManager] Created container: {container_name}")
        except ResourceExistsError:
            pass  # Container created by another worker

    # Get blob client
    blob_client = container_client.get_blob_client(blob_name)

    # Ensure lock blob exists
    try:
        await blob_client.get_blob_properties()
    except ResourceNotFoundError:
        try:
            await blob_client.upload_blob(b"lock", overwrite=False)
        except ResourceExistsError:
            pass  # Blob created by another worker

    # Try to acquire lease
    lease_client = BlobLeaseClient(blob_client)
    waited = 0
    poll_interval = 5  # Check every 5 seconds

    while waited < max_wait:
        try:
            # Try to acquire a 60-second lease
            await lease_client.acquire(lease_duration=60)
            logging.info(f"[LockManager] Acquired lock for job {job_id}")
            return True, lease_client
        except Exception as e:
            logging.warning(f"[LockManager] Lock acquisition failed for job {job_id}: {type(e).__name__}: {str(e)}")
            logging.debug(f"[LockManager] Lock unavailable for job {job_id}, waiting... ({waited}s elapsed)")
            await asyncio.sleep(poll_interval)
            waited += poll_interval

    logging.warning(f"[LockManager] Could not acquire lock for job {job_id} after {max_wait}s")
    return False, None


async def release_job_lock(lease_client: Optional[BlobLeaseClient], job_id: str):
    """
    Release the job lock.

    Args:
        lease_client: The blob lease client holding the lock
        job_id: Job identifier for logging
    """
    if not lease_client:
        return

    try:
        await lease_client.release()
        logging.info(f"[LockManager] Released lock for job {job_id}")
    except Exception as e:
        logging.warning(f"[LockManager] Error releasing lock for job {job_id}: {str(e)}")


async def wait_for_job_completion_or_lock(
    job_id: str,
    organization_id: str,
    max_wait: int = 1800
) -> Tuple[bool, str, Optional[BlobLeaseClient]]:
    """
    Wait for either lock acquisition or job completion by another worker.

    Args:
        job_id: Unique identifier for the job
        organization_id: Organization requesting the report
        max_wait: Maximum time to wait in seconds

    Returns:
        Tuple of (lock_acquired: bool, job_status: str, lease_client: Optional[BlobLeaseClient])
    """
    poll_interval = 10  # Check job status every 10 seconds
    waited = 0

    while waited < max_wait:
        # Check if job is already completed
        job = get_report_job(job_id, organization_id)
        if job:
            current_status = job.get('status', '').upper()
            if current_status in ['SUCCEEDED', 'FAILED']:
                logging.info(
                    f"[LockManager] Job {job_id} already completed with status {current_status}"
                )
                return False, current_status, None

        # Try to acquire lock
        lock_acquired, lease_client = await acquire_job_lock(job_id, max_wait=poll_interval)
        if lock_acquired:
            return True, 'RUNNING', lease_client

        waited += poll_interval

    return False, 'TIMEOUT', None


async def process_report_job(
    job_id: str,
    organization_id: str,
    dequeue_count: int = 1
) -> None:
    """
    Process a report generation job.
    
    Args:
        job_id: Unique identifier for the report job
        organization_id: Organization requesting the report
        dequeue_count: Number of times this message has been dequeued
    
    Raises:
        Exception: For transient errors that should trigger retry
    """
    logging.info(
        f"[ReportWorker] Processing job {job_id} for org {organization_id} "
        f"(dequeue_count: {dequeue_count})"
    )
    
    # Fetch job from Cosmos DB
    job = get_report_job(job_id, organization_id)
    if not job:
        error_msg = f"Job {job_id} not found in database for organization {organization_id}"
        logging.error(f"[ReportWorker] {error_msg}")
        raise ValueError(error_msg)
        
    logging.info(f"[ReportWorker] Retrieved job {job_id}: {job.get('report_key', 'unknown')}")
    
    # Check job status for idempotency
    current_status = job.get('status', '').upper()
    if current_status in ['SUCCEEDED', 'FAILED']:
        logging.info(
            f"[ReportWorker] Job {job_id} already completed with status {current_status}, skipping processing"
        )
        return
    # Update job status to RUNNING
    if current_status == 'QUEUED':
        success = update_report_job_status(job_id, organization_id, 'RUNNING')
        if not success:
            logging.error(f"[ReportWorker] Failed to update job {job_id} status to RUNNING")
            raise Exception(f"Failed to update job {job_id} status to RUNNING")
        logging.info(f"[ReportWorker] Updated job {job_id} status to RUNNING")
    else:
        logging.info(f"[ReportWorker] Job {job_id} is already RUNNING, proceeding with processing")
    
    # Get report generator
    report_key = job.get('report_key',"")
    report_name = job.get('report_name',"Report")
    if not report_key:
        logging.error(f"[ReportWorker] Job {job_id} missing report_key")
        update_report_job_status(job_id, organization_id, 'FAILED', error_payload={
            "error_type": "deterministic",
            "error_message": "Missing report_key",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return
        
    generator = get_generator(report_key)
    if not generator:
        logging.error(f"[ReportWorker] No generator found for report_key: {report_key}")
        update_report_job_status(job_id, organization_id, 'FAILED', error_payload={
            "error_type": "deterministic",
            "error_message": f"No generator found for report_key: {report_key}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return
        
    logging.info(f"[ReportWorker] Found generator for {report_key}: {generator.__class__.__name__}")

    # Generate the report
    parameters = job.get('params', {})

    # Acquire distributed lock to prevent concurrent generation (prevents 429 rate limit errors)
    logging.info(f"[ReportWorker] Acquiring lock for job {job_id}")
    lock_acquired, job_status, lease_client = await wait_for_job_completion_or_lock(
        job_id, organization_id, max_wait=1800  # Wait up to 30 minutes
    )

    # Check if job was completed by another worker
    if job_status in ['SUCCEEDED', 'FAILED']:
        logging.info(f"[ReportWorker] Job {job_id} already completed with status {job_status}")
        return

    # Check if we got the lock
    if not lock_acquired:
        error_msg = f"Could not acquire lock for job {job_id} after 30 minutes"
        logging.error(f"[ReportWorker] {error_msg}")
        update_report_job_status(job_id, organization_id, 'FAILED', error_payload={
            "error_type": "transient",
            "error_message": error_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise TimeoutError(error_msg)

    try:
        # Generate markdown content using the appropriate generator
        logging.info(f"[ReportWorker] Generating markdown content for job {job_id}")
        markdown_content = generator.generate(job_id, organization_id, parameters)
        logging.info(f"[ReportWorker] Generated markdown content ({len(markdown_content)} chars) for job {job_id}")
        
        markdown_content = markdown_content.replace("---", "")
        # Convert markdown to PDF
        logging.info(f"[ReportWorker] Converting markdown to PDF for job {job_id}")
        pdf_content_dict = {"content": markdown_content}
        pdf_bytes = dict_to_pdf(pdf_content_dict)
        logging.info(f"[ReportWorker] Generated PDF ({len(pdf_bytes)} bytes) for job {job_id}")
        
        # Store PDF in Azure Blob Storage
        result_metadata = await _store_pdf_in_blob(
            pdf_bytes=pdf_bytes,
            job_id=job_id,
            organization_id=organization_id,
            report_key=report_key,
            report_name=report_name,
            parameters=parameters
        )
        
        logging.info(f"[ReportWorker] Successfully generated report for job {job_id}")
        
        # Update job status to SUCCEEDED
        success = update_report_job_status(
            job_id, 
            organization_id, 
            'SUCCEEDED', 
            result_metadata=result_metadata
        )
        
        if not success:
            logging.error(f"[ReportWorker] Failed to update job {job_id} status to SUCCEEDED")
            raise Exception(f"Failed to update job {job_id} status to SUCCEEDED")
            
        logging.info(
            f"[ReportWorker] Completed job {job_id} successfully "
            f"(dequeue_count: {dequeue_count})"
        )
        
    except NotImplementedError as e:
        logging.error(f"[ReportWorker] Report generator not implemented: {str(e)}")
        update_report_job_status(job_id, organization_id, 'FAILED', error_payload={
            "error_type": "deterministic",
            "error_message": f"Report generator not implemented: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return

    except Exception as e:
        logging.error(f"[ReportWorker] Error generating report: {str(e)}")
        update_report_job_status(job_id, organization_id, 'FAILED', error_payload={
            "error_type": "transient",
            "error_message": str(e),
            "dequeue_count": dequeue_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise  # Re-raise to trigger Azure Storage Queue retry

    finally:
        # Always release the lock when we're done
        await release_job_lock(lease_client, job_id)

def generate_file_name(report_key: str, parameters: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if report_key == "brand_analysis":
        brand_name = parameters.get("brand_focus", "")
        return f"Brand_Analysis_{brand_name}_{timestamp}.pdf"
    elif report_key == "competitor_analysis":
        return f"Competitor_Analysis_{timestamp}.pdf"
    elif report_key == "product_analysis":
        category_name = "_".join([x.get("category", "").replace(" ", "_") for x in parameters.get("categories", [])])
        return f"Product_Analysis_{category_name}_{timestamp}.pdf"
    return f"Report_{report_key}_{timestamp}.pdf"

async def _store_pdf_in_blob(
    pdf_bytes: bytes,
    job_id: str,
    organization_id: str,
    report_key: str,
    parameters: Dict[str, Any],
    report_name: str
) -> Dict[str, Any]:
    """
    Store PDF in Azure Blob Storage and return metadata.
    
    Args:
        pdf_bytes: PDF content as bytes
        job_id: Report job ID
        organization_id: Organization ID
        report_key: Report type key
        
    Returns:
        Dict containing blob metadata and URLs
    """
    blob_service_client = await get_blob_service_client()
    container_name = "documents"
    
    # Create blob name with organization structure
    
    file_name = generate_file_name(report_key, parameters)
    blob_name = f"organization_files/{organization_id}/{file_name}"
    
    # Prepare metadata
    blob_metadata = {
        "organization_id": organization_id,
        "report_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_key": report_key
    }
    
    logging.info(f"[ReportWorker] Storing PDF in blob: {blob_name}")
    
    # Get container client and ensure container exists
    container_client = blob_service_client.get_container_client(container_name)
    try:
        await container_client.get_container_properties()
    except Exception:
        try:
            await container_client.create_container()
            logging.info(f"[ReportWorker] Created container: {container_name}")
        except Exception as e:
            # Container might already exist (race condition)
            logging.info(f"[ReportWorker] Container creation result: {str(e)}")
    
    # Upload PDF to blob storage
    blob_client = container_client.get_blob_client(blob_name)
    await blob_client.upload_blob(
        pdf_bytes,
        overwrite=True,
        metadata=blob_metadata,
    )
    
    blob_url = blob_client.url
    logging.info(f"[ReportWorker] Successfully stored PDF at: {blob_url}")
    
    # Prepare result metadata
    result_metadata = {
        "blob_url": blob_url,
        "file_name": file_name,
        "file_size": len(pdf_bytes),
        "content_type": "application/pdf",
        "blob_name": blob_name,
        "container_name": container_name,
        "metadata": blob_metadata
    }
    
    return result_metadata


def extract_message_metadata(msg) -> Tuple[Optional[str], Optional[str], Optional[str], int, str]:
    """
    Extract metadata and required fields from queue message.
    
    Args:
        msg: Azure Function Queue message
        
    Returns:
        Tuple of (job_id, organization_id, dequeue_count, message_id)
        Returns (None, None, None, 1, "unknown") if parsing fails
    """
    try:
        # Extract message metadata
        dequeue_count = msg.dequeue_count or 1
        message_id = msg.id or "unknown"
        
        logging.info(
            f"[ReportWorker] Received message {message_id} "
            f"(dequeue_count: {dequeue_count})"
        )
        
        # Parse message body
        try:
            # Handle both string and bytes message body
            if hasattr(msg, 'get_body'):
                raw_body = msg.get_body()
                if isinstance(raw_body, bytes):
                    message_body = raw_body.decode('utf-8')
                else:
                    message_body = str(raw_body)
            else:
                # Fallback for direct string access
                message_body = str(msg)
                
            payload = json.loads(message_body)
            logging.info(f"[ReportWorker] Parsed message body: {message_body}")
            
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logging.error(f"[ReportWorker] Invalid message format: {str(e)}")
            logging.error(f"[ReportWorker] Raw message: {repr(msg.get_body() if hasattr(msg, 'get_body') else msg)}")
            return None, None, None, dequeue_count, message_id
            
        # Extract required fields
        job_id = payload.get('job_id')
        organization_id = payload.get('organization_id')
        
        if not all([job_id, organization_id]):
            logging.error(f"[ReportWorker] Missing required fields in payload: {payload}")
            return None, None, None, dequeue_count, message_id
            
        return job_id, organization_id, dequeue_count, message_id
        
    except Exception as e:
        logging.error(f"[ReportWorker] Error extracting message metadata: {str(e)}")
        return None, None, None, 1, "unknown"
