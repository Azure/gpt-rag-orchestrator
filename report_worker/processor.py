"""
Report Processing Logic

This module contains the main logic for processing report generation jobs.
Separated from function_app.py for better organization and readability.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from shared.util import get_report_job, update_report_job_status
from shared.markdown_to_pdf import dict_to_pdf
from shared.blob_client_async import get_blob_service_client
from .registry import get_generator


async def process_report_job(
    job_id: str,
    organization_id: str, 
    correlation_id: Optional[str],
    dequeue_count: int = 1
) -> None:
    """
    Process a report generation job.
    
    Args:
        job_id: Unique identifier for the report job
        organization_id: Organization requesting the report
        correlation_id: Correlation ID for tracking
        dequeue_count: Number of times this message has been dequeued
    
    Raises:
        Exception: For transient errors that should trigger retry
    """
    logging.info(
        f"[ReportWorker] Processing job {job_id} for org {organization_id} "
        f"(correlation: {correlation_id}, dequeue_count: {dequeue_count})"
    )
    
    # Fetch job from Cosmos DB
    job = get_report_job(job_id, organization_id)
    if not job:
        logging.error(f"[ReportWorker] Job {job_id} not found in database")
        return
        
    logging.info(f"[ReportWorker] Retrieved job {job_id}: {job.get('report_key', 'unknown')}")
    
    # Check job status for idempotency
    current_status = job.get('status', '').upper()
    if current_status != 'QUEUED':
        logging.info(
            f"[ReportWorker] Job {job_id} status is {current_status}, skipping processing"
        )
        return
        
    # Update job status to RUNNING
    success = update_report_job_status(job_id, organization_id, 'RUNNING')
    if not success:
        logging.error(f"[ReportWorker] Failed to update job {job_id} status to RUNNING")
        raise Exception(f"Failed to update job {job_id} status to RUNNING")
        
    logging.info(f"[ReportWorker] Updated job {job_id} status to RUNNING")
    
    # Get report generator
    report_key = job.get('report_key')
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
    parameters = job.get('parameters', {})
    try:
        # Generate markdown content using the appropriate generator
        logging.info(f"[ReportWorker] Generating markdown content for job {job_id}")
        markdown_content = generator.generate(job_id, organization_id, parameters)
        logging.info(f"[ReportWorker] Generated markdown content ({len(markdown_content)} chars) for job {job_id}")
        
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
            correlation_id=correlation_id
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
            f"(correlation: {correlation_id})"
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
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise  # Re-raise to trigger Azure Storage Queue retry


async def _store_pdf_in_blob(
    pdf_bytes: bytes,
    job_id: str,
    organization_id: str,
    report_key: str,
    correlation_id: Optional[str]
) -> Dict[str, Any]:
    """
    Store PDF in Azure Blob Storage and return metadata.
    
    Args:
        pdf_bytes: PDF content as bytes
        job_id: Report job ID
        organization_id: Organization ID
        report_key: Report type key
        correlation_id: Correlation ID for tracking
        
    Returns:
        Dict containing blob metadata and URLs
    """
    blob_service_client = await get_blob_service_client()
    container_name = "documents"
    
    # Create blob name with organization structure
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = f"report_{job_id}_{timestamp}.pdf"
    blob_name = f"organization_files/{organization_id}/{file_name}"
    
    # Prepare metadata
    blob_metadata = {
        "organization_id": organization_id,
        "report_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_key": report_key,
        "correlation_id": correlation_id or ""
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
        content_settings={
            "content_type": "application/pdf"
        }
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
        Tuple of (job_id, organization_id, correlation_id, dequeue_count, message_id)
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
        correlation_id = payload.get('correlation_id')
        
        if not all([job_id, organization_id, correlation_id]):
            logging.error(f"[ReportWorker] Missing required fields in payload: {payload}")
            return None, None, None, dequeue_count, message_id
            
        return job_id, organization_id, correlation_id, dequeue_count, message_id
        
    except Exception as e:
        logging.error(f"[ReportWorker] Error extracting message metadata: {str(e)}")
        return None, None, None, 1, "unknown"
