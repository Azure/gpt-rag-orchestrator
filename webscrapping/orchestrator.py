"""
Web Scraping Orchestrator

This module provides the main entry points and orchestration logic for
the web scraping functionality. It coordinates between all the individual
components and provides both standalone and Azure Functions interfaces.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import azure.functions as func

from .utils import (
    generate_request_id,
    parse_request_body,
    create_error_response,
    create_success_response,
    calculate_duration,
    format_timestamp,
    validate_urls,
    log_processing_summary,
)
from .blob_manager import create_crawler_manager_from_env
from .parallel_processor import process_urls_parallel

_logger = logging.getLogger(__name__)


def scrape_urls_standalone(urls: List[str], request_id: str = None, organization_id: str = None) -> Dict[str, Any]:
    """
    Standalone function to scrape URLs without Azure Functions dependency.
    
    Args:
        urls: List of URLs to scrape
        request_id: Optional request identifier
        organization_id: Optional organization identifier for metadata
        
    Returns:
        Dictionary with scraping results
    """
    start_time = datetime.now(timezone.utc)
    request_id = request_id or generate_request_id()
    
    _logger.info(
        "[Orchestrator] Starting standalone scraping - request_id: %s, url_count: %d, organization_id: %s",
        request_id,
        len(urls),
        organization_id or "None",
    )

    try:
        # Validate URLs
        validated_urls = validate_urls(urls)
        
        # Initialize blob storage manager (optional)
        crawler_manager = create_crawler_manager_from_env(request_id)
        blob_storage_enabled = crawler_manager is not None
        
        if not blob_storage_enabled:
            _logger.info(
                "[Orchestrator] Proceeding without blob storage - request_id: %s",
                request_id,
            )

        # Process URLs in parallel
        results, blob_storage_results = process_urls_parallel(
            validated_urls, crawler_manager, request_id, organization_id=organization_id
        )

        # Calculate metrics
        end_time = datetime.now(timezone.utc)
        duration = calculate_duration(start_time, end_time)
        
        successful_scrapes = len([r for r in results if r["status"] == "success"])
        failed_scrapes = len([r for r in results if r["status"] == "error"])
        successful_blob_uploads = len(
            [r for r in blob_storage_results if r["status"] == "success"]
        )
        skipped_blob_uploads = len(
            [r for r in blob_storage_results if r["status"] == "skipped"]
        )

        # Log summary
        log_processing_summary(
            request_id, len(validated_urls), successful_scrapes, 
            failed_scrapes, duration, blob_storage_enabled
        )

        # Save crawler summary if available
        crawler_summary_metrics = None
        if crawler_manager:
            try:
                crawler_summary_metrics = crawler_manager.save_summary(start_time)
            except Exception as e:
                _logger.warning(
                    "[Orchestrator] Failed to save crawler summary - request_id: %s, error: %s",
                    request_id,
                    str(e),
                )

        return {
            "status": "completed",
            "message": f"Scraped {len(validated_urls)} URLs in parallel"
            + (
                f" and uploaded to blob storage"
                if blob_storage_enabled
                else " (blob storage not configured)"
            ),
            "request_id": request_id,
            "duration_seconds": round(duration, 2),
            "completed_at": format_timestamp(end_time),
            "blob_storage_enabled": blob_storage_enabled,
            "summary": {
                "total_urls": len(validated_urls),
                "successful_scrapes": successful_scrapes,
                "failed_scrapes": failed_scrapes,
                "successful_blob_uploads": successful_blob_uploads,
                "skipped_blob_uploads": skipped_blob_uploads,
            },
            "results": results,
            "blob_storage_results": blob_storage_results,
            "crawler_summary": crawler_summary_metrics,
        }

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = calculate_duration(start_time, end_time)

        _logger.error(
            "[Orchestrator] Standalone scraping failed - request_id: %s, error: %s",
            request_id,
            str(e),
            exc_info=True,
        )

        return {
            "status": "error",
            "message": f"Failed to execute scraping: {str(e)}",
            "request_id": request_id,
            "duration_seconds": round(duration, 2),
            "error_at": format_timestamp(end_time),
        }
