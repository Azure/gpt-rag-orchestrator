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
from .scraper import WebScraper
from .config import CrawlerConfig, Document, Html, HtmlParserConfig

_logger = logging.getLogger(__name__)


def scrape_single_url(url: str, request_id: str = None, organization_id: str = None) -> Dict[str, Any]:
    """
    Scrape a single URL without Azure Functions dependency.
    
    Args:
        url: Single URL to scrape
        request_id: Optional request identifier
        organization_id: Optional organization identifier for metadata
        
    Returns:
        Dictionary with scraping results
    """
    start_time = datetime.now(timezone.utc)
    request_id = request_id or generate_request_id()
    
    _logger.info(
        "[Orchestrator] Starting single URL scraping - request_id: %s, url: %s, organization_id: %s",
        request_id,
        url,
        organization_id or "None",
    )

    try:
        # Validate single URL
        validated_urls = validate_urls([url])
        validated_url = validated_urls[0]
        
        # Initialize blob storage manager
        crawler_manager = create_crawler_manager_from_env(request_id)

        # Create scraper configuration
        config = CrawlerConfig(
            documents=Document(urls=[], storage=None),
            html=Html(
                striptags=True,
                parser=HtmlParserConfig(
                    ignored_classes=["nav", "footer", "sidebar", "ads"]
                ),
            ),
        )
        
        scraper = WebScraper(config, _logger)
        
        # Scrape the single URL
        _logger.info(
            "[Orchestrator] Scraping URL: %s - request_id: %s",
            validated_url, request_id
        )
        
        scrape_result = scraper.scrape_page(validated_url, request_id)
        
        # Handle blob storage for all scraping attempts
        blob_storage_result = None
        if scrape_result["status"] == "success":
            # Format content for blob storage
            blob_data = WebScraper.format_content_for_blob_storage(
                scrape_result, request_id, organization_id
            )
            
            if crawler_manager and blob_data:
                try:
                    # Upload to blob storage
                    text_content_type = "text/plain"
                    contents = [blob_data]
                    
                    crawler_manager.store_in_blob(
                        url=validated_url, contents=contents, content_type=text_content_type
                    )
                    
                    # Update crawler summary
                    crawler_manager.crawler_summary.add_success(validated_url)
                    
                    blob_storage_result = {
                        "status": "success",
                        "message": "Successfully uploaded to blob storage",
                        "blob_path": crawler_manager._get_blob_path(validated_url, 0, text_content_type),
                    }
                    
                except Exception as blob_error:
                    crawler_manager.crawler_summary.add_failure(validated_url)
                    blob_storage_result = {
                        "status": "error",
                        "error": f"Blob storage upload failed: {str(blob_error)}",
                        "blob_path": None,
                    }
                    _logger.error(
                        "[Orchestrator] Blob storage failed for URL: %s - request_id: %s, error: %s",
                        validated_url, request_id, str(blob_error)
                    )
            elif not crawler_manager:
                # Storage not configured
                blob_storage_result = {
                    "status": "not_configured", 
                    "message": "Blob storage not configured - missing Azure storage environment variables",
                    "blob_path": None,
                }
            else:
                # Failed to format content
                blob_storage_result = {
                    "status": "error",
                    "error": "Failed to format content for blob storage",
                    "blob_path": None,
                }
        else:
            # Scraping failed
            if crawler_manager:
                crawler_manager.crawler_summary.add_failure(validated_url)
            blob_storage_result = {
                "status": "error",
                "error": f"Scraping failed: {scrape_result.get('error', 'Unknown error')}",
                "blob_path": None,
            }

        # Calculate metrics
        end_time = datetime.now(timezone.utc)
        duration = calculate_duration(start_time, end_time)
        
        success = scrape_result["status"] == "success"

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

        _logger.info(
            "[Orchestrator] Single URL scraping completed - request_id: %s, success: %s, duration: %.2fs",
            request_id, success, duration
        )

        # Generate message based on blob storage result
        if blob_storage_result.get("status") == "success":
            message = "Scraped single URL and uploaded to blob storage"
        elif blob_storage_result.get("status") == "not_configured":
            message = "Scraped single URL (blob storage not configured)"
        elif success:
            message = "Scraped single URL but blob storage failed"
        else:
            message = "Failed to scrape URL"
        
        return {
            "status": "completed" if success else "failed",
            "message": message,
            "url": validated_url,
            "request_id": request_id,
            "response_time": round(duration, 2),
            "completed_at": format_timestamp(end_time),
            "results": [scrape_result] if success else [],
            "blob_storage_result": blob_storage_result,
            "crawler_summary": crawler_summary_metrics,
        }

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = calculate_duration(start_time, end_time)

        _logger.error(
            "[Orchestrator] Single URL scraping failed - request_id: %s, error: %s",
            request_id,
            str(e),
            exc_info=True,
        )

        return {
            "status": "error",
            "message": f"Failed to scrape URL: {str(e)}",
            "url": url,
            "request_id": request_id,
            "response_time": round(duration, 2),
            "error_at": format_timestamp(end_time),
        }

