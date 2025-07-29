"""
Utility Functions for Web Scraping

This module contains common utility functions used across the webscrapping package.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import azure.functions as func

_logger = logging.getLogger(__name__)


def generate_request_id(prefix: str = "scrape") -> str:
    """
    Generate a unique request ID with timestamp.
    
    Args:
        prefix: Prefix for the request ID
        
    Returns:
        Formatted request ID string
    """
    timestamp = datetime.now(timezone.utc)
    return f"{prefix}-{timestamp.strftime('%Y%m%d-%H%M%S')}"


def parse_request_body(req: func.HttpRequest, request_id: str) -> List[str]:
    """
    Parse and validate URLs from HTTP request body.
    
    Args:
        req: Azure Functions HTTP request
        request_id: Request identifier for logging
        
    Returns:
        List of URLs to process
        
    Raises:
        ValueError: If request body is invalid
    """
    try:
        req_body = req.get_json()
        if not req_body or "urls" not in req_body:
            raise ValueError("Request body must contain 'urls' array")

        urls = req_body["urls"]
        if not isinstance(urls, list) or len(urls) == 0:
            raise ValueError("urls must be a non-empty array")

        _logger.info(
            "[Utils] Parsed %d URLs from request - request_id: %s", 
            len(urls), 
            request_id
        )
        return urls

    except ValueError as e:
        _logger.error(
            "[Utils] Request validation failed - request_id: %s, error: %s", 
            request_id, 
            str(e)
        )
        raise


def create_error_response(
    message: str, 
    request_id: str, 
    status_code: int = 400,
    additional_data: Dict[str, Any] = None
) -> func.HttpResponse:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        request_id: Request identifier
        status_code: HTTP status code
        additional_data: Optional additional data to include
        
    Returns:
        Azure Functions HTTP response
    """
    response_data = {
        "status": "error",
        "message": message,
        "request_id": request_id,
    }
    
    if additional_data:
        response_data.update(additional_data)

    return func.HttpResponse(
        json.dumps(response_data),
        headers={"Content-Type": "application/json"},
        status_code=status_code,
    )


def create_success_response(
    data: Dict[str, Any], 
    status_code: int = 200
) -> func.HttpResponse:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Azure Functions HTTP response
    """
    return func.HttpResponse(
        json.dumps(data),
        headers={"Content-Type": "application/json"},
        status_code=status_code,
    )


def calculate_duration(start_time: datetime, end_time: datetime = None) -> float:
    """
    Calculate duration between two timestamps.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp (defaults to current time)
        
    Returns:
        Duration in seconds
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    
    return (end_time - start_time).total_seconds()


def format_timestamp(timestamp: datetime = None) -> str:
    """
    Format timestamp for consistent API responses.
    
    Args:
        timestamp: Timestamp to format (defaults to current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_urls(urls: List[str]) -> List[str]:
    """
    Basic validation of URL format.
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        List of validated URLs
        
    Raises:
        ValueError: If any URL is invalid
    """
    from urllib.parse import urlparse
    
    validated_urls = []
    for url in urls:
        if not isinstance(url, str):
            raise ValueError(f"URL must be a string: {url}")
        
        if not url.strip():
            raise ValueError("URL cannot be empty")
        
        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        validated_urls.append(url.strip())
    
    return validated_urls


def log_processing_summary(
    request_id: str,
    total_urls: int,
    successful_scrapes: int,
    failed_scrapes: int,
    duration: float,
):
    """
    Log a standardized processing summary.
    
    Args:
        request_id: Request identifier
        total_urls: Total number of URLs processed
        successful_scrapes: Number of successful scrapes
        failed_scrapes: Number of failed scrapes
        duration: Processing duration in seconds
    """
    success_rate = (successful_scrapes / total_urls) * 100 if total_urls > 0 else 0
    
    _logger.info(
        "[Utils] Processing completed - request_id: %s, "
        "total: %d, success: %d (%.1f%%), failed: %d, duration: %.2fs",
        request_id,
        total_urls,
        successful_scrapes,
        success_rate,
        failed_scrapes,
        duration,
    )
