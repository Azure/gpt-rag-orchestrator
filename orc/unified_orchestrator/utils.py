"""
Utility functions for the unified orchestrator.

This module provides common utility functions used across the orchestrator,
including logging helpers, artifact transformations, and progress message formatting.
"""

import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def log_info(message: str, **kwargs):
    """
    Log message with both logger and print for visibility in Azure Functions.
    
    This dual logging approach ensures messages are visible both in structured
    logs and in the Azure Functions console output.
    
    Args:
        message: The message to log
        **kwargs: Additional keyword arguments to pass to logger.info
    """
    logger.info(message, **kwargs)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{timestamp}] {message}")


def transform_artifacts_to_images(artifacts: List[Dict]) -> List[Dict]:
    """
    Transform streaming artifacts to images_processed format.
    
    Converts artifact dictionaries from data analyst tool output into a format
    suitable for image rendering in the UI.
    
    Args:
        artifacts: List of artifact dictionaries with keys:
            - blob_path: Path to the blob storage location
            - filename: Name of the file
            - size: Size in bytes (optional)
    
    Returns:
        List of dictionaries formatted for image processing with keys:
            - file_id: Blob path identifier
            - filename: Original filename
            - size_bytes: File size in bytes
            - content_type: MIME type (always "image/png")
    """
    return [
        {
            "file_id": art.get("blob_path", ""),
            "filename": art.get("filename", "unknown"),
            "size_bytes": art.get("size", 0),
            "content_type": "image/png",
        }
        for art in artifacts
    ]


def transform_artifacts_to_blobs(artifacts: List[Dict]) -> List[Dict]:
    """
    Transform streaming artifacts to blob_urls format.
    
    Converts artifact dictionaries from data analyst tool output into a format
    suitable for blob URL references in the UI.
    
    Args:
        artifacts: List of artifact dictionaries with keys:
            - filename: Name of the file
            - blob_url: Public URL to access the blob
            - blob_path: Storage path to the blob
    
    Returns:
        List of dictionaries formatted for blob URL processing with keys:
            - filename: Original filename
            - blob_url: Public URL for accessing the blob
            - blob_path: Storage path identifier
    """
    return [
        {
            "filename": art.get("filename", "unknown"),
            "blob_url": art.get("blob_url", ""),
            "blob_path": art.get("blob_path", ""),
        }
        for art in artifacts
    ]


def get_tool_progress_message(tool_name: str, stage: str) -> str:
    """
    Get tool-specific progress message for UI.
    
    Provides user-friendly progress messages for different MCP tools at various
    stages of execution. Used to display meaningful status updates in the UI.
    
    Args:
        tool_name: Name of the MCP tool (e.g., "agentic_search", "data_analyst",
                   "web_fetch", "document_chat")
        stage: Stage of execution ("planning" or "executing")
    
    Returns:
        User-friendly progress message string. Returns a generic message if
        the tool_name or stage is not recognized.
    
    Examples:
        >>> get_tool_progress_message("agentic_search", "executing")
        "Searching your knowledge base..."
        
        >>> get_tool_progress_message("data_analyst", "planning")
        "Planning data analysis..."
    """
    tool_messages = {
        "agentic_search": {
            "planning": "Planning knowledge base search...",
            "executing": "Searching your knowledge base...",
        },
        "data_analyst": {
            "planning": "Planning data analysis...",
            "executing": "Analyzing your data...",
        },
        "web_fetch": {
            "planning": "Planning web content fetch...",
            "executing": "Fetching web content...",
        },
        "document_chat": {
            "planning": "Preparing document analysis...",
            "executing": "Reading your documents...",
        },
    }

    return tool_messages.get(tool_name, {}).get(
        stage, f"{stage.capitalize()} tools..."
    )
