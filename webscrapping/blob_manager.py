"""
Blob Storage Manager Module

This module contains functionality for managing Azure Blob Storage operations
including document storage, retrieval, and metadata management.
"""

import os
import hashlib
import logging
from urllib.parse import urlparse
from typing import List, Optional
from datetime import datetime, timezone
from azure.core.exceptions import ResourceNotFoundError

# Import BlobHandler from shared utilities
from shared.blob_utils import BlobHandler
from .config import CrawlerConfig
from .metrics import CrawlerSummary

_logger = logging.getLogger(__name__)


class WebCrawlerManager:
    """Manages web crawling operations and blob storage."""

    def __init__(self, config_name: str, configuration: CrawlerConfig, logger=None):
        self.config_name = config_name
        self.configuration = configuration
        self._logger = logger or _logger

        # Initialize blob storage if configured
        self._initialize_documents_store()

        # Initialize crawler summary
        self.crawler_summary = CrawlerSummary(config_name=self.config_name)

    def _initialize_documents_store(self):
        """Initialize the documents storage configuration."""
        if self.configuration.documents and self.configuration.documents.storage:
            self.documents_storage_account_url = (
                self.configuration.documents.storage.account
            )
            self.documents_container_name = (
                self.configuration.documents.storage.container
            )

            # Ensure the container exists
            BlobHandler.ensure_container_exists(
                storage_account_url=self.documents_storage_account_url,
                container_name=self.documents_container_name,
            )
        else:
            self.documents_storage_account_url = None
            self.documents_container_name = None

    def store_in_blob(self, url: str, contents: list, content_type: str):
        """Store content chunks in blob storage."""
        for _chunk_num, _content in enumerate(contents):
            self._store_in_blob(
                url=url,
                chunk_num=_chunk_num,
                content=_content["content"],
                metadata=_content["metadata"],
                content_type=content_type,
            )

    def _store_in_blob(
        self,
        url: str,
        chunk_num: int,
        content: bytes,
        metadata: dict,
        content_type: str,
    ):
        """Store individual content chunk in blob storage."""
        _blob_path = self._get_blob_path(url, chunk_num, content_type)
        self._logger.debug("Blob path is: %s", _blob_path)
        _checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()

        _is_same_content = self._is_same_content(_checksum, _blob_path)
        if not _is_same_content:
            if _is_same_content is None:
                self._logger.debug(
                    "Content is not present for %s. Uploading blob...",
                    _blob_path,
                )
                self.crawler_summary.add_new_page(url)
            else:
                self._logger.debug(
                    "Content is changed for %s. Uploading blob...",
                    _blob_path,
                )
                self.crawler_summary.add_updated_page(url)

            _metadata = {
                "source_address": url,
                "checksum": _checksum,
            }
            _metadata.update(metadata)

            BlobHandler.upload(
                storage_account_url=self.documents_storage_account_url,
                container_name=self.documents_container_name,
                blob_path=_blob_path,
                content=content,
                metadata=_metadata,
                overwrite=True,
            )
        else:
            self._logger.debug("Content is unchanged for %s.", _blob_path)

    def _get_blob_path(self, url: str, chunk_num: int, content_type: str) -> str:
        """Generate blob path for the given URL and chunk."""
        _parsed_url = urlparse(url)
        _blob_path = f"{_parsed_url.netloc}{_parsed_url.path}"
        _base_name = os.path.basename(_blob_path)
        _file_name, _file_extension = os.path.splitext(_base_name)

        # Force .txt extension for text/plain content
        if "text/plain" in content_type:
            if len(_file_name) == 0:
                _file_name = "default"
            _file_extension = ".txt"
        elif len(_file_name) == 0:
            if "application/pdf" in content_type:
                _file_name = "default"
                _file_extension = ".pdf"
            elif "text/html" in content_type:
                _file_name = "default"
                _file_extension = (
                    ".txt"  # HTML content becomes plain text after processing
                )
            else:
                _file_name = "default"
                _file_extension = ".txt"

        _blob_path = os.path.join(
            f"WebCrawler_Output/{_blob_path}",
            f"{_file_name}_{chunk_num}{_file_extension}",
        )
        return _blob_path

    def _is_same_content(self, checksum: str, blob_path: str):
        """Check if the content is the same as what's already stored."""
        try:
            _blob_data = BlobHandler.download(
                storage_account_url=self.documents_storage_account_url,
                container_name=self.documents_container_name,
                blob_path=blob_path,
            )
            _blob_checksum = hashlib.md5(
                _blob_data,
                usedforsecurity=False,
            ).hexdigest()

            self._logger.debug("Incoming chunk checksum is  %s", checksum)
            self._logger.debug("Stored chunk checksum is  %s", _blob_checksum)

            return checksum == _blob_checksum
        except ResourceNotFoundError:
            return None

    def save_summary(self, run_start_time: datetime):
        """Save the crawler summary and return metrics."""
        if self.crawler_summary.end_time is None:
            self.crawler_summary.end_time = datetime.now(timezone.utc)

        return self.crawler_summary.get_metrics()

    def is_blob_storage_available(self) -> bool:
        """Check if blob storage is properly configured and available."""
        return (
            self.documents_storage_account_url is not None
            and self.documents_container_name is not None
        )


def create_crawler_manager_from_env(request_id: str) -> Optional[WebCrawlerManager]:
    """
    Create a WebCrawlerManager instance from environment variables.
    
    Args:
        request_id: Unique identifier for the request
        
    Returns:
        WebCrawlerManager instance or None if not configured
    """
    try:
        # Check if blob storage is configured via environment variables
        AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        documents_container = "documents"

        if not AZURE_STORAGE_ACCOUNT_URL:
            _logger.warning(
                "[BlobManager] No blob storage configuration found - request_id: %s. Skipping blob storage.",
                request_id,
            )
            return None

        _logger.info(
            "[BlobManager] Initializing WebCrawlerManager - request_id: %s, storage_url: %s, container: %s",
            request_id,
            AZURE_STORAGE_ACCOUNT_URL,
            documents_container,
        )

        # Import configuration models here to avoid circular imports
        from .config import CrawlerConfig, Document, Storage, Html, HtmlParserConfig

        # Create a minimal configuration object
        minimal_config = CrawlerConfig(
            documents=Document(
                urls=[],  # We don't need start URLs for individual page scraping
                storage=Storage(
                    account=AZURE_STORAGE_ACCOUNT_URL, container=documents_container
                ),
            ),
            html=Html(
                striptags=True,
                parser=HtmlParserConfig(
                    ignored_classes=["nav", "footer", "sidebar", "ads"]
                ),
            ),
        )

        # Create WebCrawlerManager with the minimal config
        crawler_manager = WebCrawlerManager(
            config_name=f"scrape_pages_{request_id}",
            configuration=minimal_config,
            logger=_logger,
        )

        return crawler_manager

    except Exception as e:
        _logger.error(
            "[BlobManager] Failed to initialize WebCrawlerManager - request_id: %s, error: %s",
            request_id,
            str(e),
            exc_info=True,
        )
        return None 