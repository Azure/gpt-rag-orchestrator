"""
Blob Storage Manager Module

This module contains functionality for managing Azure Blob Storage operations
including document storage, retrieval, and metadata management.
"""

import os
import hashlib
import logging
from urllib.parse import urlparse
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from azure.core.exceptions import ResourceNotFoundError

# Import BlobHandler from shared utilities
from shared.blob_utils import BlobHandler
from .config import CrawlerConfig
from .metrics import CrawlerSummary

_logger = logging.getLogger(__name__)


class WebCrawlerManager:
    """Manages web crawling operations and blob storage."""

    def __init__(self, configuration: CrawlerConfig, logger=None):
        self.configuration = configuration
        self._logger = logger or _logger

        # Initialize blob storage if configured
        self._initialize_documents_store()

        # Initialize crawler summary
        self.crawler_summary = CrawlerSummary()

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
                self.crawler_summary.add_success(url)
            else:
                self._logger.debug(
                    "Content is changed for %s. Uploading blob...",
                    _blob_path,
                )
                self.crawler_summary.add_success(url)

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

    def store_multipage_results_in_blob(self, formatted_pages: List[Dict[str, Any]], content_type: str = "text/plain"):
        """
        Store multiple pages from multipage crawl results in blob storage.
        
        Args:
            formatted_pages: List of formatted page data from format_multipage_content_for_blob_storage
            content_type: Content type for blob storage (default: text/plain)
        """
        successful_uploads = []
        failed_uploads = []
        duplicate_pages = []
        
        for page_data in formatted_pages:
            try:
                page_url = page_data.get('url', '')
                page_content = page_data.get('content', b'')
                page_metadata = page_data.get('metadata', {})
                
                # Generate blob path for this page
                blob_path = self._get_blob_path_for_multipage(page_url, page_metadata, content_type)
                
                # Calculate checksum for deduplication
                import hashlib
                checksum = hashlib.md5(page_content, usedforsecurity=False).hexdigest()
                
                # Check if content with this URL already exists anywhere in storage
                content_already_exists = self._content_exists_by_checksum(checksum, page_url)
                
                if not content_already_exists:
                    # Content is new or this is the first occurrence - always store it
                    page_metadata["checksum"] = checksum
                    page_metadata["source_address"] = page_url
                    
                    # Check if this exact path existed before (for proper status)
                    path_existed = self._blob_path_exists(blob_path)
                    
                    # Upload to blob storage
                    BlobHandler.upload(
                        storage_account_url=self.documents_storage_account_url,
                        container_name=self.documents_container_name,
                        blob_path=blob_path,
                        content=page_content,
                        metadata=page_metadata,
                        overwrite=True,
                    )
                    
                    # Store the page successfully
                    self.crawler_summary.add_success(page_url)
                    successful_uploads.append({
                        "url": page_url,
                        "blob_path": blob_path,
                        "status": "success",
                    })
                        
                    self._logger.debug(
                        "[BlobManager] Successfully uploaded page to blob: %s -> %s",
                        page_url, blob_path
                    )
                else:
                    # Duplicate content already exists in storage - skip this occurrence
                    duplicate_pages.append({
                        "url": page_url,
                        "blob_path": blob_path,
                        "status": "duplicate",
                        "message": "Content already exists in storage"
                    })
                    self._logger.debug(
                        "[BlobManager] Duplicate URL found for page: %s (URL already processed in this session)", page_url
                    )
                    
            except Exception as e:
                # Track failed upload
                failed_uploads.append({
                    "url": page_data.get('url', 'unknown'),
                    "error": str(e)
                })
                self.crawler_summary.add_failure(page_data.get('url', 'unknown'))
                
                self._logger.error(
                    "[BlobManager] Failed to upload page to blob: %s, error: %s",
                    page_data.get('url', 'unknown'), str(e)
                )
        
        return {
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "duplicate_pages": duplicate_pages,
            "total_processed": len(formatted_pages),
            "total_successful": len(successful_uploads),
            "total_failed": len(failed_uploads),
            "total_duplicates": len(duplicate_pages)
        }

    def _get_blob_path_for_multipage(self, url: str, metadata: dict, content_type: str) -> str:
        """
        Generate blob path for multipage crawl results.
        
        Args:
            url: The page URL
            metadata: Page metadata containing crawl session info
            content_type: Content type for the blob
            
        Returns:
            Generated blob path for the page
        """
        from urllib.parse import urlparse
        import os
        
        # Parse the URL
        parsed_url = urlparse(url)
        base_blob_path = f"{parsed_url.netloc}{parsed_url.path}"
        base_name = os.path.basename(base_blob_path)
        file_name, file_extension = os.path.splitext(base_name)
        
        # Handle file extension based on content type
        if "text/plain" in content_type:
            if len(file_name) == 0:
                file_name = "page"
            file_extension = ".txt"
        elif len(file_name) == 0:
            file_name = "page"
            file_extension = ".txt"
        
        
        # Generate unique blob path with multipage prefix
        blob_path = os.path.join(
            f"WebCrawler_Output/{base_blob_path}",
            f"{file_name}{file_extension}"
        )
        
        return blob_path

    def _content_exists_by_checksum(self, checksum: str, url: str = None) -> bool:
        """
        Check if content with this checksum already exists anywhere in blob storage.
        
        This implementation maintains a simple in-memory set of URLs we've 
        seen in this session to avoid duplicates within the same crawl.
        Now considers each unique URL as unique content regardless of checksum.
        
        Args:
            checksum: MD5 checksum of the content (kept for backward compatibility)
            url: The page URL to check for uniqueness
            
        Returns:
            True if content with this URL exists, False otherwise
        """
        # Initialize session URL tracker if not exists
        if not hasattr(self, '_session_urls'):
            self._session_urls = set()
        
        # If URL is provided, use URL-based deduplication (preferred method)
        if url:
            # Check if we've already processed this URL in this session
            if url in self._session_urls:
                return True
            
            # Add URL to our tracker and allow storage
            self._session_urls.add(url)
            return False
        
        # Fallback to checksum-based deduplication if no URL provided
        if not hasattr(self, '_session_checksums'):
            self._session_checksums = set()
        
        # Check if we've already processed this checksum in this session
        if checksum in self._session_checksums:
            return True
        
        # Add checksum to our tracker and allow storage
        self._session_checksums.add(checksum)
        return False
    
    def _blob_path_exists(self, blob_path: str) -> bool:
        """
        Check if a blob exists at the given path.
        
        Args:
            blob_path: Path to check
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            BlobHandler.download(
                storage_account_url=self.documents_storage_account_url,
                container_name=self.documents_container_name,
                blob_path=blob_path,
            )
            return True
        except Exception:
            return False


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