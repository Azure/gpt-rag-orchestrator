"""
Parallel Processing Module

This module handles parallel processing of multiple URLs for web scraping operations.
It manages concurrent execution using ThreadPoolExecutor and coordinates between
scraping and blob storage operations.
"""

import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

from .config import CrawlerConfig
from .scraper import WebScraper
from .blob_manager import WebCrawlerManager

_logger = logging.getLogger(__name__)


class ParallelScrapingProcessor:
    """Handles parallel processing of web scraping operations."""

    def __init__(self, config: CrawlerConfig = None, max_workers: int = 10):
        self.config = config or self._create_default_config()
        self.max_workers = max_workers
        self.scraper = WebScraper(self.config, _logger)

    def process_urls(
        self,
        urls: List[str],
        crawler_manager: Optional[WebCrawlerManager] = None,
        request_id: str = None,
        organization_id: str = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process multiple URLs in parallel with optional blob storage.

        Args:
            urls: List of URLs to process
            crawler_manager: Optional WebCrawlerManager for blob storage
            request_id: Optional request identifier for logging
            organization_id: Optional organization identifier for metadata

        Returns:
            Tuple of (scraping_results, blob_storage_results)
        """
        results = []
        blob_storage_results = []

        # Determine if blob storage is available
        blob_storage_available = crawler_manager is not None

        # Use ThreadPoolExecutor with limited workers to prevent overwhelming servers
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(urls))
        ) as executor:
            # Submit all scraping tasks
            if blob_storage_available:
                future_to_url = {
                    executor.submit(
                        self._process_single_url_with_blob_storage,
                        url,
                        crawler_manager,
                        request_id,
                        organization_id,
                    ): url
                    for url in urls
                }
            else:
                # If no blob storage, use the simpler scraping function
                future_to_url = {
                    executor.submit(
                        self._process_single_url_without_blob_storage,
                        url,
                        request_id,
                    ): url
                    for url in urls
                }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if blob_storage_available:
                        scrape_result, blob_result = future.result()
                        results.append(scrape_result)
                        blob_storage_results.append(blob_result)

                        _logger.info(
                            "[ParallelProcessor] Successfully processed with storage - request_id: %s, url: %s, blob_status: %s",
                            request_id,
                            url,
                            blob_result.get("status", "unknown"),
                        )
                    else:
                        # No blob storage - just scraping results
                        scrape_result = future.result()
                        results.append(scrape_result)

                        # Create a placeholder blob result indicating no storage
                        blob_storage_results.append(
                            {
                                "url": url,
                                "status": "skipped",
                                "message": "Blob storage not configured",
                                "blob_path": None,
                            }
                        )

                        _logger.info(
                            "[ParallelProcessor] Successfully processed (no storage) - request_id: %s, url: %s",
                            request_id,
                            url,
                        )
                except Exception as e:
                    error_result = {
                        "url": url,
                        "status": "error",
                        "error": str(e),
                        "content": None,
                        "content_type": None,
                        "title": None,
                    }
                    blob_error_result = {
                        "url": url,
                        "status": "error",
                        "error": str(e),
                        "blob_path": None,
                    }
                    results.append(error_result)
                    blob_storage_results.append(blob_error_result)

                    _logger.error(
                        "[ParallelProcessor] Failed to process - request_id: %s, url: %s, error: %s",
                        request_id,
                        url,
                        str(e),
                    )

        return results, blob_storage_results

    def _process_single_url_without_blob_storage(
        self, url: str, request_id: str
    ) -> Dict[str, Any]:
        """Process a single URL without blob storage."""
        return self.scraper.scrape_page(url, request_id)

    def _process_single_url_with_blob_storage(
        self, url: str, crawler_manager: WebCrawlerManager, request_id: str, organization_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process a single URL with blob storage."""
        try:
            # First, scrape the content
            scrape_result = self.scraper.scrape_page(url, request_id)

            if scrape_result["status"] != "success":
                # If scraping failed, return the error for both operations
                blob_result = {
                    "url": url,
                    "status": "error",
                    "error": f"Scraping failed: {scrape_result['error']}",
                    "blob_path": None,
                }
                return scrape_result, blob_result

            # Format content for blob storage
            blob_data = WebScraper.format_content_for_blob_storage(
                scrape_result, request_id, organization_id
            )

            if not blob_data:
                blob_result = {
                    "url": url,
                    "status": "error",
                    "error": "Failed to format content for blob storage",
                    "blob_path": None,
                }
                return scrape_result, blob_result

            # Upload to blob storage
            try:
                text_content_type = "text/plain"
                contents = [blob_data]

                crawler_manager.store_in_blob(
                    url=url, contents=contents, content_type=text_content_type
                )

                # Update crawler summary
                crawler_manager.crawler_summary.add_success(url)

                blob_result = {
                    "url": url,
                    "status": "success",
                    "message": "Successfully uploaded to blob storage",
                    "blob_path": crawler_manager._get_blob_path(url, 0, text_content_type),
                    "content_size_bytes": len(blob_data["content"]),
                }

            except Exception as blob_error:
                crawler_manager.crawler_summary.add_failure(url)
                blob_result = {
                    "url": url,
                    "status": "error",
                    "error": f"Blob storage upload failed: {str(blob_error)}",
                    "blob_path": None,
                }

            return scrape_result, blob_result

        except Exception as e:
            # If anything fails, return error results for both operations
            error_scrape_result = {
                "url": url,
                "status": "error",
                "error": str(e),
                "content": None,
                "content_type": None,
                "title": None,
            }

            error_blob_result = {
                "url": url,
                "status": "error",
                "error": str(e),
                "blob_path": None,
            }

            return error_scrape_result, error_blob_result

    @staticmethod
    def _create_default_config() -> CrawlerConfig:
        """Create a default configuration for processing."""
        from .config import CrawlerConfig, Document, Html, HtmlParserConfig

        return CrawlerConfig(
            documents=Document(urls=[], storage=None),
            html=Html(
                striptags=True,
                parser=HtmlParserConfig(
                    ignored_classes=["nav", "footer", "sidebar", "ads"]
                ),
            ),
        )

    def get_processing_summary(
        self, results: List[Dict[str, Any]], blob_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a summary of processing results.

        Args:
            results: List of scraping results
            blob_results: List of blob storage results

        Returns:
            Dictionary containing processing summary
        """
        successful_scrapes = len([r for r in results if r["status"] == "success"])
        failed_scrapes = len([r for r in results if r["status"] == "error"])
        successful_blob_uploads = len(
            [r for r in blob_results if r["status"] == "success"]
        )
        failed_blob_uploads = len(
            [r for r in blob_results if r["status"] == "error"]
        )
        skipped_blob_uploads = len(
            [r for r in blob_results if r["status"] == "skipped"]
        )

        return {
            "total_urls": len(results),
            "successful_scrapes": successful_scrapes,
            "failed_scrapes": failed_scrapes,
            "successful_blob_uploads": successful_blob_uploads,
            "failed_blob_uploads": failed_blob_uploads,
            "skipped_blob_uploads": skipped_blob_uploads,
            "success_rate": (
                (successful_scrapes / len(results)) * 100 if results else 0
            ),
        }


def process_urls_parallel(
    urls: List[str],
    crawler_manager: Optional[WebCrawlerManager] = None,
    request_id: str = None,
    max_workers: int = 10,
    organization_id: str = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function for parallel URL processing.

    Args:
        urls: List of URLs to process
        crawler_manager: Optional WebCrawlerManager for blob storage
        request_id: Optional request identifier for logging
        max_workers: Maximum number of concurrent workers
        organization_id: Optional organization identifier for metadata

    Returns:
        Tuple of (scraping_results, blob_storage_results)
    """
    processor = ParallelScrapingProcessor(max_workers=max_workers)
    return processor.process_urls(urls, crawler_manager, request_id, organization_id) 