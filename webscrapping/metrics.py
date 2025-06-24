"""
Metrics and Summary Tracking Module

This module contains classes for tracking web scraping metrics,
execution statistics, and crawler summaries.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

_logger = logging.getLogger(__name__)


class CrawlerSummary:
    """Tracks crawler execution metrics and statistics."""

    def __init__(self, config_name: str):
        self.activity: str = "scrape_pages"
        self.config_name: str = config_name
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None

        self.success_pages: List[str] = []
        self.failure_pages: List[str] = []
        self.visited_urls: List[str] = []
        self.new_pages: List[str] = []
        self.updated_pages: List[str] = []

        self.closed_reason: Optional[str] = None
        self.log: Optional[str] = None

    def add_success(self, url: str):
        """Add a successful page scrape to the summary."""
        self.success_pages.append(url)
        if url not in self.visited_urls:
            self.visited_urls.append(url)

    def add_failure(self, url: str):
        """Add a failed page scrape to the summary."""
        self.failure_pages.append(url)
        if url not in self.visited_urls:
            self.visited_urls.append(url)

    def add_new_page(self, url: str):
        """Mark a page as newly scraped."""
        if url not in self.new_pages:
            self.new_pages.append(url)

    def add_updated_page(self, url: str):
        """Mark a page as updated."""
        if url not in self.updated_pages:
            self.updated_pages.append(url)

    def finalize(self, closed_reason: str = None):
        """Finalize the summary with end time and closing reason."""
        self.end_time = datetime.now(timezone.utc)
        self.closed_reason = closed_reason

    def get_metrics(self):
        """Get summary metrics for the crawl session."""
        DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
        return {
            "activity": self.activity,
            "config_name": self.config_name,
            "start_time": self.start_time.strftime(DATETIME_FORMAT),
            "end_time": (
                self.end_time.strftime(DATETIME_FORMAT) if self.end_time else None
            ),
            "duration": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else None
            ),
            "success": len(self.success_pages),
            "failure": len(self.failure_pages),
            "processed": len(self.visited_urls),
            "new": len(self.new_pages),
            "updated": len(self.updated_pages),
            "log": self.log,
            "closed_reason": self.closed_reason,
        }

    def get_success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total = len(self.visited_urls)
        if total == 0:
            return 0.0
        return (len(self.success_pages) / total) * 100

    def __str__(self) -> str:
        """String representation of the summary."""
        return (
            f"CrawlerSummary(config={self.config_name}, "
            f"processed={len(self.visited_urls)}, "
            f"success={len(self.success_pages)}, "
            f"failure={len(self.failure_pages)})"
        ) 