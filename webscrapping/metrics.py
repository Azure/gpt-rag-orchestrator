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

    def __init__(self):
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None

        self.success_pages: List[str] = []
        self.failure_pages: List[str] = []
        self.visited_urls: List[str] = []

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


    def get_metrics(self):
        """Get summary metrics for the crawl session."""
        DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
        return {
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
            f"processed={len(self.visited_urls)}, "
            f"success={len(self.success_pages)}, "
            f"failure={len(self.failure_pages)})"
        ) 