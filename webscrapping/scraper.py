"""
Web Scraper Module

This module contains the core web scraping functionality for processing
different types of web content including HTML, PDF, and plain text.
"""

import logging
import requests
import re
from datetime import datetime, timezone
from typing import Dict, Any
from .config import CrawlerConfig
from .html_parser import HtmlParser

_logger = logging.getLogger(__name__)


class WebScraper:
    """Core web scraping functionality."""

    def __init__(self, config: CrawlerConfig = None, logger=None):
        self.config = config
        self._logger = logger or _logger
        self.html_parser = HtmlParser(config, logger) if config else None

    def scrape_page(self, url: str, request_id: str = None) -> Dict[str, Any]:
        """
        Scrape a single page and return processed content.

        This function handles the actual scraping logic for individual URLs.
        It supports HTML, PDF, and plain text content types.

        Args:
            url: The URL to scrape
            request_id: Optional request identifier for logging

        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            # Set headers to mimic a real browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Make HTTP request with timeout
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Get content type from response headers
            content_type = response.headers.get("Content-Type", "").lower()

            # Extract content based on content type
            if "text/html" in content_type:
                return self._process_html_content(url, response, content_type)
            elif "application/pdf" in content_type:
                return self._process_pdf_content(url, response, content_type)
            elif "text/plain" in content_type:
                return self._process_text_content(url, response, content_type)
            else:
                return self._process_unknown_content(url, response, content_type)

        except requests.exceptions.Timeout:
            return self._create_error_result(url, "Request timeout after 30 seconds")
        except requests.exceptions.ConnectionError:
            return self._create_error_result(url, "Connection error - unable to reach the URL")
        except requests.exceptions.HTTPError as e:
            return self._create_error_result(url, f"HTTP error: {e.response.status_code}")
        except Exception as e:
            return self._create_error_result(url, str(e))

    def _process_html_content(self, url: str, response: requests.Response, content_type: str) -> Dict[str, Any]:
        """Process HTML content using the HtmlParser."""
        if self.html_parser:
            # Use the HTML parser for HTML content
            parsed_content = self.html_parser.clean_html(response.text)

            # Extract title from the first chunk
            title = (
                parsed_content[0]["metadata"]["title"] if parsed_content else "No Title"
            )

            # Combine all chunks into a single content string
            combined_content = " ".join(
                [
                    (
                        chunk["content"].decode("utf-8")
                        if isinstance(chunk["content"], bytes)
                        else str(chunk["content"])
                    )
                    for chunk in parsed_content
                ]
            )
        else:
            # Basic HTML processing without parser
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "HTML Document"
            combined_content = soup.get_text()

        return self._create_success_result(url, combined_content, content_type, title)

    def _process_pdf_content(self, url: str, response: requests.Response, content_type: str) -> Dict[str, Any]:
        """Process PDF content using PyMuPDF."""
        try:
            import fitz  # PyMuPDF

            pdf_document = fitz.open(stream=response.content, filetype="pdf")
            text_content = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content += page.get_text()

            # Get title from PDF metadata if available
            metadata = pdf_document.metadata
            title = metadata.get("title", "PDF Document") or "PDF Document"
            pdf_document.close()

            combined_content = text_content.strip()
            return self._create_success_result(url, combined_content, content_type, title)

        except Exception as pdf_error:
            title = "PDF Document (extraction failed)"
            combined_content = f"PDF content could not be extracted: {str(pdf_error)}"
            return self._create_success_result(url, combined_content, content_type, title)

    def _process_text_content(self, url: str, response: requests.Response, content_type: str) -> Dict[str, Any]:
        """Process plain text content."""
        try:
            combined_content = response.content.decode("utf-8")
            # Use first non-empty line as title
            lines = combined_content.splitlines()
            title = "Plain Text Document"
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:
                    title = (
                        stripped_line[:100] + "..."
                        if len(stripped_line) > 100
                        else stripped_line
                    )
                    break
        except UnicodeDecodeError:
            combined_content = response.content.decode("utf-8", errors="replace")
            title = "Plain Text Document"

        return self._create_success_result(url, combined_content, content_type, title)

    def _process_unknown_content(self, url: str, response: requests.Response, content_type: str) -> Dict[str, Any]:
        """Process content with unknown content type."""
        title = "Unknown Content Type"
        combined_content = (
            response.text[:1000] + "..."
            if len(response.text) > 1000
            else response.text
        )
        return self._create_success_result(url, combined_content, content_type, title)

    def _create_success_result(self, url: str, content: str, content_type: str, title: str) -> Dict[str, Any]:
        """Create a successful scraping result."""
        return {
            "url": url,
            "status": "success",
            "content": content,
            "content_type": content_type,
            "title": title,
            "content_length": len(content),
            "error": None,
        }

    def _create_error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create an error result."""
        return {
            "url": url,
            "status": "error",
            "error": error_message,
            "content": None,
            "content_type": None,
            "title": None,
        }

    @staticmethod
    def clean_metadata_value(value):
        """Clean metadata value for Azure blob storage."""
        if value is None:
            return ""
        cleaned = str(value)
        # Keep only safe characters
        cleaned = re.sub(r"[^\w\s\-\.\,\:\(\)]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Limit length
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        return cleaned.strip()

    @staticmethod
    def format_content_for_blob_storage(scrape_result: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        Format scraped content for blob storage.
        
        Args:
            scrape_result: Result from scrape_page method
            request_id: Request identifier
            
        Returns:
            Dictionary with formatted content and metadata for blob storage
        """
        if scrape_result["status"] != "success":
            return None

        # Convert scraped content to bytes
        content_bytes = scrape_result["content"].encode("utf-8")

        # Create metadata for blob storage
        metadata = {
            "title": WebScraper.clean_metadata_value(scrape_result["title"]),
            "original_content_type": WebScraper.clean_metadata_value(
                scrape_result["content_type"]
            ),
            "stored_content_type": "text/plain",
            "content_length": str(scrape_result["content_length"]),
            "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_id": request_id,
        }

        return {
            "content": content_bytes,
            "metadata": metadata,
        } 