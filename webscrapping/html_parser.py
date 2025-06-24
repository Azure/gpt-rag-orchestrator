"""
HTML Parser Module

This module contains HTML processing and content cleaning functionality
for web scraping operations.
"""

import re
import logging
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownTextSplitter
from .config import CrawlerConfig

_logger = logging.getLogger(__name__)


class HtmlParser:
    """HTML parser for processing web content."""

    def __init__(self, configuration: CrawlerConfig, logger=None):
        self.configuration = configuration
        self._logger = logger or _logger

    def clean_html(self, content: str):
        """
        Clean HTML content and return processed chunks.

        Returns a list of dictionaries with content and metadata.
        Content is returned as UTF-8 encoded bytes.
        """
        # Parse HTML content
        soup = BeautifulSoup(content, "html.parser")

        # Extract title for the page
        _page_title = soup.title.string if soup.title else "No Title"

        # Extract text content from the page
        if self.configuration.html and self.configuration.html.striptags:
            _page_text = self._clean_text(soup)
        else:
            _page_text = content

        self._logger.debug("Page title is: %s", _page_title)
        # Clean the title for metadata (remove problematic characters but keep it readable)
        _title = _page_title.strip()
        # Replace characters that might cause issues in HTTP headers
        _title = _title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # Remove excessive whitespace
        _title = " ".join(_title.split())
        # Truncate very long titles to prevent metadata issues
        if len(_title) > 200:
            _title = _title[:197] + "..."
        self._logger.debug("Page title after cleaning is: %s", _title)

        # Perform chunking using the custom_markdown_chunking method
        chunks = self.custom_markdown_chunking(_page_text)
        # Generate metadata for each chunk
        return [
            {
                "content": chunk.encode("utf-8"),
                "metadata": {"title": _title},
            }
            for chunk in chunks
        ]

    def _clean_text(self, soup):
        """Clean text by removing extra whitespace and special characters."""
        # Get the list of CSS classes to remove from the config
        classes_to_remove = []
        if (
            self.configuration.html
            and self.configuration.html.parser
            and self.configuration.html.parser.ignored_classes
        ):
            classes_to_remove = self.configuration.html.parser.ignored_classes

        # Iterate over all tags and remove specific classes
        for tag in soup.find_all(True):  # finds all HTML tags
            if "class" in tag.attrs:
                # Filter out the specific classes from the class list
                tag["class"] = [
                    cls for cls in tag["class"] if cls not in classes_to_remove
                ]

                # If the class attribute is now empty, remove it completely
                if not tag["class"]:
                    tag.attrs.pop("class")

        # Serialize the modified HTML back to a string
        page_text = re.sub(r"[\n\r\t]", "", soup.get_text()).strip()

        # Remove common date formats
        content = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", "", page_text)

        # Remove common time formats
        content = re.sub(
            r"\b\d{1,2}:\d{2}(?::\d{2})?\s?(AM|PM|am|pm)?\b", "", page_text
        )

        return content

    def custom_markdown_chunking(self, content):
        """Split content using MarkdownTextSplitter from LangChain."""
        # Check if content is already plain text or HTML
        if self.configuration.html and self.configuration.html.striptags:
            # Content is already plain text, split it directly
            splitter = MarkdownTextSplitter()
            chunks = splitter.split_text(content)
        else:
            # Content is HTML, process it as HTML
            soup = BeautifulSoup(content, "html.parser")
            html = str(soup)
            splitter = MarkdownTextSplitter()
            chunks = splitter.split_text(html)

        if chunks:
            return chunks
        else:
            # Fallback: return content as single chunk if splitting fails
            return [content] 