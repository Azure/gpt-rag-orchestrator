"""
Configuration models for web scraping functionality.

This module defines Pydantic models used to configure web crawling and scraping operations.
"""

from typing import List, Optional
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class Storage(BaseModel):
    """Storage configuration for Azure Blob Storage."""
    account: str
    container: str


class Document(BaseModel):
    """Document configuration."""
    urls: List[str]
    storage: Optional[Storage] = None


class HtmlParserConfig(BaseModel):
    """HTML parser configuration."""
    ignored_classes: List[str]


class Html(BaseModel):
    """HTML processing configuration."""
    striptags: bool
    parser: HtmlParserConfig


class CrawlerConfig(BaseModel):
    """Main crawler configuration."""
    documents: Document
    html: Optional[Html] = None