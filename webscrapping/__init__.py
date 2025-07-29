"""
Web Scrapping Package

A simplified web scraping solution for single URL processing with Azure Blob Storage integration.

Main Components:
- HTML Parser: Process and clean HTML content
- Web Scraper: Core scraping functionality for different content types
- Blob Manager: Azure Blob Storage integration
- Single URL Processor: Process one URL at a time
- Metrics: Tracking and reporting
- Orchestrator: Main coordination and entry points

Usage Examples:

# Single URL scraping (without Azure Functions)
from webscrapping import scrape_single_url
result = scrape_single_url('http://example.com')

# Individual components
from webscrapping.scraper import WebScraper
from webscrapping.html_parser import HtmlParser
from webscrapping.blob_manager import WebCrawlerManager

# Azure Functions entry point - use scrape-pages endpoint
"""

from .orchestrator import scrape_single_url
from .scraper import WebScraper  
from .html_parser import HtmlParser
from .blob_manager import WebCrawlerManager, create_crawler_manager_from_env
from .metrics import CrawlerSummary
from .config import CrawlerConfig, Document, Storage, Html, HtmlParserConfig

# Main entry points
__all__ = [
    # Main functions
    'scrape_single_url',
    
    # Core classes
    'WebScraper',
    'HtmlParser', 
    'WebCrawlerManager',
    'CrawlerSummary',
    
    # Configuration
    'CrawlerConfig',
    'Document',
    'Storage', 
    'Html',
    'HtmlParserConfig',
    
    # Utility functions
    'create_crawler_manager_from_env',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Web Scraping Team'
__description__ = 'Simple single URL web scraping with Azure Blob Storage integration' 