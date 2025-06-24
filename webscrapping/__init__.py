"""
Web Scrapping Package

A comprehensive web scraping solution with parallel processing and Azure Blob Storage integration.

Main Components:
- HTML Parser: Process and clean HTML content
- Web Scraper: Core scraping functionality for different content types
- Blob Manager: Azure Blob Storage integration
- Parallel Processor: Concurrent URL processing
- Metrics: Tracking and reporting
- Orchestrator: Main coordination and entry points

Usage Examples:

# Standalone scraping (without Azure Functions)
from webscrapping import scrape_urls_standalone
result = scrape_urls_standalone(['http://example.com', 'http://example2.com'])

# Individual components
from webscrapping.scraper import WebScraper
from webscrapping.html_parser import HtmlParser
from webscrapping.blob_manager import WebCrawlerManager

# Azure Functions entry point
from webscrapping.orchestrator import main as azure_function_main
"""

from .orchestrator import scrape_urls_standalone
from .scraper import WebScraper  
from .html_parser import HtmlParser
from .blob_manager import WebCrawlerManager, create_crawler_manager_from_env
from .parallel_processor import ParallelScrapingProcessor, process_urls_parallel
from .metrics import CrawlerSummary
from .config import CrawlerConfig, Document, Storage, Html, HtmlParserConfig

# Main entry points
__all__ = [
    # Main functions
    'scrape_urls_standalone',
    
    # Core classes
    'WebScraper',
    'HtmlParser', 
    'WebCrawlerManager',
    'ParallelScrapingProcessor',
    'CrawlerSummary',
    
    # Configuration
    'CrawlerConfig',
    'Document',
    'Storage', 
    'Html',
    'HtmlParserConfig',
    
    # Utility functions
    'create_crawler_manager_from_env',
    'process_urls_parallel',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Web Scraping Team'
__description__ = 'Parallel web scraping with Azure Blob Storage integration' 