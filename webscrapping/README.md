# Web Scrapping Package

This package provides a comprehensive web scraping solution with parallel processing capabilities and optional Azure Blob Storage integration. The code has been refactored into focused, maintainable modules.

## 📁 File Structure

The package has been organized into the following modules:

### Core Components

- **`config.py`** - Configuration models and data structures
- **`html_parser.py`** - HTML content processing and cleaning
- **`scraper.py`** - Core web scraping functionality for different content types
- **`blob_manager.py`** - Azure Blob Storage operations and management
- **`parallel_processor.py`** - Concurrent URL processing using ThreadPoolExecutor
- **`metrics.py`** - Execution tracking and summary reporting
- **`utils.py`** - Common utility functions and helpers
- **`orchestrator.py`** - Main coordination logic and entry points

## 🚀 Usage Examples

### Standalone Scraping (No Azure Functions)

```python
from webscrapping import scrape_urls_standalone

# Simple usage
urls = ['http://example.com', 'http://example2.com']
result = scrape_urls_standalone(urls)

print(f"Status: {result['status']}")
print(f"Processed {result['summary']['total_urls']} URLs")
print(f"Success rate: {result['summary']['successful_scrapes']}/{result['summary']['total_urls']}")
```

### Azure Functions Integration

```python
# In your Azure Function
from webscrapping.orchestrator import main

def main(req: func.HttpRequest) -> func.HttpResponse:
    return main(req)
```

### Using Individual Components

```python
from webscrapping.scraper import WebScraper
from webscrapping.html_parser import HtmlParser
from webscrapping.config import CrawlerConfig

# Initialize scraper with custom config
config = CrawlerConfig(...)
scraper = WebScraper(config)

# Scrape a single page
result = scraper.scrape_page('http://example.com')
```

### Parallel Processing

```python
from webscrapping.parallel_processor import process_urls_parallel
from webscrapping.blob_manager import create_crawler_manager_from_env

# Set up blob storage (optional)
crawler_manager = create_crawler_manager_from_env('my-request-id')

# Process URLs in parallel
urls = ['http://example.com', 'http://example2.com']
results, blob_results = process_urls_parallel(
    urls, 
    crawler_manager=crawler_manager,
    max_workers=5
)
```

## 🔧 Features

### Content Type Support
- **HTML**: Full parsing with tag removal and content cleaning
- **PDF**: Text extraction using PyMuPDF
- **Plain Text**: Direct processing with encoding handling
- **Unknown Types**: Graceful fallback processing

### Parallel Processing
- Configurable concurrency (default: 10 workers)
- Efficient ThreadPoolExecutor implementation
- Proper error handling per URL
- Request rate limiting to prevent server overload

### Azure Blob Storage Integration
- Optional blob storage for scraped content
- Content deduplication using MD5 checksums
- Metadata preservation and enrichment
- Automatic container management

### Error Handling & Logging
- Comprehensive error handling for each processing stage
- Structured logging with request IDs
- Timeout management (30-second default)
- Graceful degradation when services are unavailable

### Metrics & Monitoring
- Detailed execution summaries
- Success/failure tracking
- Processing duration measurement
- Blob storage operation metrics

## ⚙️ Configuration

### Environment Variables

```bash
# Required for blob storage (optional)
AZURE_STORAGE_ACCOUNT_URL=https://yourstorageaccount.blob.core.windows.net
```

### Configuration Models

```python
from webscrapping.config import CrawlerConfig, Document, Storage, Html, HtmlParserConfig

config = CrawlerConfig(
    documents=Document(
        urls=['http://example.com'],
        storage=Storage(
            account="https://yourstorageaccount.blob.core.windows.net",
            container="documents"
        )
    ),
    html=Html(
        striptags=True,
        parser=HtmlParserConfig(
            ignored_classes=["nav", "footer", "sidebar", "ads"]
        )
    )
)
```

## 🏗️ Architecture Benefits

### Separation of Concerns
Each module has a single, well-defined responsibility:
- **HTML Parser**: Only handles HTML processing
- **Scraper**: Only handles HTTP requests and content extraction
- **Blob Manager**: Only handles Azure storage operations
- **Parallel Processor**: Only handles concurrent execution
- **Orchestrator**: Only coordinates between components

### Testability
- Each component can be unit tested independently
- Mock-friendly interfaces
- Clear input/output contracts

### Maintainability
- Smaller, focused files (100-300 lines each vs 1145 lines)
- Easy to locate and modify specific functionality
- Reduced cognitive load when working on specific features

### Reusability
- Components can be imported and used independently
- Flexible configuration options
- Both standalone and Azure Functions support

## 📊 Performance Characteristics

- **Concurrency**: Up to 10 parallel requests by default
- **Timeout**: 30 seconds per request
- **Memory**: Efficient streaming for large content
- **Error Recovery**: Individual URL failures don't affect others

## 🔍 Monitoring & Debugging

All components use structured logging with consistent formats:

```
[ComponentName] Operation description - request_id: xyz, additional_context
```

Request IDs are generated automatically and propagated through all operations for easy tracing.

