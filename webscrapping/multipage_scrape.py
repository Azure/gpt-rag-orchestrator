# To install: pip install tavily-python
import json
import logging
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

_logger = logging.getLogger(__name__)


def _get_tavily_client() -> TavilyClient:
    """Initialize and return a Tavily client with proper API key validation."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required but not set")
    
    try:
        return TavilyClient(api_key=api_key)
    except Exception as e:
        _logger.error(f"Failed to initialize Tavily client: {e}")
        raise ValueError(f"Invalid Tavily API key or client initialization failed: {e}")


def _validate_url(url: str) -> bool:
    """Validate that the provided URL is properly formatted.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False

WEBSCRAPE_PROMPT = """
Crawl and extract all pages related to branding, marketing, services, pricing, service offerings, and other business-relevant or informative content. Exclude pages that are legal, administrative, or unrelated—such as terms and conditions, privacy policies, disclaimers, or other non-business-focused pages.
Only scrape english pages.
"""
def crawl_website(url: str, limit: int = 30, max_depth: int = 4, max_breadth: int = 15) -> Dict[str, Any]:
    """Crawl a website using Tavily API and extract relevant business content.
    
    Args:
        url: The website URL to crawl
        limit: Maximum number of pages to crawl (default: 30)
        max_depth: Maximum crawl depth (default: 4)
        max_breadth: Maximum breadth per level (default: 15)
        
    Returns:
        Dictionary containing crawl results and metadata
        
    Raises:
        ValueError: If URL is invalid or API key is not configured
    """
    if not _validate_url(url):
        raise ValueError(f"Invalid URL format: {url}")
    
    try:
        client = _get_tavily_client()
        _logger.info(f"Starting crawl for {url} with limit={limit}, depth={max_depth}, breadth={max_breadth}")
        
        response = client.crawl(
            url=url,
            instructions=WEBSCRAPE_PROMPT,
            limit=limit,
            max_depth=max_depth,
            max_breadth=max_breadth,
            extract_depth="advanced",
            categories=["Documentation", "Blogs", "E-Commerce", "Media", "Pricing"],
            allow_external=False
        )
        
        _logger.info(f"Successfully crawled {url}, found {len(response.get('results', []))} results")
        return response
        
    except ValueError:
        raise
    except Exception as e:
        _logger.error(f"Error crawling website {url}: {e}")
        return {
            "base_url": url,
            "results": [],
            "response_time": 0.0,
            "error": str(e)
        }

def crawl_website_and_save(url: str, output_file: str = "response.jsonl", 
                          limit: int = 30, max_depth: int = 4, max_breadth: int = 15) -> bool:
    """Crawl a website and save results to a JSON Lines file.
    
    Args:
        url: The website URL to crawl
        output_file: Path to output file (default: "response.jsonl")
        limit: Maximum number of pages to crawl (default: 30)
        max_depth: Maximum crawl depth (default: 4)
        max_breadth: Maximum breadth per level (default: 15)
        
    Returns:
        True if results were saved successfully, False otherwise
        
    Raises:
        ValueError: If URL is invalid or API key is not configured
        IOError: If file write operation fails
    """
    try:
        response = crawl_website(url, limit, max_depth, max_breadth)
        
        if response.get('results'):
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
                _logger.info(f"Successfully saved {len(response['results'])} results to {output_file}")
                return True
            except IOError as e:
                _logger.error(f"Failed to write results to {output_file}: {e}")
                raise IOError(f"Could not save results to file: {e}")
        else:
            error_msg = response.get('error', 'Unknown error')
            _logger.warning(f"No results found for {url}. Error: {error_msg}")
            return False
            
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        _logger.error(f"Unexpected error in crawl_website_and_save: {e}")
        raise

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        success = crawl_website_and_save("https://www.salesfactory.com/")
        if success:
            print("Crawling completed successfully")
        else:
            print("Crawling completed but no results found")
    except Exception as e:
        _logger.error(f"Crawling failed: {e}")
        print(f"Error: {e}")
        exit(1)