import logging
import os
import requests
from datetime import datetime, timezone
import azure.functions as func
from tenacity import retry, wait_exponential, stop_after_attempt
from shared import cosmos_client_async

logger = logging.getLogger(__name__)

# Environment variables
WEB_APP_URL = os.getenv("WEB_APP_URL", None)
TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

def main(mytimer: func.TimerRequest) -> None:
    """Main function for report scheduler - runs at 2:00 AM UTC every Sunday"""
    
    # Check if the environment variable is set
    if not WEB_APP_URL:
        logger.error("WEB_APP_URL environment variable not set")
        return
    
    try:
        organizations = get_all_organizations()
        logger.info(f"Total organizations fetched: {len(organizations)}")
        logger.debug(f"Organizations: {organizations}")
    except Exception as e:
        logger.error(f"Error fetching organizations from Cosmos DB: {str(e)}")
        return

    if not organizations:
        logger.warning("No organizations found.")
        return

    start_time = datetime.now(timezone.utc)
    logger.info(f"Report scheduler started at {start_time}")
    
    try:
        logger.info("Starting HTTP requests to endpoints for each organization")

        for org in organizations:
            org_id = org.get("id")
            industry_description = org.get("industry_description")

            if not org_id or not industry_description:
                logger.warning(f"Organization missing required fields (id or industry_description): {org}")
                continue
            REPORT_JOBS_API_ENDPOINT = f"{WEB_APP_URL}/api/report-jobs?organization_id={org_id}"
            logger.info(f"Sending request to: {REPORT_JOBS_API_ENDPOINT}")

            logger.info(f"Fetching brands for organization: {org_id}")
            try:
                organization_brands = get_brands(org_id)
            except Exception as e:
                logger.error(f"Error fetching brands for organization {org_id}: {str(e)}")

            if not organization_brands:
                logger.warning(f"No brands found for organization: {org_id}")
                continue

            logger.info(f"Brands found for organization: {org_id}. Proceeding with report generation.")

            for brand in organization_brands:
                brand_name = brand.get("name")
                
                if not brand_name:
                    logger.warning(f"Brand missing required fields: {brand}")
                    continue

                payload = create_brands_payload(brand_name, industry_description)

                try:
                    response = send_http_request(REPORT_JOBS_API_ENDPOINT, payload)
                    log_response_result(REPORT_JOBS_API_ENDPOINT, response)
                except Exception as e:
                    logger.error(f"Failed to send request to {REPORT_JOBS_API_ENDPOINT}: {str(e)}")

            organization_products = []
            try:
                organization_products = get_products(org_id)
            except Exception as e:
                logger.error(f"Error fetching products for organization {org_id}: {str(e)}")

            if not organization_products:
                logger.warning(f"No products found for organization: {org_id}")
            else:
                logger.info(f"Products found for organization: {org_id}. Proceeding with report generation.")
                category_map = {}
                for product in organization_products:
                    product_name = product.get("name")
                    product_category = product.get("category")
                    if not product_name or not product_category:
                        logger.warning(f"Product missing required fields: {product}")
                        continue

                    if product_category not in category_map:
                        category_map[product_category] = []
                    category_map[product_category].append(product_name)

                for product_category, product_names in category_map.items():
                    payload = create_products_payload(product_names, product_category)
                    try:
                        response = send_http_request(REPORT_JOBS_API_ENDPOINT, payload)
                        log_response_result(REPORT_JOBS_API_ENDPOINT, response)
                    except Exception as e:
                        logger.error(f"Failed to send request to {REPORT_JOBS_API_ENDPOINT}: {str(e)}")

            organization_competitors = []
            try:
                organization_competitors = get_competitors(org_id)
            except Exception as e:
                logger.error(f"Error fetching competitors for organization {org_id}: {str(e)}")

            if not organization_competitors:
                logger.warning(f"No competitors found for organization: {org_id}")
            else:
                logger.info(f"Competitors found for organization: {org_id}. Proceeding with report generation.")
                brand_names = [brand.get("name") for brand in organization_brands if brand.get("name")]
                if not brand_names:
                    logger.warning(f"No valid brand names found for organization: {org_id}")
                    continue

                for competitor in organization_competitors:
                    competitor_name = competitor.get("name")
                    if not competitor_name:
                        logger.warning(f"Competitor missing required fields: {competitor}")
                        continue
                    payload = create_competitors_payload(competitor_name, brand_names, industry_description)
                    try:
                        response = send_http_request(REPORT_JOBS_API_ENDPOINT, payload)
                        log_response_result(REPORT_JOBS_API_ENDPOINT, response)
                    except Exception as e:
                        logger.error(f"Failed to send request to {REPORT_JOBS_API_ENDPOINT}: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in report scheduler: {str(e)}")
    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info(
            f"Report scheduler completed at {end_time}. Duration: {duration}"
        )

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def send_http_request(url: str, payload: dict) -> requests.Response:
    """Send HTTP request with retry logic"""
    logger.debug(f"Sending HTTP request to {url}")
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    
    logger.debug(f"Received response from {url}: {response.status_code}")
    return response

def log_response_result(endpoint: str, response: requests.Response) -> None:
    """Log the results of the endpoint call"""
    try:
        response_json = response.json() if response.content else {}
    except Exception:
        response_json = {"error": "Could not parse JSON response"}
    
    log_data = {
        "endpoint": endpoint,
        "status_code": response.status_code,
        "response_time": response.elapsed.total_seconds(),
        "response_size": len(response.content),
        "response_data": response_json
    }
    
    if response.status_code == 200:
        logger.info(f"Successfully called {endpoint}: {log_data}")
    elif response.status_code in [401, 403]:
        logger.warning(f"Authentication/Authorization issue with {endpoint}: {log_data}")
    elif response.status_code in [404, 405]:
        logger.warning(f"Endpoint not found or method not allowed for {endpoint}: {log_data}")
    elif response.status_code >= 500:
        logger.error(f"Server error for {endpoint}: {log_data}")
    else:
        logger.warning(f"Unexpected status code for {endpoint}: {log_data}")

def get_all_organizations():
    """Get all elements from the 'organizations' container in Cosmos DB."""
    db_name = os.environ.get("AZURE_DB_NAME")
    if not db_name:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    container = cosmos_client_async.get_container(db_name, "organizations")
    # Query to fetch all documents
    query = "SELECT * FROM c"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    return items

def get_brands(organization_id: str):
    """Get elements from the 'brands' container based on the provided organization_id."""
    db_name = os.environ.get("AZURE_DB_NAME")
    if not db_name:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    container = cosmos_client_async.get_container(db_name, "brands")
    # Query to fetch all documents
    query = f"SELECT * FROM c WHERE c.organization_id = '{organization_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    return items

def create_brands_payload(brand_name: str, industry_description: str) -> dict:
    """Create the payload for the brands API request."""
    return {
        "report_key": "brand_analysis",
        "report_name": brand_name,
        "params": {
            "brand_focus": brand_name,
            "industry_context": industry_description
        }
    }

def create_products_payload(product_names: list[str], category: str) -> dict:
    """Create the payload for the products API request."""
    return {
        "report_key": "product_analysis",
        "report_name": product_names[0] if len(product_names) == 1 else ", ".join(product_names),
        "params": {
            "categories": {
                "product": product_names,
                "category": category
            }
        }
    }

def create_competitors_payload(competitor_name: list[str], brands: list[str], industry_description: str) -> dict:
    """Create the payload for the competitors API request."""
    return {
        "report_key": "competitor_analysis",
        "report_name": competitor_name,
        "params": {
            "categories": {
                "brands": brands,
                "competitors": competitor_name,
            },
            "industry_context": industry_description
        }
    }


def get_products(organization_id: str):
    """Get elements from the 'products' container based on the provided organization_id."""
    db_name = os.environ.get("AZURE_DB_NAME")
    if not db_name:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    container = cosmos_client_async.get_container(db_name, "products")
    # Query to fetch all documents
    query = f"SELECT * FROM c WHERE c.organization_id = '{organization_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    return items

def get_competitors(organization_id: str):
    """Get elements from the 'competitors' container based on the provided organization_id."""
    db_name = os.environ.get("AZURE_DB_NAME")
    if not db_name:
        raise ValueError("AZURE_DB_NAME environment variable not set")
    container = cosmos_client_async.get_container(db_name, "competitors")
    # Query to fetch all documents
    query = f"SELECT * FROM c WHERE c.organization_id = '{organization_id}'"
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    return items