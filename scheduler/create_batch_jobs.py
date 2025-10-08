"""
Create batch report jobs for ALL organizations by querying Cosmos DB.

This script fetches all organizations, then for each organization fetches brands,
products, and competitors from their respective Cosmos containers and creates all
necessary report jobs automatically.
"""
import uuid
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

from shared.cosmos_jobs import cosmos_container
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import os

COSMOS_URI = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_DB_NAME = os.getenv("AZURE_DB_NAME")

def get_cosmos_client():
    """Get Cosmos DB client using Managed Identity"""
    return CosmosClient(COSMOS_URI, credential=DefaultAzureCredential())

def fetch_items_for_org(container_name: str, organization_id: str):
    """Fetch all items from a Cosmos container for a specific organization"""
    try:
        client = get_cosmos_client()
        database = client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(container_name)

        query = "SELECT * FROM c WHERE c.organization_id = @organization_id"
        params = [{"name": "@organization_id", "value": organization_id}]

        items = list(container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        return items
    except Exception as e:
        print(f"Error fetching {container_name} for org {organization_id}: {e}")
        return []

def create_batch_jobs(include_brands=True, include_products=True, include_competitors=True):
    """Create all report jobs for all organizations"""

    schedule_time = datetime.now(timezone.utc).isoformat()

    total_created = 0
    org_summaries = []

    try:
        client = get_cosmos_client()
        database = client.get_database_client(COSMOS_DB_NAME)
        org_container = database.get_container_client("organizations")

        all_orgs = list(org_container.query_items(
            query="SELECT * FROM c",
            enable_cross_partition_query=True
        ))

        print(f"Found {len(all_orgs)} organizations to process")
        print("=" * 80)

        # Get reportJobs container
        jobs_container = cosmos_container()

        # Process each organization
        for org in all_orgs:
            organization_id = org.get("id")
            industry_description = org.get("industry_description", "")

            if not organization_id:
                continue

            print(f"\nProcessing organization: {organization_id}")
            org_jobs_created = 0

            # 1. Create brand analysis jobs
            if include_brands:
                brands = fetch_items_for_org("brands", organization_id)
                print(f"  Found {len(brands)} brands")

                for brand in brands:
                    brand_name = brand.get("name")
                    if not brand_name:
                        continue

                    job_id = str(uuid.uuid4())
                    job_doc = {
                        "id": job_id,
                        "job_id": job_id,
                        "tenant_id": organization_id,
                        "organization_id": organization_id,
                        "idempotency_key": str(uuid.uuid4()),
                        "report_key": "brand_analysis",
                        "report_name": brand_name,
                        "params": {
                            "brand_focus": brand_name,
                            "industry_context": industry_description
                        },
                        "status": "QUEUED",
                        "schedule_time": schedule_time,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }

                    jobs_container.create_item(job_doc)
                    org_jobs_created += 1
                    print(f"    Created brand job: {brand_name}")

            # 2. Create product analysis jobs
            if include_products:
                products = fetch_items_for_org("products", organization_id)
                print(f"  Found {len(products)} products")

                # Group products by category
                category_map = {}
                for product in products:
                    product_name = product.get("name")
                    product_category = product.get("category")
                    if not product_name or not product_category:
                        continue

                    if product_category not in category_map:
                        category_map[product_category] = []
                    category_map[product_category].append(product_name)

                # Create one job per category
                for category, product_names in category_map.items():
                    job_id = str(uuid.uuid4())
                    report_name = product_names[0] if len(product_names) == 1 else ", ".join(product_names)

                    job_doc = {
                        "id": job_id,
                        "job_id": job_id,
                        "tenant_id": organization_id,
                        "organization_id": organization_id,
                        "idempotency_key": str(uuid.uuid4()),
                        "report_key": "product_analysis",
                        "report_name": report_name,
                        "params": {
                            "categories": [
                                {
                                    "product": product_names,
                                    "category": category
                                }
                            ]
                        },
                        "status": "QUEUED",
                        "schedule_time": schedule_time,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }

                    jobs_container.create_item(job_doc)
                    org_jobs_created += 1
                    print(f"    Created product job: {category} - {report_name}")

            # 3. Create competitor analysis jobs
            if include_competitors:
                competitors = fetch_items_for_org("competitors", organization_id)
                brands = fetch_items_for_org("brands", organization_id) if not include_brands else brands
                print(f"  Found {len(competitors)} competitors")

                brand_names = [brand.get("name") for brand in brands if brand.get("name")]

                for competitor in competitors:
                    competitor_name = competitor.get("name")
                    if not competitor_name or not brand_names:
                        continue

                    job_id = str(uuid.uuid4())
                    job_doc = {
                        "id": job_id,
                        "job_id": job_id,
                        "tenant_id": organization_id,
                        "organization_id": organization_id,
                        "idempotency_key": str(uuid.uuid4()),
                        "report_key": "competitor_analysis",
                        "report_name": competitor_name,
                        "params": {
                            "categories": [
                                {
                                    "brands": brand_names,
                                    "competitors": [competitor_name]
                                }
                            ],
                            "industry_context": industry_description
                        },
                        "status": "QUEUED",
                        "schedule_time": schedule_time,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }

                    jobs_container.create_item(job_doc)
                    org_jobs_created += 1
                    print(f"    Created competitor job: {competitor_name}")

            # Add organization summary
            if org_jobs_created > 0:
                org_summaries.append({
                    "organization_id": organization_id,
                    "jobs_created": org_jobs_created
                })
                total_created += org_jobs_created
                print(f"  Total jobs created for {organization_id}: {org_jobs_created}")

        print("\n" + "=" * 80)
        print(f"SUMMARY: Created {total_created} jobs across {len(org_summaries)} organizations")
        print("=" * 80)

        for org_summary in org_summaries:
            print(f"  {org_summary['organization_id']}: {org_summary['jobs_created']} jobs")

        return {
            "total_created": total_created,
            "organizations": org_summaries
        }

    except Exception as e:
        print(f"Error creating batch jobs: {e}")
        raise

if __name__ == "__main__":
    result = create_batch_jobs()
    print(f"\ Done! Created {result['total_created']} jobs")