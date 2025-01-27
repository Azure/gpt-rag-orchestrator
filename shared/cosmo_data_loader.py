import json
import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential
import uuid
from datetime import datetime, timezone
from typing import List
import logging

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmosDBLoader:
    def __init__(self, container_name: str, 
                db_uri: str = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/" if os.environ.get('AZURE_DB_ID') else None, 
                credential: str = DefaultAzureCredential(), 
                database_name: str = os.environ.get('AZURE_DB_NAME') if os.environ.get('AZURE_DB_NAME') else None):

        self.container_name = container_name
        self.db_uri = db_uri
        self.credential = credential
        self.database_name = database_name

        if not all([self.db_uri, self.credential, self.database_name, self.container_name]):
            raise ValueError("Missing required environment variables for Cosmos DB connection")

        self.client = CosmosClient(url=self.db_uri, credential= self.credential, consistency_level="Session")
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)

    def create_container(self, partition_key: list = ['/companyId', '/reportType']):
        try: 
            self.container = self.database.create_container(
                id=self.container_name,
                partition_key=PartitionKey(path = partition_key, kind="MultiHash"),
                analytical_storage_ttl= -1,
                offer_throughput= 400, 
            )
            logger.info(f"Container {self.container_name} created successfully")
        except exceptions.CosmosResourceExistsError:
            logger.info(f"Container {self.container_name} already exists")
            self.container = self.database.get_container_client(self.container_name)
        except exceptions.CosmosHttpResponseError:
            raise 


    def upload_data(self, data_file_path: str) -> None:
        """
        Upload data from a JSON file to Cosmos DB
        """
        try:
            # Read the JSON file
            with open(data_file_path, 'r') as f:
                data = json.load(f)

            # Upload each schedule to Cosmos DB
            for item in data:
                if item.get('id', None) is None:
                    item['id'] = str(uuid.uuid4())

                item['lastRun'] = datetime.now(timezone.utc).isoformat()

                self.container.create_item(item)
            
            logger.info(f"Successfully uploaded {len(data)} items to Cosmos DB")
        except Exception as e:
            logger.error(f"Error uploading data to Cosmos DB: {str(e)}")
    
    def delete_data(self, company_id: str = None, report_type: str = None) -> None:
        try:
            if company_id and report_type:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}' AND c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key=[item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {company_id} {report_type} from Cosmos DB")
                return True
            elif company_id:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key= [item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {company_id} from Cosmos DB")
                return True
            elif report_type:
                for item in self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                ):
                    self.container.delete_item(item, partition_key= [item['companyId'], item['reportType']])
                logger.info(f"Successfully deleted {report_type} from Cosmos DB")
                return True
        except Exception as e:
            logger.error(f"Error deleting data from Cosmos DB: {str(e)}")
            return False
    
    def get_data(self, company_id: str = None, report_type: str = None, frequency: str = None) -> list:
        try:
            if company_id and report_type and frequency:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}' AND c.reportType = '{report_type}' AND c.frequency = '{frequency}'",
                    enable_cross_partition_query=True
                )
            elif frequency:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.frequency = '{frequency}'",
                    enable_cross_partition_query=True
                )
            elif company_id:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.companyId = '{company_id}'",
                    enable_cross_partition_query=True
                )   
            elif report_type:
                return self.container.query_items(
                    query=f"SELECT * FROM c WHERE c.reportType = '{report_type}'",
                    enable_cross_partition_query=True
                )
            else:
                return self.container.query_items(
                    query=f"SELECT * FROM c",
                    enable_cross_partition_query=True
                )
        except Exception as e:
            logger.error(f"Error getting data from Cosmos DB: {str(e)}")
            return []
    
    def update_last_run(self, data: dict) -> None:
        """ 
        update the last run time in Cosmos DB
        """
        try:
            self.container.upsert_item(data)
            logger.info(f"Successfully updated last run in Cosmos DB")
        except Exception as e:
            logger.error(f"Error updating last run in Cosmos DB: {str(e)}")

    def get_organizations(self):
        try:
            query = "SELECT * FROM c"
            items = self.container.query_items(query, enable_cross_partition_query=True)
            organizations = []
            for item in items:
                if "name" in item and "id" in item and "subscriptionId" in item:
                    organizations.append({
                        "id": item["id"],
                        "name": item["name"],
                        "subscriptionId": item["subscriptionId"]
                    })
            return organizations
        except Exception as e:
            logger.error(f"Error getting organizations from Cosmos DB: {str(e)}")
            return []

    def get_users_by_organizations(self, organization_ids: List[str]) -> List[dict]:
        """
        Get the list of users from Cosmos DB
        """
        try:
            if not organization_ids or len(organization_ids) == 0:
                return []
            formatted_organization_ids = "('" + "','".join(organization_ids) + "')"
            query = f"SELECT * FROM c WHERE c.isReportEmailReceiver = 'true' AND c.data.organizationId IN {formatted_organization_ids}"
            items = self.container.query_items(query, enable_cross_partition_query=True)
            users_by_org = {}
            for item in items:
                if "data" in item and "email" in item["data"]:
                    if "organizationId" in item["data"]:
                        organization_id = item["data"]["organizationId"]
                        if organization_id not in users_by_org:
                            users_by_org[organization_id] = []
                        users_by_org[organization_id].append(item["data"]["email"])
            return users_by_org
        except Exception as e:
            logger.error(f"Error getting users from Cosmos DB: {str(e)}")
            return []

    
if __name__ == "__main__":
    # run the script to upload data to Cosmos DB
    data_file_path = os.path.join(os.path.dirname(__file__), "data/companyID_schedules.json")
    container_name = "schedules"
    db_uri = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/" if os.environ.get('AZURE_DB_ID') else None
    credential = DefaultAzureCredential()
    database_name = os.environ.get('AZURE_DB_NAME') if os.environ.get('AZURE_DB_NAME') else None


    cosmos_db_loader = CosmosDBLoader(container_name=container_name, db_uri=db_uri, credential=credential, database_name=database_name)
    # create the container if it doesn't exist
    cosmos_db_loader.create_container()
    # upload the data to the container
    cosmos_db_loader.upload_data(data_file_path)
