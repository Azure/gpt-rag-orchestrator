import logging
import os
import requests 
from datetime import datetime, timezone 
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import azure.functions as func
from azure.cosmos import CosmosClient
# logger setting 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

""" 
Task: 
1. This is a weekly scheduler that will run every week on monday at 13:00 PM UTC -> 8:00 AM EST
2. It will loop through all the reports in the WEEKLY_REPORTS list 
3. Then it will create a report for each weekly report type using the /api/reports/generate/curation endpoint 
4. Once done, the report will be uploaded to blob storage 

To do:
5. trigger the email function to an email to send to admin the link to the report that has been created 
"""


WEEKLY_REPORTS = ['Weekly_Economics'] 

CURATION_REPORT_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/generate/curation'

EMAIL_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/digest'

TIMEOUT_SECONDS = 300

class CosmoDBManager:
    def __init__(self, container_name: str = 'subscription_emails', 
                 db_uri: str = os.environ['AZURE_COSMOS_ENDPOINT'], 
                 credential: str = os.environ['AZURE_COSMOS_KEY'], 
                 database_name: str = os.environ['AZURE_DB_NAME']):
        self.container_name = container_name
        self.db_uri = db_uri
        self.credential = credential
        self.database_name = database_name

        if not all([self.container_name, self.db_uri, self.credential, self.database_name]):
            raise ValueError("Missing required environment variables for Cosmos DB connection")

        self.client = CosmosClient(url=self.db_uri, credential=self.credential, consistency_level="Session")
        self.database = self.client.get_database_client(self.database_name)
        self.container = self.database.get_container_client(self.container_name)
    
    def get_email_list(self) -> List[str]:
        query = "SELECT * FROM c"
        items = self.container.query_items(query, enable_cross_partition_query=True)
        email_list: List[str] = []
        for item in items:
            email_list.append(item['email'])
        return email_list
    
MAX_RETRIES = 3

def generate_report(report_topic: str) -> Optional[Dict]:
    """Generate a report and return the response if successful """

    payload = {
        'report_topic': report_topic
    }

    @retry(stop = stop_after_attempt(MAX_RETRIES), wait = wait_exponential(multiplier=1, min=4, max=10))
    def _make_report_request():
        logger.debug(f"Sending request to generate report for {report_topic}")
        response = requests.post(
            CURATION_REPORT_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload, 
            timeout=TIMEOUT_SECONDS
        )
        logger.debug(f"Received response for report generation request for {report_topic}")
        return response.json()

    try:
        report_response = _make_report_request()
        logger.info(f"Report generation response for {report_topic}: {report_response}")
        return report_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to generate report for {report_topic}: {str(e)}")
        return None

def send_report_email(blob_link: str, report_name: str) -> bool:
    """Send email with report link and return success status """
    cosmo_db_manager = CosmoDBManager()
    email_list = cosmo_db_manager.get_email_list()

    email_payload = {
        'report_blob_link': blob_link,
        'email_subject': 'Sales Factory Weekly Report',
        'recipients': email_list,
        'save_email': 'yes'
    }

    try: 
        logger.debug(f"Sending email for report {report_name} with blob link {blob_link}")
        response = requests.post(
            EMAIL_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=email_payload,
            timeout=TIMEOUT_SECONDS
        )
        response_json = response.json()
        logger.info(f"Email response for {report_name}: {response_json}")
        return response_json.get('status') == 'success'
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email for {report_name}: {str(e)}")
        return False

def main(mytimer: func.TimerRequest) -> None: 

    utc_timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(f"Weekly report generation started at {utc_timestamp}")

    for report in WEEKLY_REPORTS:
        logger.info(f"Generating report for {report} at {utc_timestamp}")
        response_json = generate_report(report)

        if not response_json or response_json.get('status') != 'success':
            logger.error(f"Failed to generate report for {report} at {utc_timestamp}")
            continue

        # extract blob link and send email 
        blob_link = response_json.get('blob_link')

        if not blob_link:
            logger.error(f"Failed to extract blob link for {report} at {utc_timestamp}")
            continue 

        if send_report_email(blob_link, report):
            logger.info(f"Report {report} sent successfully at {utc_timestamp}")
        else:
            logger.error(f"Failed to send email for {report} at {utc_timestamp}")

    logger.info(f"Weekly report generation completed at {datetime.now(timezone.utc).isoformat()}")

