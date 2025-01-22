import logging
import os
import requests 
from datetime import datetime, timezone 
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import azure.functions as func  
from .exceptions import CompanyNameRequiredError
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field 
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from enum import Enum
# logger setting 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


""" 
Monthly Scheduler: 
1. The trigger will run at 8am EST every 1st date of the month 
2. It will loop through all the reports in the MONTHLY_REPORTS list 
3. Then it will create a report for each monthly report type using the /api/reports/generate/curation endpoint 
4. Once done, the report will be uploaded to blob storage 

To do:
5. trigger the email function to an email to send to admin the link to the report that has been created 
6. automatically send the report to user using the blob link got from the curation endpoint 
check other way to use endpoint 

"""

MONTHLY_REPORTS = ['Monthly_Economics', 'Ecommerce', "Home_Improvement", "Company_Analysis"]

COMPANY_NAME = ["Home Depot", "Lowes"]

CURATION_REPORT_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/generate/curation'

EMAIL_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/digest'

TIMEOUT_SECONDS = 300

MAX_RETRIES = 3

class CosmoDBManager:
    def __init__(self, container_name: str, 
                 db_uri: str, 
                 credential: str, 
                 database_name: str):
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
        query = "SELECT * FROM c where c.isActive = true"
        items = self.container.query_items(query, enable_cross_partition_query=True)
        email_list: List[str] = []
        for item in items:
            if "email" in item:
                email_list.append(item['email'])
        return email_list

def generate_report(report_topic: str, company_name: Optional[str] = None) -> Optional[Dict]:
    """Generate a report and return the response if successful """

    payload = {
        'report_topic': report_topic
    }

    if payload['report_topic'] == "Company_Analysis":
        if not company_name:
            logger.error(f"Company name is required for Company Analysis report")
            raise CompanyNameRequiredError("Company name is required for Company Analysis report")
        else:
            logger.info(f"Company name is {company_name}")
            payload['company_name'] = company_name

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

    container_name = 'subscription_emails'
    db_uri = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/" if os.environ.get('AZURE_DB_ID') else None
    credential = DefaultAzureCredential()
    database_name = os.environ.get('AZURE_DB_NAME') if os.environ.get('AZURE_DB_NAME') else None

    cosmo_db_manager = CosmoDBManager(
        container_name=container_name,
        db_uri=db_uri,
        credential=credential,
        database_name=database_name
    )
    email_list = cosmo_db_manager.get_email_list()

    email_payload = {
        'blob_link': blob_link,
        'email_subject': 'Sales Factory Monthly Report',
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

        if response_json.get('status') == 'error':
            raise requests.exceptions.RequestException(response_json.get('message'))
        
        return response_json.get('status') == 'success'
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email for {report_name}: {str(e)}")
        return False



class ReportType(str, Enum):
    """Enum for report types to ensure type safety """

    MONTHLY_ECONOMICS = "Monthly_Economics"
    ECOMMERCE = "Ecommerce"
    HOME_IMPROVEMENT = "Home_Improvement"
    COMPANY_ANALYSIS = "Company_Analysis"
    WEEKLY_ECONOMICS = "Weekly_Economics"

class ReportConfig(BaseModel): 
    """Configuration for report generation"""
    
    report_topic: ReportType
    company_name: Optional[str] = None
    timeout: int = Field(default = TIMEOUT_SECONDS)

    def to_payload(self) -> Dict:
        """Convert config to API payload"""

        payload = {
            "report_topic": self.report_topic
        }
        if self.company_name: 
            payload['company_name'] = self.company_name
        return payload
    
class ReportResult(BaseModel): 
    """Model for report processing results"""
    report_type: ReportType
    company_name: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    blob_link: Optional[str] = None
    timestamp: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))

    @property 
    def report_name(self) -> str: 
        """Generate a formmated report name"""
        return f"{self.report_type}_{self.company_name}" if self.company_name else str(self.report_type)
    
class EmailPayload(BaseModel): 
    """Model for email request payload"""
    blob_link: str
    email_subject: str = "Sales Factory Monthly Report"
    recipients: List[str]
    save_email: str = "yes"

def process_single_report(
        report_type: ReportType, 
        company_name: Optional[str] = None, 
) -> ReportResult: 
    """Process a single report generation and email sending"""
    try: 
        config = ReportConfig(
            report_topic = report_type,
            company_name = company_name
        )

        response_json = generate_report(
            report_topic = config.report_topic, 
            company_name = config.company_name
        )

        if not response_json or response_json.get('status') != 'success': 
            return ReportResult(
                report_type = report_type, 
                company_name = company_name, 
                success = False, 
                error_message = response_json.get('message')
            )

        blob_link = response_json.get('report_url')
        if not blob_link: 
            return ReportResult(
                report_type = report_type, 
                company_name = company_name, 
                success = False, 
                error_message = "Failed to extract blob link"
            )
        
        # create email payload 
        cosmos_db_manager = CosmoDBManager(
            container_name = 'subscription_emails', 
            db_uri = f"https://{os.environ['AZURE_DB_ID']}.documents.azure.com:443/", 
            credential = DefaultAzureCredential(), 
            database_name = os.environ.get('AZURE_DB_NAME')
        )

        email_payload = EmailPayload(
            blob_link = blob_link,
            recipients = cosmos_db_manager.get_email_list()
        )

        report_name = f"{report_type}_{company_name}" if company_name else str(report_type)
        email_success = send_report_email(email_payload.blob_link, report_name)

        return ReportResult(
            report_type=report_type,
            company_name=company_name,
            blob_link=blob_link,
            success=email_success,
            error_message=None if email_success else "Failed to send email"
        )

    except Exception as e:
        logger.exception(f"Error processing report {report_type}")
        return ReportResult(
            report_type=report_type,
            company_name=company_name,
            success=False,
            error_message=str(e)
        )
    

def process_company_reports(report_type: ReportType, 
                            companies: List[str]) -> List[ReportResult]: 
    """Process reports for multiple companies"""
    with ThreadPoolExecutor() as executor: 
        results = list(executor.map(lambda company: process_single_report(report_type, company), companies))
    return results

def main(mytimer: func.TimerRequest) -> None: 
    """Main function to process monthly reports"""
    start_time = datetime.now(timezone.utc)

    logger.info(f"Monthly report generation started at {start_time}")

    all_results: List[ReportResult] = []

    try: 
        for report in ReportType: 
            if report == ReportType.COMPANY_ANALYSIS:
                # process company reports in parallel 
                company_results = process_company_reports(report, COMPANY_NAME)
                all_results.extend(company_results)
            else: 
                # process regular reports 
                result = process_single_report(report)
                all_results.append(result)

        # log summary of results 
        successful_reports = [r for r in all_results if r.success]
        failed_reports = [r for r in all_results if not r.success]

        logger.info(f"Successfully processed {len(successful_reports)} reports")
        for result in successful_reports: 
            logger.info(f"Success: {result.report_name} - {result.blob_link}")

        if failed_reports: 
            logger.error(f"failed to process {len(failed_reports)} reports")
            for result in failed_reports: 
                logger.error(f"Failed: {result.report_name} - {result.error_message}")

    except Exception as e: 
        logger.exception("An error occurred during report processing")

    finally: 
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time 
        logger.info(f"Monthly report generation completed at {end_time}. Duration: {duration}")


