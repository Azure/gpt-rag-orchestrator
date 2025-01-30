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
from shared.cosmo_data_loader import CosmosDBLoader
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

# get company name from cosmos db
COMPANY_NAME = CosmosDBLoader(container_name="companyAnalysis").get_company_list()

CURATION_REPORT_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/generate/curation'

EMAIL_ENDPOINT = f'{os.environ["WEB_APP_URL"]}/api/reports/digest'

STRIPE_SUBSCRIPTION_ENDPOINT = (
    f'{os.environ["WEB_APP_URL"]}/api/subscriptions/<subscription_id>/tiers'
)

TIMEOUT_SECONDS = 300

MAX_RETRIES = 3


class ReportType(str, Enum):
    """Enum for report types to ensure type safety"""

    MONTHLY_ECONOMICS = "Monthly_Economics"
    ECOMMERCE = "Ecommerce"
    HOME_IMPROVEMENT = "Home_Improvement"
    COMPANY_ANALYSIS = "Company_Analysis"


class ReportConfig(BaseModel):
    """Configuration for report generation"""

    report_topic: ReportType
    company_name: Optional[str] = None
    timeout: int = Field(default=TIMEOUT_SECONDS)

    def to_payload(self) -> Dict:
        """Convert config to API payload"""

        payload = {"report_topic": self.report_topic}
        if self.company_name:
            payload["company_name"] = self.company_name
        return payload


class ReportResult(BaseModel):
    """Model for report processing results"""

    report_type: ReportType
    company_name: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    blob_link: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def report_name(self) -> str:
        """Generate a formmated report name"""
        return (
            f"{self.report_type}_{self.company_name}"
            if self.company_name
            else str(self.report_type)
        )


class EmailPayload(BaseModel):
    """Model for email request payload"""

    blob_link: str
    email_subject: str = "Sales Factory Monthly Report"
    recipients: List[str]
    save_email: str = "yes"


def generate_report(
    report_topic: str, company_name: Optional[str] = None
) -> Optional[Dict]:
    """Generate a report and return the response if successful"""

    payload = {"report_topic": report_topic}

    if payload["report_topic"] == "Company_Analysis":
        if not company_name:
            logger.error(f"Company name is required for Company Analysis report")
            raise CompanyNameRequiredError(
                "Company name is required for Company Analysis report"
            )
        else:
            logger.info(f"Company name is {company_name}")
            payload["company_name"] = company_name

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _make_report_request():
        logger.debug(f"Sending request to generate report for {report_topic}")
        response = requests.post(
            CURATION_REPORT_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        logger.debug(
            f"Received response for report generation request for {report_topic}"
        )
        return response.json()

    try:
        report_response = _make_report_request()
        logger.info(f"Report generation response for {report_topic}: {report_response}")
        return report_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to generate report for {report_topic}: {str(e)}")
        return None


def send_report_email(
    blob_link: str,
    report_name: str,
    organization_name: str,
    email_list: List[str],
) -> bool:
    """Send email with report link and return success status"""

    email_subject = f"{organization_name} Monthly Report"

    email_payload = {
        "blob_link": blob_link,
        "email_subject": email_subject,
        "recipients": email_list,
        "save_email": "yes",
    }

    try:
        logger.debug(
            f"Sending email for report {report_name} with blob link {blob_link}"
        )
        response = requests.post(
            EMAIL_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=email_payload,
            timeout=TIMEOUT_SECONDS,
        )
        response_json = response.json()
        logger.info(f"Email response for {report_name}: {response_json}")

        if response_json.get("status") == "error":
            raise requests.exceptions.RequestException(response_json.get("message"))

        return response_json.get("status") == "success"
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send email for {report_name}: {str(e)}")
        return False


def process_single_report(
    report_type: ReportType,
    company_name: Optional[str] = None,
) -> ReportResult:
    """Process a single report generation and email sending"""

    if company_name:
        logger.info(f"Processing report {report_type} for {company_name}")
    else:
        logger.info(f"Processing report {report_type}")

    try:
        config = ReportConfig(report_topic=report_type, company_name=company_name)

        response_json = generate_report(
            report_topic=config.report_topic, company_name=config.company_name
        )

        if not response_json or response_json.get("status") != "success":
            return ReportResult(
                report_type=report_type,
                company_name=company_name,
                success=False,
                error_message=response_json.get("message"),
            )

        blob_link = response_json.get("report_url")
        if not blob_link:
            return ReportResult(
                report_type=report_type,
                company_name=company_name,
                success=False,
                error_message="Failed to extract blob link",
            )

        return ReportResult(
            report_type=report_type,
            company_name=company_name,
            blob_link=blob_link,
            success=True,
            error_message=None,
        )

    except Exception as e:
        logger.exception(f"Error processing report {report_type}")
        return ReportResult(
            report_type=report_type,
            company_name=company_name,
            success=False,
            error_message=str(e),
        )


def process_company_reports(
    report_type: ReportType, companies: List[str]
) -> List[ReportResult]:
    """Process reports for multiple companies"""
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda company: process_single_report(report_type, company), companies
            )
        )
    return results


def check_subscription_statuses(orgs: List[Dict]) -> List[Dict]:
    """Check if the subscription is active and it has financial assistant tier"""

    organizations = []

    if len(orgs) == 0:
        logger.error("No active organizations found")
        return organizations

    def check_subscription_status(subscription_id: str) -> dict:
        try:
            response = requests.get(
                STRIPE_SUBSCRIPTION_ENDPOINT.replace(
                    "<subscription_id>", subscription_id
                ),
                timeout=TIMEOUT_SECONDS,
                headers={
                    "Content-Type": "application/json",
                    "X-MS-CLIENT-PRINCIPAL-ID": "00000000-0000-0000-0000-000000000000",
                },
            )
            response_json = response.json()

            if "subscriptionData" not in response_json:
                raise requests.exceptions.RequestException(
                    "Subscription data not found in response"
                )

            if "status" not in response_json["subscriptionData"]:
                raise requests.exceptions.RequestException(
                    "Subscription status not found in response"
                )

            if response_json["subscriptionData"]["status"] != "active":
                raise requests.exceptions.RequestException("Subscription is not active")

            if "subscriptionTiers" not in response_json:
                raise requests.exceptions.RequestException(
                    "Subscription tiers not found in response"
                )

            return {
                "subscription_id": subscription_id,
                "tier": response_json["subscriptionTiers"],
            }

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to check subscription status for {subscription_id}: {str(e)}"
            )
            return {"subscription_id": subscription_id, "tier": []}

    for org in orgs:
        logger.info(f"Checking subscription status for {org['name']}")
        sub_status = check_subscription_status(org["subscriptionId"])
        if len(sub_status["tier"]) > 0 and (
            "Financial Assistant" in sub_status["tier"]
            or "Premium + Financial Assistant" in sub_status["tier"]
        ):
            organizations.append(org)

    return organizations


def main(mytimer: func.TimerRequest) -> None:
    """Main function to process monthly reports"""
    start_time = datetime.now(timezone.utc)

    logger.info(f"Monthly report generation started at {start_time}")

    orgs_db_manager = CosmosDBLoader(container_name="organizations")
    users_db_manager = CosmosDBLoader(container_name="users")

    organizations = check_subscription_statuses(orgs_db_manager.get_organizations())
    organization_ids = [org["id"] for org in organizations]
    users_by_org = users_db_manager.get_users_by_organizations(organization_ids)

    logger.info(f"Found organizations: {users_by_org}")

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

        for organization, email_list in users_by_org.items():
            organization_name = next(
                org["name"] for org in organizations if org["id"] == organization
            )
            logger.info(
                f"Email list for organization {organization_name}: {email_list}"
            )

            for successful_report in successful_reports:
                report_type = successful_report.report_type
                blob_link = successful_report.blob_link
                company_name = successful_report.company_name

                report_name = (
                    f"{report_type}_{company_name}"
                    if company_name
                    else str(report_type)
                )

                logger.info(
                    f"Sending report {report_name} for organization: {organization_name}"
                )

                logger.info(f"Sending email for report {report_name}")

                if send_report_email(
                    blob_link, report_name, organization_name, email_list
                ):
                    logger.info(f"Report {report_name} sent successfully")
                else:
                    logger.error(f"Failed to send email for {report_name}")
                    logger.error(f"Report with blob link {blob_link} was not sent")

    except Exception as e:
        logger.exception("An error occurred during report processing")

    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time
        logger.info(
            f"Monthly report generation completed at {end_time}. Duration: {duration}"
        )
