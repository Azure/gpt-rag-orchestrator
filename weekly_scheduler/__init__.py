import logging
import os
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import azure.functions as func
from shared.cosmo_data_loader import CosmosDBLoader
from enum import Enum


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


class ReportType(str, Enum):
    """Enum for report types to ensure type safety"""

    WEEKLY_ECONOMICS = "Weekly_Economics"


WEB_APP_URL = os.getenv("WEB_APP_URL", None)
CURATION_REPORT_ENDPOINT = f'{WEB_APP_URL}/api/reports/generate/curation'
EMAIL_ENDPOINT = f'{WEB_APP_URL}/api/reports/digest'
STRIPE_SUBSCRIPTION_ENDPOINT = (
    f'{WEB_APP_URL}/api/subscriptions/<subscription_id>/tiers'
)

TIMEOUT_SECONDS = 300

MAX_RETRIES = 3


def generate_report(report_topic: ReportType) -> Optional[Dict]:
    """Generate a report and return the response if successful"""

    payload = {"report_topic": report_topic,}

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
    report_name: ReportType,
    organization_name: str,
    email_list: List[str],
) -> bool:
    """Send email with report link and return success status"""

    email_subject = f"{organization_name} Weekly Report"

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
    """ Main function to generate weekly reports and send emails """

    # Check if the environment variable is set
    if not WEB_APP_URL:
        logger.error("WEB_APP_URL environment variable not set")
        return
    
    start_time = datetime.now(timezone.utc)
    logger.info(f"Weekly report generation started at {start_time}")

    orgs_db_manager = CosmosDBLoader(container_name="organizations")
    users_db_manager = CosmosDBLoader(container_name="users")

    organizations = check_subscription_statuses(orgs_db_manager.get_organizations())
    organization_ids = [org["id"] for org in organizations]
    users_by_org = users_db_manager.get_users_by_organizations(organization_ids)

    try:
        for report in ReportType:
            logger.info(f"Generating report {report}")
            response_json = generate_report(report)

            if not response_json or response_json.get("status") != "success":
                logger.error(f"Failed to generate report for {report}")
                continue

            # extract blob link and send email
            blob_link = response_json.get("report_url")

            if not blob_link:
                logger.error(f"Failed to extract blob link for {report}")
                continue

            for organization, email_list in users_by_org.items():
                organization_name = next(
                    org["name"] for org in organizations if org["id"] == organization
                )
                logger.info(
                    f"Generating report {report} for organization: {organization_name}"
                )
                logger.info(
                    f"Email list for organization {organization_name}: {email_list}"
                )

                if send_report_email(blob_link, report, organization_name, email_list):
                    logger.info(f"Report {report} sent successfully")
                else:
                    logger.error(f"Failed to send email for {report}")
                    logger.error(f"Report with blob link {blob_link} was not sent")
    except Exception as e:
        logger.exception("An error occurred during report processing")

    finally:
        end_time = datetime.now(timezone.utc)
        duration = end_time - start_time 
        logger.info(
            f"Weekly report generation completed at {end_time}. Duration: {duration}"
        )
