import datetime
import logging
import os
from typing import List, Dict
from datetime import datetime, timezone, timedelta
import azure.functions as func
from shared.cosmos_db import was_summarized_today
from shared.cosmo_data_loader import CosmosDBLoader
import requests

# logger setting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_APP_URL = os.getenv("WEB_APP_URL", None)
PROCESS_AND_SUMMARIZE_ENDPOINT = f"{WEB_APP_URL}/api/SECEdgar/financialdocuments/process-and-summarize"
EMAIL_ENDPOINT = f'{WEB_APP_URL}/api/reports/digest'
STRIPE_SUBSCRIPTION_ENDPOINT = f"{WEB_APP_URL}/api/subscriptions/<subscription_id>/tiers"

TIMEOUT_SECONDS = 300


class LastRunUpdateError(Exception):
    """Custom exception for when the last run time update fails"""
    def __init__(self, schedule_id: str, original_error: Exception):
        self.schedule_id = schedule_id
        self.original_error = original_error
        super().__init__(
            f"Failed to update last run time for schedule {schedule_id}: {str(original_error)}"
        )

def send_report_email(
    blob_link: str,
    report_name: str,
    organization_name: str,
    email_list: List[str],
) -> bool:
    """Send email with report link and return success status"""

    email_subject = f"{organization_name} {report_name} Summarization Report"

    email_payload = {
        "blob_link": blob_link,
        "email_subject": email_subject,
        "recipients": email_list,
        "save_email": "yes",
        "is_summarization": True,
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


def trigger_document_fetch(
    schedule: dict, schedules_db_manager: CosmosDBLoader
) -> bool:
    """
    Queue the document fetch task based on schedule configuration

    Args:
        schedule (dict): schedule document containing companyId, reportType, frequency, lastRun, and other attributes

    Returns:
        bool: True if document fetch was successful, False otherwise

    Raises:
        LastRunUpdateError: If updating the last run time fails
    """
    # Move these outside the function if possible since they're used across multiple calls

    start_time = datetime.now(timezone.utc).isoformat()

    # Initialize summarization object to return
    summarization = {
        "was_summarized_today": False,
        "equity_name": None,
        "financial_type": None,
        "remote_blob_url": None,
        "summary": None,
    }

    try:
        if was_summarized_today(schedule):
            logging.info(
                f"Skipping document fetch for {schedule['companyId']} {schedule['reportType']} as it was already summarized today"
            )
            schedule["summarized_today"] = False  # reset the summarized_today flag
            return False

        payload = {
            "equity_id": schedule["companyId"],
            "filing_type": schedule["reportType"],
            "after_date": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        }
        # "after_date": datetime.now(timezone.utc).strftime('%Y-%m-%d')

        response = requests.post(
            PROCESS_AND_SUMMARIZE_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )

        success_code = response.status_code
        schedule["summarized_today"] = success_code == 200

        if schedule["summarized_today"]:
            response_json = response.json()
            if "summary_process" not in response_json:
                raise requests.exceptions.RequestException(
                    f"Missing summary_process in response for: {schedule['companyId']} - {schedule['reportType']}"
                )
            else:
                logging.info(
                    f"Successfully triggered document fetch for: {schedule['companyId']} - {schedule['reportType']}"
                )

                summarization["was_summarized_today"] = True
                summarization["equity_name"] = response_json["summary_process"][
                    "equity_name"
                ]
                summarization["financial_type"] = response_json["summary_process"][
                    "financial_type"
                ]
                summarization["remote_blob_url"] = response_json["summary_process"][
                    "remote_blob_url"
                ]
                summarization["summary"] = response_json["summary_process"]["summary"]
        elif success_code == 404:
            raise requests.exceptions.RequestException(
                f"No new documents found for the given equity and report type {schedule['companyId']} - {schedule['reportType']}. Last checked time: {start_time}"
            )
        else:
            raise requests.exceptions.RequestException(
                f"Failed to trigger document fetch for {schedule['companyId']} - {schedule['reportType']}"
            )
    except Exception as e:
        logging.error(f"Error in [trigger_document_fetch]: {str(e)}")
        schedule["summarized_today"] = False
        return summarization
    finally:
        # Always update last run time, regardless of outcome
        schedule["lastRun"] = start_time
        try:
            schedules_db_manager.update_last_run(schedule)
        except Exception as e:
            raise LastRunUpdateError(schedule.get("id", "unknown"), e)

    return summarization


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


def main(timer: func.TimerRequest) -> None:
    """ Main entry point for the function """
    
    # Check if the environment variable is set
    if not WEB_APP_URL:
        logger.error("WEB_APP_URL environment variable not set")
        return

    # Main scheduling logic
    schedules_db_manager = CosmosDBLoader(container_name="schedules")
    orgs_db_manager = CosmosDBLoader(container_name="organizations")
    users_db_manager = CosmosDBLoader(container_name="users")

    organizations = check_subscription_statuses(orgs_db_manager.get_organizations())
    organization_ids = [org["id"] for org in organizations]
    users_by_org = users_db_manager.get_users_by_organizations(organization_ids)

    logger.info(f"Found organizations: {users_by_org}")

    success_count = 0
    try:
        # get all schedules that are active and have a frequency of twice_a_day
        active_schedules = schedules_db_manager.get_data(frequency="twice_a_day")

        for schedule in active_schedules:
            logging.info(
                f"Triggering fetch for schedule {schedule['id']} {schedule['companyId']} {schedule['reportType']}"
            )
            try:
                summarization = trigger_document_fetch(schedule, schedules_db_manager)

                if (summarization["was_summarized_today"]):
                    success_count += 1
                    # send email to all users in the organization
                    for organization, email_list in users_by_org.items():
                        organization_name = next(
                            org["name"]
                            for org in organizations
                            if org["id"] == organization
                        )

                        blob_link = summarization["remote_blob_url"]
                        report_name = f"{summarization['equity_name']} {summarization['financial_type']}"

                        if send_report_email(
                            blob_link, report_name, organization_name, email_list
                        ):
                            logger.info(f"Report {report_name} sent successfully")
                        else:
                            logger.error(f"Failed to send email for {report_name}")
                            logger.error(f"Report with blob link {blob_link} was not sent")

            except LastRunUpdateError as e:
                logging.error(f"Failed to update schedule last run time: {str(e)}")
                # We might want to implement retry logic here
                continue
        logging.info(f"Successfully triggered fetch for {success_count} schedules")
    except Exception as e:
        logging.error(f"Error in scheduler: {str(e)}")
