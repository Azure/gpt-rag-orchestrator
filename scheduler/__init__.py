import datetime
import logging
import json
import os
from datetime import datetime,UTC
import azure.functions as func
from shared.cosmos_db import was_summarized_today
from shared.cosmo_data_loader import CosmosDBLoader
import requests

class LastRunUpdateError(Exception):
    """Custom exception for when the last run time update fails"""
    def __init__(self, schedule_id: str, original_error: Exception):
        self.schedule_id = schedule_id
        self.original_error = original_error
        super().__init__(f"Failed to update last run time for schedule {schedule_id}: {str(original_error)}")

def trigger_document_fetch(schedule: dict) -> bool:
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
    cosmos_data_loader = CosmosDBLoader(os.getenv('SCHEDULES_CONTAINER'))
    process_and_summarize_url = os.getenv('PROCESS_AND_SUMMARIZE_URL')
    
    start_time = datetime.now(UTC).isoformat()
    
    try:
        if was_summarized_today(schedule):
            logging.info(f"Skipping document fetch for {schedule['companyId']} {schedule['reportType']} as it was already summarized today")
            schedule['summarized_today'] = False # reset the summarized_today flag
            return False

        payload = {
            "equity_id": schedule['companyId'],
            "filing_type": schedule['reportType'],
            "after_date": datetime.now(UTC).strftime('%Y-%m-%d')
        }

        response = requests.post(
            process_and_summarize_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )

        response_json = response.json()
        success_code = response_json.get('code', 500)
        schedule['summarized_today'] = (success_code == 200)

        # Log results
        logging.info(f"Response code: {success_code}")
        
        if schedule['summarized_today']:
            logging.info(f"Successfully triggered document fetch for: {schedule['companyId']} - {schedule['reportType']}")
        elif success_code == 404:
            logging.error(f"No new uploaded documents found for: {schedule['companyId']} - {schedule['reportType']}. Last checked time: {start_time}")
        else:
            logging.error(f"Failed to trigger document fetch for: {schedule['companyId']} - {schedule['reportType']}")

    except Exception as e:
        logging.error(f"Error in trigger_document_fetch: {str(e)}")
        schedule['summarized_today'] = False
        return False
    
    finally:
        # Always update last run time, regardless of outcome
        schedule['lastRun'] = start_time
        try:
            cosmos_data_loader.update_last_run(schedule)
        except Exception as e:
            raise LastRunUpdateError(schedule.get('id', 'unknown'), e)

    return schedule['summarized_today']

def main(timer: func.TimerRequest) -> None:

    # schedule cosmos attributes 
    # id
    # lastRun
    # frequency
    # companyId
    # reportType
    # isActive

    # Main scheduling logic
    cosmos_data_loader = CosmosDBLoader('schedules')
    success_count = 0
    try:
        # get all schedules that are active and have a frequency of twice_a_day
        active_schedules = cosmos_data_loader.get_data(frequency="twice_a_day")
        for schedule in active_schedules:
            logging.info(f"Triggering fetch for schedule {schedule['id']}")
            try:
                success = trigger_document_fetch(schedule)
                if success:
                    success_count += 1  
            except LastRunUpdateError as e:
                logging.error(f"Failed to update schedule last run time: {str(e)}")
                # You might want to implement retry logic here
                continue
        logging.info(f"Successfully triggered fetch for {success_count} schedules")
    except Exception as e:
        logging.error(f"Error in scheduler: {str(e)}")
