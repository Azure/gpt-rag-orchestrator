import datetime
import logging
import os
from . import monitoring
import azure.functions as func

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

def main(monitor: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logging.info('[monitoring] python timer trigger function ran at %s', utc_timestamp)

    if monitor.past_due:
        logging.info('[monitoring] the timer is past due!')

    monitoring.run()