import os
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

_logger = logging.getLogger("appinsights")
_conn = os.getenv("APPINSIGHTS_CONN")
if _conn:
    _logger.addHandler(AzureLogHandler(connection_string=_conn))
    _logger.setLevel(logging.INFO)


def log_event(event: str, **props):
    if _conn:
        _logger.info(event, extra={"custom_dimensions": props})
