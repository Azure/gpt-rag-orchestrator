"""
Provides dependencies for API calls.
"""
import logging
import os
from fastapi import Depends, HTTPException, Header
from connectors.appconfig import AppConfigClient
from fastapi.security import APIKeyHeader   

__config: AppConfigClient = None

def get_config(action: str = None) -> AppConfigClient:
    global __config

    if action == "refresh":
        __config = AppConfigClient()
    elif __config is None:
        __config = AppConfigClient()

    return __config

async def validate_dapr_token(
    dapr_api_token: str = Header(None, alias="dapr-api-token")
):
    expected = os.getenv("APP_API_TOKEN")
    if expected is None:
        # for local development, use a default token
        expected = "dev-token"
    if dapr_api_token != expected:
        logging.warning("Invalid Dapr token")
        raise HTTPException(401, detail="Unauthorized")
    return True

def validate_api_key_header(x_api_key: str = Depends(APIKeyHeader(name='X-API-KEY'))):
    result = x_api_key == get_config().get(f'ORCHESTRATOR_APP_APIKEY')
    
    if not result:
        logging.error('Invalid API key. You must provide a valid API key in the X-API-KEY header.')
        raise HTTPException(
            status_code = 401,
            detail = 'Invalid API key. You must provide a valid API key in the X-API-KEY header.'
        )

def handle_exception(exception: Exception, status_code: int = 500):
    logging.error(exception, stack_info=True, exc_info=True)
    raise HTTPException(
        status_code = status_code,
        detail = str(exception)
    ) from exception
