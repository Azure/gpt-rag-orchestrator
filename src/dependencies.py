"""
Provides dependencies for API calls.
"""
import logging
import os
from fastapi import HTTPException, Header
from connectors.appconfig import AppConfigClient

__config: AppConfigClient = None

def get_config(action: str = None) -> AppConfigClient:
    global __config

    if action == "refresh":
        __config = AppConfigClient()
    elif __config is None:
        __config = AppConfigClient()

    return __config

async def validate_auth(
    dapr_api_token: str = Header(None, alias="dapr-api-token"),
    x_api_key: str = Header(None, alias="X-API-KEY")
):
    """
    Authentication dependency (no authorization here):
    1) Prefer dapr-api-token if present; otherwise use X-API-KEY.
    2) Missing or invalid credentials => 401 Unauthorized.
    3) 403 Forbidden should be used only by downstream authorization checks (not here).
    """

    # 1) Check dapr-api-token first if provided
    expected_dapr = os.getenv("APP_API_TOKEN") or "dev-token"
    if dapr_api_token is not None:
        if dapr_api_token != expected_dapr:
            logging.warning("Invalid Dapr token")
            raise HTTPException(status_code=401, detail="Invalid Dapr token")
        return True

    # 2) Fallback to API key if no dapr token
    try:
        expected_api_key = get_config().get("ORCHESTRATOR_APP_APIKEY", default=os.getenv("ORCHESTRATOR_APP_APIKEY"))
    except Exception:
        expected_api_key = os.getenv("ORCHESTRATOR_APP_APIKEY")

    if not x_api_key:
        # Missing credentials -> 401
        raise HTTPException(status_code=401, detail="Missing credentials. Provide dapr-api-token or X-API-KEY")

    if not expected_api_key or x_api_key != expected_api_key:
        logging.error("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

def handle_exception(exception: Exception, status_code: int = 500):
    logging.error(exception, stack_info=True, exc_info=True)
    raise HTTPException(
        status_code=status_code,
        detail=str(exception)
    ) from exception
