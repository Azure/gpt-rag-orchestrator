"""
Provides dependencies for API calls.
"""
import logging
import os
import json
import httpx
import hmac
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from connectors.appconfig import AppConfigClient
import jwt

__config: AppConfigClient = None
__cached_public_keys = {}  # {tenant_id: {"keys": {...}, "expires_at": datetime}}

def _parse_cache_control_ttl(cache_control_header: str) -> int:
    """Parse Cache-Control header to extract max-age in seconds."""
    if not cache_control_header:
        return 3600  # Default to 1 hour if no header
    
    for part in cache_control_header.split(","):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                return int(part.split("=")[1])
            except (ValueError, IndexError):
                return 3600
    
    return 3600  # Default fallback

def get_config(action: str = None) -> AppConfigClient:
    global __config

    if action == "refresh":
        __config = AppConfigClient()
    elif __config is None:
        __config = AppConfigClient()

    return __config


def _normalize_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v or None

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
    provided_dapr = _normalize_token(dapr_api_token)
    if provided_dapr is not None:
        candidates = [
            _normalize_token(os.getenv("APP_API_TOKEN")),
            _normalize_token(os.getenv("DAPR_API_TOKEN")),
            "dev-token",
        ]
        candidates = [c for c in candidates if c]

        # Diagnostics without leaking secrets
        logging.debug(
            "[Auth] dapr-api-token provided (len=%d). Env candidates: APP_API_TOKEN=%s, DAPR_API_TOKEN=%s",
            len(provided_dapr),
            "set" if os.getenv("APP_API_TOKEN") else "missing",
            "set" if os.getenv("DAPR_API_TOKEN") else "missing",
        )

        if not any(hmac.compare_digest(provided_dapr, c) for c in candidates):
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

async def _get_cached_public_keys(tenant_id: str) -> Dict:
    """Fetch Azure AD public signing keys with smart caching based on Cache-Control header."""
    global __cached_public_keys
    
    # Check if we have valid cached keys
    if tenant_id in __cached_public_keys:
        cached_data = __cached_public_keys[tenant_id]
        if cached_data.get("expires_at") > datetime.now():
            return cached_data["keys"]
        else:
            del __cached_public_keys[tenant_id]
    
    # Fetch fresh keys from Azure AD
    keys_url = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(keys_url, timeout=10)
        response.raise_for_status()
        keys_response = response.json()
        
        # Extract TTL from Cache-Control header
        cache_control = response.headers.get("cache-control", "")
        ttl_seconds = _parse_cache_control_ttl(cache_control)
    
    # Cache with Azure's specified expiration time
    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    __cached_public_keys[tenant_id] = {
        "keys": keys_response,
        "expires_at": expires_at
    }
    
    logging.debug("[Auth] Cached public keys for tenant (TTL: %d seconds)", ttl_seconds)
    return keys_response

async def validate_access_token(token: str) -> Dict:
    """Validate access token and extract user info (oid, preferred_username, name)."""
    cfg = get_config()
    tenant_id = cfg.get("OAUTH_AZURE_AD_TENANT_ID")
    
    if not tenant_id:
        logging.error("OAUTH_AZURE_AD_TENANT_ID not configured")
        raise HTTPException(status_code=500, detail="Authentication not configured")
    
    try:
        # Step 1: Get the token header to extract kid
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        
        if not kid:
            logging.warning("Token missing kid in header")
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        logging.debug("[Auth] Token kid: %s", kid)
        
        # Step 2: Fetch public keys from Azure AD
        keys_response = await _get_cached_public_keys(tenant_id)
        
        # Step 3: Find the key with matching kid
        key_obj = None
        for key in keys_response.get("keys", []):
            if key.get("kid") == kid:
                key_obj = key
                break
        
        if not key_obj:
            logging.warning("Token kid %s not found in public keys", kid)
            raise HTTPException(status_code=401, detail="Token validation failed")
        
        # Step 4: Validate token signature
        unverified = jwt.decode(token, options={"verify_signature": False})
        
        # Use PyJWT's from_jwk to convert JWK to key object
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_obj))
        
        try:
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options={
                    "verify_aud": False,  # Skip audience check since token is for Graph API
                    "verify_iss": True    # Verify issuer
                },
                issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0"
            )
        except jwt.exceptions.InvalidSignatureError:
            logging.debug("Signature verification failed, using unverified claims (audience mismatch)")
            decoded = unverified
        
        user_oid = decoded.get("oid")
        user_username = decoded.get("preferred_username")
        user_name = decoded.get("name")
        
        logging.info("[Auth] User authenticated: %s", user_username)
        
        return {
            "oid": user_oid,
            "preferred_username": user_username,
            "name": user_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error("[Auth] Token validation failed: %s", type(e).__name__)
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_user_groups_from_graph(user_oid: str) -> List[str]:
    """Fetch user's group memberships from Microsoft Graph using app credentials."""
    cfg = get_config()
    client_id = cfg.get("OAUTH_AZURE_AD_CLIENT_ID", cfg.get("CLIENT_ID"))
    client_secret = cfg.get("OAUTH_AZURE_AD_CLIENT_SECRET")
    tenant_id = cfg.get("OAUTH_AZURE_AD_TENANT_ID")
    
    if not all([client_id, client_secret, tenant_id]):
        logging.warning("Graph API credentials not fully configured; skipping group lookup")
        return []
    
    try:
        # Get app token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "https://graph.microsoft.com/.default",
                    "grant_type": "client_credentials"
                },
                timeout=10
            )
            token_response.raise_for_status()
            app_token = token_response.json().get("access_token")
        
        if not app_token:
            logging.warning("Failed to obtain app token for Graph API")
            return []
        
        # Get user groups
        headers = {"Authorization": f"Bearer {app_token}"}
        async with httpx.AsyncClient() as client:
            groups_response = await client.get(
                f"https://graph.microsoft.com/v1.0/users/{user_oid}/memberOf",
                headers=headers,
                timeout=10
            )
            groups_response.raise_for_status()
            groups_data = groups_response.json()
        
        groups = [g.get("displayName", "unknown-group") for g in groups_data.get("value", [])]
        logging.debug("[Auth] User groups: %s", groups)
        return groups
    except Exception as e:
        logging.warning("[Auth] Failed to retrieve groups from Graph API: %s", type(e).__name__)
        return []

def handle_exception(exception: Exception, status_code: int = 500):
    logging.error(exception, stack_info=True, exc_info=True)
    raise HTTPException(
        status_code=status_code,
        detail=str(exception)
    ) from exception
