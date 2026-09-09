"""Scope-aware delegated token exchange shared by Search and Foundry IQ."""

import hashlib
import json
import logging
import time
from enum import Enum
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import aiohttp

from dependencies import get_config


_obo_cache: dict[str, Any] = {}
_MAX_OBO_CACHE_ENTRIES = 256


class RetrievalAuthorizationMode(str, Enum):
    """Internal request policy; service identity never substitutes for a user."""

    USER_REQUIRED = "user_required"
    SERVICE_ONLY = "service_only"


def resolve_retrieval_authorization(
    assertion: Optional[str], allow_anonymous: bool, source_requires_user: bool = False,
) -> RetrievalAuthorizationMode:
    if not isinstance(allow_anonymous, bool) or not isinstance(source_requires_user, bool):
        raise ValueError("Invalid retrieval authorization policy")
    if assertion is not None and not isinstance(assertion, str):
        raise ValueError("Invalid retrieval assertion")
    if (assertion and assertion.strip()) or not allow_anonymous or source_requires_user:
        return RetrievalAuthorizationMode.USER_REQUIRED
    return RetrievalAuthorizationMode.SERVICE_ONLY


async def require_retrieval_token(
    mode: RetrievalAuthorizationMode,
    callback: Optional[Callable[[], Awaitable[Optional[str]]]],
) -> Optional[str]:
    """Preflight before document access; errors and cancellation propagate."""
    if not isinstance(mode, RetrievalAuthorizationMode):
        raise ValueError("Invalid retrieval authorization mode")
    logging.info("retrieval_authorization_mode=%s", mode.value)
    if mode is RetrievalAuthorizationMode.SERVICE_ONLY:
        return None
    token = await callback() if callback is not None else None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("Required retrieval authorization unavailable")
    return token


def classify_retrieval_error(error: Any) -> tuple[int, str]:
    """Return the existing safe auth/error log level and marker."""
    msg = str(error) if error is not None else ""
    if "401" in msg or "403" in msg:
        return logging.ERROR, "[Retrieval][AUTH_FAILURE]"
    return logging.WARNING, "[Retrieval][ERROR]"


async def acquire_obo_token(
    api_access_token: Optional[str],
    scope: str,
    allow_anonymous: bool = False,
) -> Optional[str]:
    """Acquire a delegated token; preserve strict and anonymous caller semantics."""
    scope = (scope or "").strip()
    if not scope:
        raise ValueError("OBO scope must not be empty")
    if not api_access_token:
        if allow_anonymous:
            return None
        raise RuntimeError("Missing incoming user access token for OBO exchange")

    fp = hashlib.sha256(api_access_token.encode()).hexdigest()
    cache_key = f"{fp}:{scope}"
    cached = _obo_cache.get(cache_key)
    if cached and time.time() < cached.get("expires_at", 0):
        return cached["token"]

    cfg = get_config()
    tenant_id = (cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", allow_none=True) or "").strip() or None
    client_id = (cfg.get_value("OAUTH_AZURE_AD_CLIENT_ID", allow_none=True) or "").strip() or None
    client_secret = (cfg.get_value("OAUTH_AZURE_AD_CLIENT_SECRET", allow_none=True) or "").strip() or None

    if not tenant_id or not client_id or not client_secret:
        logging.warning("[OBO] Missing Entra config for OBO (tenant=%s client=%s secret=%s)",
                        "set" if tenant_id else "missing", "set" if client_id else "missing", "set" if client_secret else "missing")
        if allow_anonymous:
            return None
        raise RuntimeError("OBO configuration is incomplete")

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    form = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "requested_token_use": "on_behalf_of",
        "scope": scope,
        "assertion": api_access_token,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=form) as resp:
            raw = await resp.text()
            if resp.status >= 400:
                level, marker = classify_retrieval_error(resp.status)
                logging.log(
                    level,
                    "%s OBO token exchange failed (status=%d scope_fp=%s)",
                    marker,
                    resp.status,
                    hashlib.sha256(scope.encode()).hexdigest()[:12],
                    extra={
                        "retrieval_status": resp.status,
                        "retrieval_credential_type": "obo",
                    },
                )
                if allow_anonymous:
                    return None
                raise RuntimeError(f"OBO token exchange failed: status={resp.status}")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logging.error("[OBO] Non-JSON response from token endpoint")
                if allow_anonymous:
                    return None
                raise RuntimeError(
                    "OBO token endpoint returned an invalid response"
                ) from exc
            token = data.get("access_token")
            if token:
                ttl = int(data.get("expires_in", 0))
                now = time.time()
                expired_keys = [
                    key
                    for key, entry in _obo_cache.items()
                    if now >= entry.get("expires_at", 0)
                ]
                for key in expired_keys:
                    _obo_cache.pop(key, None)
                while len(_obo_cache) >= _MAX_OBO_CACHE_ENTRIES:
                    _obo_cache.pop(next(iter(_obo_cache)))
                _obo_cache[cache_key] = {
                    "token": token,
                    "expires_at": now + max(0, ttl - 30),
                }
                logging.info("[OBO] Acquired delegated token")
                return token
            if allow_anonymous:
                return None
            raise RuntimeError("OBO token endpoint response missing access_token")
