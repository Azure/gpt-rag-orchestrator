"""Admin dashboard API.

Exposes a small read-only surface used by the SPA mounted at ``/dashboard``.
All routes live under ``/api/dashboard`` and are protected by an admin gate:

* When ``OAUTH_AZURE_AD_TENANT_ID`` is configured (auth is on), the caller
  must present a valid bearer token whose ``roles`` claim contains
  ``"Admin"``. The token must be obtained with the ``api://<client_id>/...``
  scope so the app-role claim is included.
* When auth is off, the gate is a no-op (the dashboard mirrors the
  development experience of the rest of the orchestrator).

The router is only mounted when ``ENABLE_DASHBOARD`` is true, so the routes
do not exist at all in default deployments.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from connectors.appconfig import AppConfigClient
from connectors.cosmosdb_admin import (
    cache_get,
    cache_set,
    fetch_overview,
    list_conversations,
    read_conversation,
)
from dependencies import get_config, validate_access_token
from schemas import (
    DashboardConversationDetail,
    DashboardConversationListResponse,
    DashboardConversationSummary,
    DashboardOverview,
    DashboardVersionResponse,
)
from util.jwt_utils import extract_bearer_token


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Admin gate
# ---------------------------------------------------------------------------

async def require_admin(
    authorization: Optional[str] = Header(None),
    cfg: AppConfigClient = Depends(get_config),
) -> None:
    """Require the caller to be an Admin when auth is enabled.

    When ``OAUTH_AZURE_AD_TENANT_ID`` is not configured the orchestrator is
    running in unauthenticated (dev) mode, so the gate is a no-op. When it is
    configured, a bearer token is required and the resolved claims must
    include ``Admin`` in the ``roles`` array.
    """
    tenant_id = cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", default=None, allow_none=True)
    if not tenant_id:
        return  # auth off, dashboard open like the rest of the app in dev

    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        claims = await validate_access_token(token)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("[dashboard] token validation failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    roles = claims.get("roles") or []
    if "Admin" not in roles:
        raise HTTPException(status_code=403, detail="Admin role required")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/version", response_model=DashboardVersionResponse)
async def get_version() -> DashboardVersionResponse:
    """Return the orchestrator version string for the dashboard header chip.

    Unauthenticated by design (mirrors ``GET /version`` used by other
    surfaces) so the SPA can render its shell before the user signs in.
    """
    from pathlib import Path

    version_file = Path(__file__).resolve().parent.parent.parent / "VERSION"
    try:
        version = version_file.read_text().strip() or "unknown"
    except OSError:
        version = "unknown"
    return DashboardVersionResponse(version=version)


@router.get(
    "/overview",
    response_model=DashboardOverview,
    dependencies=[Depends(require_admin)],
)
async def get_overview(days: int = Query(30, ge=1, le=180)) -> DashboardOverview:
    """Return aggregated counts and a daily conversations series."""
    cache_key = f"overview:{days}"
    cached = cache_get(cache_key)
    if cached is not None:
        return DashboardOverview(**cached)

    data = await fetch_overview(days=days)
    cache_set(cache_key, data)
    return DashboardOverview(**data)


@router.get(
    "/conversations",
    response_model=DashboardConversationListResponse,
    dependencies=[Depends(require_admin)],
)
async def get_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None, max_length=200),
) -> DashboardConversationListResponse:
    """Return a page of conversations across all users, newest first."""
    # Fetch one extra row to detect whether a next page exists.
    rows = await list_conversations(skip=skip, limit=limit + 1, search=search)
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]
    items = [DashboardConversationSummary(**row) for row in rows]
    return DashboardConversationListResponse(
        conversations=items, has_more=has_more, skip=skip, limit=limit
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=DashboardConversationDetail,
    dependencies=[Depends(require_admin)],
)
async def get_conversation_detail(conversation_id: str) -> DashboardConversationDetail:
    """Return the full conversation document, including messages."""
    doc = await read_conversation(conversation_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return DashboardConversationDetail(**doc)
