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
from datetime import datetime, timezone
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
    DashboardAuthConfigResponse,
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


@router.get("/auth-config", response_model=DashboardAuthConfigResponse)
async def get_auth_config(
    cfg: AppConfigClient = Depends(get_config),
) -> DashboardAuthConfigResponse:
    """Return the MSAL configuration the SPA needs to sign the user in.

    Unauthenticated by design (mirrors ``/version``): the SPA calls this
    before any protected endpoint so it can decide whether to bootstrap MSAL
    and which tenant/client/scope to use. When
    ``OAUTH_AZURE_AD_TENANT_ID`` is not set the response only carries
    ``auth_enabled=false`` so no tenant/client information leaks in dev mode.

    The full authority URL and the API scope are derived server-side so the
    browser never has to concatenate values that must match the app
    registration. ``OAUTH_AZURE_AD_API_SCOPE`` is honored when explicitly set
    for tenants where the scope needs to differ from the default
    ``api://<client-id>/access_as_user``.
    """
    tenant_id = cfg.get_value("OAUTH_AZURE_AD_TENANT_ID", default=None, allow_none=True)
    if not tenant_id:
        return DashboardAuthConfigResponse(auth_enabled=False)

    client_id = cfg.get_value("OAUTH_AZURE_AD_CLIENT_ID", default=None, allow_none=True)
    if not client_id:
        # Auth is half-configured (tenant set, client id missing). Surface the
        # misconfiguration to the SPA rather than pretending auth is off,
        # which would silently let the dashboard load without a sign-in gate.
        raise HTTPException(
            status_code=500,
            detail=(
                "OAUTH_AZURE_AD_TENANT_ID is set but OAUTH_AZURE_AD_CLIENT_ID is "
                "missing; the dashboard cannot bootstrap MSAL until both are configured."
            ),
        )

    api_scope = cfg.get_value(
        "OAUTH_AZURE_AD_API_SCOPE", default=None, allow_none=True
    ) or f"api://{client_id}/access_as_user"

    return DashboardAuthConfigResponse(
        auth_enabled=True,
        client_id=client_id,
        tenant_id=tenant_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        api_scope=api_scope,
    )


_MAX_RANGE_DAYS = 365


def _parse_iso_date(value: str, field: str, *, end_of_day: bool = False) -> int:
    """Parse a YYYY-MM-DD (or full ISO) value into UTC epoch seconds.

    For a bare ``YYYY-MM-DD`` the result is start-of-day UTC, or end-of-day
    UTC (``23:59:59``) when ``end_of_day`` is True. A full ISO datetime is
    respected verbatim. This is used for the Overview ``to`` query parameter,
    which operators expect to be inclusive (``from=2026-06-15&to=2026-06-19``
    should include conversations created on 2026-06-19) -- the previous
    implementation treated the date as midnight, silently dropping the whole
    last day (#247 Bug 2).

    Raises 400 on a malformed value so the frontend can surface a helpful
    message instead of a server-side traceback.
    """
    try:
        # Accept either ``YYYY-MM-DD`` or a full ISO datetime; treat naive
        # values as UTC to keep the dashboard consistent with the rest of the
        # admin surface.
        if "T" in value or " " in value:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(value, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field}: expected YYYY-MM-DD or ISO 8601 ({exc})",
        )


@router.get(
    "/overview",
    response_model=DashboardOverview,
    dependencies=[Depends(require_admin)],
    response_model_by_alias=True,
)
async def get_overview(
    days: int = Query(30, ge=1, le=_MAX_RANGE_DAYS),
    from_: Optional[str] = Query(None, alias="from", description="Window start, UTC YYYY-MM-DD"),
    to: Optional[str] = Query(None, description="Window end, UTC YYYY-MM-DD"),
) -> DashboardOverview:
    """Return aggregated counts and a daily conversations series.

    The chart, Engagement panel, and Active users card follow the selected
    window. When ``from``/``to`` are omitted the trailing ``days`` window is
    used (default 30). When provided, both must be present and the range is
    capped at ``_MAX_RANGE_DAYS`` to avoid unbounded Cosmos scans (#241).
    The four trailing-window KPI cards (Today / 7d / 30d) stay fixed in
    ``fetch_overview``.
    """
    from_ts: Optional[int] = None
    to_ts: Optional[int] = None
    if from_ is not None or to is not None:
        if from_ is None or to is None:
            raise HTTPException(
                status_code=400, detail="Both 'from' and 'to' must be provided together"
            )
        from_ts = _parse_iso_date(from_, "from")
        to_ts = _parse_iso_date(to, "to", end_of_day=True)
        # Today's end-of-day UTC -- any bound past this is operator nonsense
        # (and would surface as an empty chart). Frontend caps the picker at
        # today, but we defend on the API too so a hand-crafted query still
        # gets a 400 (#247 Bug 1).
        today_end = int(
            datetime.now(timezone.utc)
            .replace(hour=23, minute=59, second=59, microsecond=0)
            .timestamp()
        )
        if from_ts > today_end or to_ts > today_end:
            raise HTTPException(
                status_code=400, detail="'from' and 'to' cannot be in the future"
            )
        if from_ts > to_ts:
            raise HTTPException(status_code=400, detail="'from' must be <= 'to'")
        window_days = ((to_ts - from_ts) // 86400) + 1
        if window_days > _MAX_RANGE_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"Range exceeds {_MAX_RANGE_DAYS} days (got {window_days})",
            )
        cache_key = f"overview:{from_ts}:{to_ts}"
    else:
        cache_key = f"overview:days:{days}"

    cached = cache_get(cache_key)
    if cached is not None:
        return DashboardOverview(**cached)

    data = await fetch_overview(days=days, from_ts=from_ts, to_ts=to_ts)
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
    response_model_by_alias=True,
)
async def get_conversation_detail(conversation_id: str) -> DashboardConversationDetail:
    """Return the conversation document with messages reconstructed from ``questions[]``.

    The orchestrator persists user prompts under ``questions`` and stores
    assistant replies on the Azure AI Foundry agent thread, not in Cosmos
    (see ``src/orchestration/orchestrator.py`` lines 145-152). Previously
    the dashboard returned ``messages: []`` for every conversation, which
    rendered as ``user (empty)``/``assistant (empty)`` cards (#247 Bug 4).
    We now project ``questions[]`` into user-role message entries and pass
    ``thread_id`` and ``feedback`` through so the frontend can render a
    friendly Foundry-deep-link note instead of empty cards.
    """
    doc = await read_conversation(conversation_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Reconstruct user turns from ``questions[]`` so the dialog has something
    # readable even though assistant replies live on the Foundry thread.
    reconstructed: List[Dict[str, Any]] = []
    for q in doc.get("questions") or []:
        if not isinstance(q, dict):
            continue
        text = q.get("text") or q.get("question") or ""
        reconstructed.append(
            {
                "role": "user",
                "content": text,
                "question_id": q.get("question_id"),
            }
        )

    payload = dict(doc)
    # Preserve any pre-existing ``messages`` (forward compatibility if a
    # future change starts persisting them) but otherwise use the
    # reconstructed list.
    if not payload.get("messages"):
        payload["messages"] = reconstructed
    payload.setdefault("feedback", doc.get("feedback") or [])
    payload.setdefault("thread_id", doc.get("thread_id"))
    return DashboardConversationDetail(**payload)
