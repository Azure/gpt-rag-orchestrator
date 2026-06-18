"""Cross-partition Cosmos DB queries used by the admin dashboard.

These helpers intentionally do not filter by ``principal_id`` partition key
because the dashboard is administrative and lists conversations across all
users. They are kept in a separate module from ``cosmosdb.py`` so the
user-scoped helpers stay strictly user-scoped and cannot be accidentally
reused with cross-partition semantics.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from connectors.cosmosdb import get_cosmosdb_client


def _container():
    """Resolve the conversations container client from the shared Cosmos client."""
    cosmos = get_cosmosdb_client()
    container_name = cosmos.cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
    return cosmos._get_container(container_name)


def _epoch_n_days_ago(days: int) -> int:
    """Return the Unix epoch seconds for ``days`` days before now (UTC)."""
    now = datetime.now(timezone.utc).timestamp()
    return int(now - days * 86400)


def _start_of_day_utc(ts: float) -> str:
    """Format a Unix timestamp as YYYY-MM-DD in UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


async def fetch_overview(days: int = 30) -> Dict[str, Any]:
    """Aggregate dashboard counts and a daily series for the last ``days`` days.

    Returns a dict with::

        {
            "total":              <int total non-deleted conversations>,
            "today":              <int created in the last 24h>,
            "last_7_days":        <int created in the last 7d>,
            "last_30_days":       <int created in the last 30d>,
            "active_users":       <int distinct principal_id in the window>,
            "avg_turns":          <float average user-turns per conversation>,
            "conversations_per_day": [{"date": "YYYY-MM-DD", "count": n}, ...],
            "window_days":        <int>,
        }
    """
    container = _container()
    days = max(1, int(days))
    cutoff = _epoch_n_days_ago(days)

    # Single cross-partition pass: pull the fields we need and aggregate in Python.
    # ``_ts`` is in seconds. ``messages`` is an array; we sample ``ARRAY_LENGTH``
    # server-side to avoid shipping full message bodies.
    query = (
        "SELECT c._ts, c.principal_id, ARRAY_LENGTH(c.messages) AS message_count "
        "FROM c "
        "WHERE (NOT IS_DEFINED(c.isDeleted) OR c.isDeleted = false)"
    )

    docs: List[Dict[str, Any]] = []
    # ``partition_key=None`` mirrors the existing cross-partition pattern in
    # ``CosmosDBClient.list_documents`` and replaces the deprecated
    # ``enable_cross_partition_query`` kwarg in modern azure-cosmos SDKs.
    async for item in container.query_items(query=query, partition_key=None):
        docs.append(item)

    total = len(docs)

    now_ts = datetime.now(timezone.utc).timestamp()
    today_cutoff = int(now_ts - 86400)
    week_cutoff = int(now_ts - 7 * 86400)

    today = 0
    last_7 = 0
    last_30 = 0
    per_day: Dict[str, int] = {}
    active_users: set[str] = set()
    turns_in_window: List[int] = []

    for doc in docs:
        ts = doc.get("_ts") or 0
        if ts >= today_cutoff:
            today += 1
        if ts >= week_cutoff:
            last_7 += 1
        if ts >= cutoff:
            last_30 += 1
            pid = doc.get("principal_id")
            if pid:
                active_users.add(str(pid))
            # Count user turns. ``messages`` includes both user and assistant
            # entries; a turn is one round-trip, so divide by 2 and floor.
            msg_count = doc.get("message_count") or 0
            if isinstance(msg_count, (int, float)) and msg_count > 0:
                turns_in_window.append(int(msg_count) // 2)
            day = _start_of_day_utc(ts)
            per_day[day] = per_day.get(day, 0) + 1

    avg_turns = round(sum(turns_in_window) / len(turns_in_window), 2) if turns_in_window else 0.0

    # Emit a dense series so the chart shows zero-days too.
    series: List[Dict[str, Any]] = []
    for i in range(days - 1, -1, -1):
        day = _start_of_day_utc(now_ts - i * 86400)
        series.append({"date": day, "count": per_day.get(day, 0)})

    return {
        "total": total,
        "today": today,
        "last_7_days": last_7,
        "last_30_days": last_30,
        "active_users": len(active_users),
        "avg_turns": avg_turns,
        "conversations_per_day": series,
        "window_days": days,
    }


async def list_conversations(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return paginated conversation summaries across all users, newest first."""
    container = _container()
    skip = max(0, int(skip))
    limit = max(1, min(int(limit), 200))

    base_select = (
        "SELECT c.id, c.name, c.principal_id, c._ts, c.lastUpdated, "
        "ARRAY_LENGTH(c.messages) AS message_count FROM c "
        "WHERE (NOT IS_DEFINED(c.isDeleted) OR c.isDeleted = false)"
    )

    params: List[Dict[str, Any]] = [
        {"name": "@skip", "value": skip},
        {"name": "@limit", "value": limit},
    ]
    if search:
        query = (
            f"{base_select} AND (CONTAINS(LOWER(c.name ?? ''), @search) "
            "OR CONTAINS(LOWER(c.principal_id ?? ''), @search) "
            "OR CONTAINS(LOWER(c.id ?? ''), @search)) "
            "ORDER BY c._ts DESC OFFSET @skip LIMIT @limit"
        )
        params.append({"name": "@search", "value": search.lower()})
    else:
        query = f"{base_select} ORDER BY c._ts DESC OFFSET @skip LIMIT @limit"

    out: List[Dict[str, Any]] = []
    async for item in container.query_items(
        query=query, parameters=params, partition_key=None
    ):
        out.append(item)

    logging.debug("[CosmosDB][admin] listed %d conversations (skip=%d limit=%d)", len(out), skip, limit)
    return out


async def read_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Return the full conversation document by id, regardless of partition."""
    container = _container()
    query = (
        "SELECT * FROM c WHERE c.id = @id "
        "AND (NOT IS_DEFINED(c.isDeleted) OR c.isDeleted = false)"
    )
    params = [{"name": "@id", "value": conversation_id}]
    async for item in container.query_items(
        query=query, parameters=params, partition_key=None
    ):
        return item
    return None


# ---------------------------------------------------------------------------
# Lightweight in-memory cache shared by the dashboard endpoints.
# Keeps the dashboard cheap to refresh without putting load on Cosmos for
# every page render. TTL is intentionally short so admin views stay fresh.
# ---------------------------------------------------------------------------

_CACHE_TTL = 60.0  # seconds
_cache: Dict[str, tuple[float, Any]] = {}


def cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if (time.monotonic() - stored_at) > _CACHE_TTL:
        _cache.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.monotonic(), value)


def cache_clear() -> None:
    """Drop the entire cache. Used by tests."""
    _cache.clear()
