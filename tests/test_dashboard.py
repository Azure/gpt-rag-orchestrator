"""Tests for the admin dashboard API.

These cover the admin gate (auth off, missing token, wrong role, right
role), the cross-partition Cosmos helpers, and the FastAPI router endpoints
(pagination boundaries + empty store + 404 detail).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncIter:
    """Async-iterable mock that yields a fixed list (CosmosDB query_items shape)."""

    def __init__(self, items: List[Dict[str, Any]]):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _make_app(admin_dep_override=None):
    """Build a fresh FastAPI app that mounts the dashboard router."""
    from api.dashboard import router, require_admin

    app = FastAPI()
    app.include_router(router)
    if admin_dep_override is not None:
        app.dependency_overrides[require_admin] = admin_dep_override
    return app


# ---------------------------------------------------------------------------
# require_admin gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_require_admin_noop_when_auth_off():
    """When the tenant id is not configured the gate is a no-op."""
    from api.dashboard import require_admin

    cfg = MagicMock()
    cfg.get_value = MagicMock(return_value=None)
    # Should not raise and should return None.
    result = await require_admin(authorization=None, cfg=cfg)
    assert result is None


@pytest.mark.asyncio
async def test_require_admin_rejects_missing_token():
    from api.dashboard import require_admin

    cfg = MagicMock()
    cfg.get_value = MagicMock(return_value="tenant-id")
    with pytest.raises(HTTPException) as exc:
        await require_admin(authorization=None, cfg=cfg)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_require_admin_rejects_non_admin_role():
    from api import dashboard

    cfg = MagicMock()
    cfg.get_value = MagicMock(return_value="tenant-id")
    with patch.object(dashboard, "validate_access_token", new=AsyncMock(return_value={"roles": ["User"]})):
        with pytest.raises(HTTPException) as exc:
            await dashboard.require_admin(authorization="Bearer abc", cfg=cfg)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_admin_accepts_admin_role():
    from api import dashboard

    cfg = MagicMock()
    cfg.get_value = MagicMock(return_value="tenant-id")
    with patch.object(dashboard, "validate_access_token", new=AsyncMock(return_value={"roles": ["Admin"]})):
        result = await dashboard.require_admin(authorization="Bearer abc", cfg=cfg)
    assert result is None


# ---------------------------------------------------------------------------
# Cosmos admin helpers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_overview_buckets_anonymous_principal_ids():
    """Anonymous traffic uses ``anonymous-<conversation_id>`` as both partition
    key and ``principal_id`` (to avoid hot-partitioning, see
    ``orchestration/orchestrator.py``). Counting those naively makes every
    anonymous conversation look like a distinct active user, which inflates
    the Overview metric. Regression for the v2.8.10 sandbox bug where 57
    anonymous conversations reported 57 active users.
    """
    from connectors import cosmosdb_admin

    import time
    now = int(time.time())
    docs = [
        # Three anonymous conversations - all the same logical user.
        {"_ts": now - 3600, "principal_id": "anonymous-aaa", "message_count": 2},
        {"_ts": now - 7200, "principal_id": "anonymous-bbb", "message_count": 2},
        {"_ts": now - 10800, "principal_id": "anonymous", "message_count": 2},
        # One authenticated user (Entra object id shape).
        {
            "_ts": now - 14400,
            "principal_id": "11111111-2222-3333-4444-555555555555",
            "message_count": 4,
        },
    ]

    container = MagicMock()
    container.query_items = MagicMock(return_value=_AsyncIter(docs))
    cosmos = MagicMock()
    cosmos.cfg.get = MagicMock(return_value="conversations")
    cosmos._get_container = MagicMock(return_value=container)

    with patch.object(cosmosdb_admin, "get_cosmosdb_client", return_value=cosmos):
        cosmosdb_admin.cache_clear()
        result = await cosmosdb_admin.fetch_overview(days=7)

    # 1 anonymous bucket + 1 authenticated user, not 4.
    assert result["active_users"] == 2


@pytest.mark.asyncio
async def test_fetch_overview_aggregates_buckets_and_users():
    from connectors import cosmosdb_admin

    import time
    now = int(time.time())
    docs = [
        {"_ts": now - 3600, "principal_id": "user-1", "message_count": 4},
        {"_ts": now - 3 * 86400, "principal_id": "user-2", "message_count": 6},
        {"_ts": now - 10 * 86400, "principal_id": "user-1", "message_count": 2},
        {"_ts": now - 60 * 86400, "principal_id": "user-3", "message_count": 8},
    ]

    container = MagicMock()
    container.query_items = MagicMock(return_value=_AsyncIter(docs))
    cosmos = MagicMock()
    cosmos.cfg.get = MagicMock(return_value="conversations")
    cosmos._get_container = MagicMock(return_value=container)

    with patch.object(cosmosdb_admin, "get_cosmosdb_client", return_value=cosmos):
        cosmosdb_admin.cache_clear()
        result = await cosmosdb_admin.fetch_overview(days=30)

    assert result["total"] == 4
    assert result["today"] == 1
    assert result["last_7_days"] == 2
    assert result["last_30_days"] == 3
    # user-1 appears twice but counts once.
    assert result["active_users"] == 2
    # Dense series spans 30 days.
    assert len(result["conversations_per_day"]) == 30
    # avg_turns is messages//2 over the in-window docs (4//2, 6//2, 2//2)
    assert result["avg_turns"] == pytest.approx((2 + 3 + 1) / 3, abs=0.01)
    # query_items was invoked with partition_key=None (cross-partition).
    _, kwargs = container.query_items.call_args
    assert kwargs.get("partition_key") is None


@pytest.mark.asyncio
async def test_fetch_overview_handles_empty_store():
    from connectors import cosmosdb_admin

    container = MagicMock()
    container.query_items = MagicMock(return_value=_AsyncIter([]))
    cosmos = MagicMock()
    cosmos.cfg.get = MagicMock(return_value="conversations")
    cosmos._get_container = MagicMock(return_value=container)

    with patch.object(cosmosdb_admin, "get_cosmosdb_client", return_value=cosmos):
        cosmosdb_admin.cache_clear()
        result = await cosmosdb_admin.fetch_overview(days=7)

    assert result["total"] == 0
    assert result["today"] == 0
    assert result["last_7_days"] == 0
    assert result["active_users"] == 0
    assert result["avg_turns"] == 0.0
    assert len(result["conversations_per_day"]) == 7
    assert all(p["count"] == 0 for p in result["conversations_per_day"])


@pytest.mark.asyncio
async def test_list_conversations_passes_paging_and_search():
    from connectors import cosmosdb_admin

    docs = [{"id": f"c-{i}", "name": f"chat {i}", "_ts": 1700000000 - i} for i in range(3)]
    container = MagicMock()
    container.query_items = MagicMock(return_value=_AsyncIter(docs))
    cosmos = MagicMock()
    cosmos.cfg.get = MagicMock(return_value="conversations")
    cosmos._get_container = MagicMock(return_value=container)

    with patch.object(cosmosdb_admin, "get_cosmosdb_client", return_value=cosmos):
        rows = await cosmosdb_admin.list_conversations(skip=5, limit=10, search="Foo")

    assert [r["id"] for r in rows] == ["c-0", "c-1", "c-2"]
    _, kwargs = container.query_items.call_args
    assert kwargs.get("partition_key") is None
    params = {p["name"]: p["value"] for p in kwargs["parameters"]}
    assert params["@skip"] == 5
    assert params["@limit"] == 10
    assert params["@search"] == "foo"


# ---------------------------------------------------------------------------
# Endpoints (with require_admin overridden to a no-op)
# ---------------------------------------------------------------------------

def _allow_admin():
    return None


def test_version_endpoint_is_public(tmp_path, monkeypatch):
    from api.dashboard import router  # noqa: F401 - ensure module loaded

    app = _make_app()  # no override; /version is not gated
    client = TestClient(app)
    r = client.get("/api/dashboard/version")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert isinstance(data["version"], str) and data["version"]


def test_overview_endpoint_uses_cache(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    payload = {
        "total": 1,
        "today": 0,
        "last_7_days": 1,
        "last_30_days": 1,
        "active_users": 1,
        "avg_turns": 0.0,
        "conversations_per_day": [{"date": "2025-01-01", "count": 1}],
        "window_days": 1,
    }
    fetch_mock = AsyncMock(return_value=payload)
    monkeypatch.setattr(dashboard, "fetch_overview", fetch_mock)

    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r1 = client.get("/api/dashboard/overview?days=1")
    r2 = client.get("/api/dashboard/overview?days=1")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json()
    # Cache hit on the second call: only one fetch performed.
    assert fetch_mock.await_count == 1


def test_conversations_endpoint_returns_paginated_list(monkeypatch):
    from api import dashboard

    rows = [
        {"id": f"id-{i}", "name": f"chat {i}", "principal_id": "u", "_ts": 1700000000 - i, "message_count": 4}
        for i in range(6)  # one more than the requested limit of 5
    ]
    list_mock = AsyncMock(return_value=rows)
    monkeypatch.setattr(dashboard, "list_conversations", list_mock)

    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations?skip=0&limit=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["conversations"]) == 5
    assert body["has_more"] is True
    assert body["skip"] == 0
    assert body["limit"] == 5
    # fetch is limit+1 to detect the extra row.
    list_mock.assert_awaited_once()
    assert list_mock.await_args.kwargs["limit"] == 6


def test_conversations_endpoint_empty(monkeypatch):
    from api import dashboard

    monkeypatch.setattr(dashboard, "list_conversations", AsyncMock(return_value=[]))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations")
    assert r.status_code == 200
    body = r.json()
    assert body["conversations"] == []
    assert body["has_more"] is False


def test_conversation_detail_404(monkeypatch):
    from api import dashboard

    monkeypatch.setattr(dashboard, "read_conversation", AsyncMock(return_value=None))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations/does-not-exist")
    assert r.status_code == 404


def test_conversation_detail_returns_document(monkeypatch):
    from api import dashboard

    doc = {
        "id": "c-1",
        "name": "demo",
        "principal_id": "user-1",
        "_ts": 1700000000,
        "lastUpdated": "2025-01-01T00:00:00Z",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    monkeypatch.setattr(dashboard, "read_conversation", AsyncMock(return_value=doc))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations/c-1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "c-1"
    assert len(body["messages"]) == 2


def test_overview_rejects_when_auth_on_without_admin(monkeypatch):
    """End-to-end gate check using the real require_admin dependency."""
    from api import dashboard
    from dependencies import get_config

    cfg = MagicMock()
    cfg.get_value = MagicMock(return_value="tenant-id")
    monkeypatch.setattr(
        dashboard,
        "validate_access_token",
        AsyncMock(return_value={"roles": ["User"]}),
    )

    app = _make_app()  # use real gate
    # FastAPI resolves Depends(get_config) by reference, so override the dep
    # rather than monkeypatching the module attribute.
    app.dependency_overrides[get_config] = lambda: cfg
    client = TestClient(app)
    r = client.get(
        "/api/dashboard/overview",
        headers={"Authorization": "Bearer abc"},
    )
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Overview time-range picker (#241)
# ---------------------------------------------------------------------------

def _overview_payload(**overrides):
    base = {
        "total": 1,
        "today": 0,
        "last_7_days": 1,
        "last_30_days": 1,
        "active_users": 1,
        "avg_turns": 0.0,
        "conversations_per_day": [{"date": "2026-06-12", "count": 1}],
        "window_days": 8,
        "from": "2026-06-12",
        "to": "2026-06-19",
        "in_window_count": 1,
    }
    base.update(overrides)
    return base


def test_overview_accepts_custom_from_to_range(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    fetch_mock = AsyncMock(return_value=_overview_payload())
    monkeypatch.setattr(dashboard, "fetch_overview", fetch_mock)

    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/overview?from=2026-06-12&to=2026-06-19")
    assert r.status_code == 200, r.text
    body = r.json()
    # Field name (not alias) is emitted for date columns; "from" stays as the
    # alias because Python forbids ``from`` as a field name.
    assert body["from"] == "2026-06-12"
    assert body["to"] == "2026-06-19"
    assert body["in_window_count"] == 1
    # Backend was called with explicit epoch bounds.
    fetch_mock.assert_awaited_once()
    kwargs = fetch_mock.await_args.kwargs
    assert kwargs["from_ts"] is not None
    assert kwargs["to_ts"] is not None
    assert kwargs["from_ts"] < kwargs["to_ts"]


def test_overview_rejects_partial_range(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    monkeypatch.setattr(dashboard, "fetch_overview", AsyncMock(return_value=_overview_payload()))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)

    r1 = client.get("/api/dashboard/overview?from=2026-06-12")
    r2 = client.get("/api/dashboard/overview?to=2026-06-19")
    assert r1.status_code == 400
    assert r2.status_code == 400
    assert "together" in r1.json()["detail"]


def test_overview_rejects_inverted_range(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    monkeypatch.setattr(dashboard, "fetch_overview", AsyncMock(return_value=_overview_payload()))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/overview?from=2026-06-19&to=2026-06-12")
    assert r.status_code == 400
    assert "<=" in r.json()["detail"]


def test_overview_rejects_range_over_365_days(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    monkeypatch.setattr(dashboard, "fetch_overview", AsyncMock(return_value=_overview_payload()))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/overview?from=2024-01-01&to=2026-01-01")
    assert r.status_code == 400
    assert "365" in r.json()["detail"]


def test_overview_rejects_malformed_date(monkeypatch):
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    monkeypatch.setattr(dashboard, "fetch_overview", AsyncMock(return_value=_overview_payload()))
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/overview?from=not-a-date&to=2026-06-19")
    assert r.status_code == 400
    assert "from" in r.json()["detail"]


def test_overview_cache_key_is_range_specific(monkeypatch):
    """Two different custom ranges must not share a cache entry."""
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    fetch_mock = AsyncMock(return_value=_overview_payload())
    monkeypatch.setattr(dashboard, "fetch_overview", fetch_mock)

    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    client.get("/api/dashboard/overview?from=2026-06-01&to=2026-06-07")
    client.get("/api/dashboard/overview?from=2026-06-08&to=2026-06-14")
    # Distinct ranges -> distinct cache keys -> two backend calls.
    assert fetch_mock.await_count == 2

    # Same range called twice -> cache hit -> still two total backend calls.
    client.get("/api/dashboard/overview?from=2026-06-08&to=2026-06-14")
    assert fetch_mock.await_count == 2


# ---------------------------------------------------------------------------
# Overview ``to`` is end-of-day UTC inclusive (#247 Bug 2)
# ---------------------------------------------------------------------------


def test_overview_to_is_end_of_day_inclusive(monkeypatch):
    """``to=2026-06-19`` must include docs created during 2026-06-19.

    The previous build parsed ``to`` as midnight UTC of that day, which
    dropped everything during the last day of the range -- visible to
    operators as "today's conversations missing from the Overview chart"
    (#247 Bug 2). With the end-of-day fix, a doc with ``_ts`` at
    2026-06-19T13:00:00Z must land inside the window.
    """
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    fetch_mock = AsyncMock(return_value=_overview_payload())
    monkeypatch.setattr(dashboard, "fetch_overview", fetch_mock)
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/overview?from=2026-06-15&to=2026-06-19")
    assert r.status_code == 200, r.text

    kwargs = fetch_mock.await_args.kwargs
    # to_ts must be end-of-day UTC (23:59:59) so the Cosmos ``_ts <= to_ts``
    # filter in fetch_overview includes the whole last day.
    from datetime import datetime, timezone

    expected_to = int(
        datetime(2026, 6, 19, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    )
    expected_from = int(
        datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    assert kwargs["from_ts"] == expected_from
    assert kwargs["to_ts"] == expected_to

    # And a sample doc at 13:00 on 2026-06-19 must be inside the window.
    doc_ts = int(
        datetime(2026, 6, 19, 13, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    assert kwargs["from_ts"] <= doc_ts <= kwargs["to_ts"]


def test_overview_rejects_future_dates(monkeypatch):
    """Custom range bounds past today's UTC end-of-day must 400 (#247 Bug 1)."""
    from api import dashboard
    from connectors import cosmosdb_admin

    cosmosdb_admin.cache_clear()
    monkeypatch.setattr(
        dashboard, "fetch_overview", AsyncMock(return_value=_overview_payload())
    )
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    # 2099-01-01 is comfortably past today's date in any plausible CI clock.
    r = client.get("/api/dashboard/overview?from=2026-06-15&to=2099-01-01")
    assert r.status_code == 400
    assert "future" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Conversation detail reconstruction from ``questions[]`` (#247 Bug 4)
# ---------------------------------------------------------------------------


def test_conversation_detail_reconstructs_from_questions(monkeypatch):
    """Docs persist user prompts under ``questions[]`` not ``messages[]``.

    The orchestrator stores assistant replies on the Azure AI Foundry agent
    thread, so before this fix every dashboard conversation rendered as
    ``user (empty)``/``assistant (empty)`` cards (#247 Bug 4). The detail
    endpoint now projects ``questions[]`` into user-role message entries and
    passes ``thread_id``/``feedback`` through so the UI can deep-link to
    Foundry instead of showing nothing.
    """
    from api import dashboard

    doc = {
        "id": "c-7",
        "name": "What is the fuel tank capacity of the Volkswagen d...",
        "principal_id": "user-42",
        "_ts": 1700000000,
        "lastUpdated": "2026-06-19T13:05:00Z",
        "questions": [
            {"question_id": "q1", "text": "What is the fuel tank capacity?"},
            {"question_id": "q2", "text": "And the trunk volume?"},
        ],
        "thread_id": "thread_abc123",
        "feedback": [{"question_id": "q1", "rating": "thumbs_up"}],
    }
    monkeypatch.setattr(
        dashboard, "read_conversation", AsyncMock(return_value=doc)
    )
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations/c-7")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["id"] == "c-7"
    assert body["thread_id"] == "thread_abc123"
    assert len(body["feedback"]) == 1
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "What is the fuel tank capacity?"
    assert body["messages"][0]["question_id"] == "q1"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "And the trunk volume?"


def test_conversation_detail_empty_questions_returns_empty_messages(monkeypatch):
    """A doc with no ``questions`` and no ``messages`` still returns 200."""
    from api import dashboard

    doc = {
        "id": "c-empty",
        "principal_id": "user-1",
        "_ts": 1700000000,
    }
    monkeypatch.setattr(
        dashboard, "read_conversation", AsyncMock(return_value=doc)
    )
    app = _make_app(admin_dep_override=_allow_admin)
    client = TestClient(app)
    r = client.get("/api/dashboard/conversations/c-empty")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"] == []
    assert body["thread_id"] is None
    assert body["feedback"] == []
