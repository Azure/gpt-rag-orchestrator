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
