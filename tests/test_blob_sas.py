"""Tests for the blob citation signing helper.

Signing must be transparent: anything that is not one of our own unsigned blob
URLs has to come back untouched, and every failure path has to degrade to the
original link rather than dropping the citation.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util import blob_sas  # noqa: E402


ACCOUNT = "stexample"
BLOB_URL = f"https://{ACCOUNT}.blob.core.windows.net/documents/report.txt"


@pytest.fixture(autouse=True)
def _clear_cache():
    blob_sas.reset_delegation_key_cache()
    yield
    blob_sas.reset_delegation_key_cache()


@pytest.fixture
def signing_env(monkeypatch):
    """Configure an account and stub out the two Azure calls."""
    calls = {"keys": 0, "signatures": []}

    async def fake_key(account):
        calls["keys"] += 1
        return f"delegation-key-for-{account}"

    def fake_generate(**kwargs):
        calls["signatures"].append(kwargs)
        return "sv=2024-01-01&sig=fake-signature"

    monkeypatch.setattr(blob_sas, "_storage_account_name", lambda: ACCOUNT)
    monkeypatch.setattr(blob_sas, "_get_user_delegation_key", fake_key)
    monkeypatch.setattr("azure.storage.blob.generate_blob_sas", fake_generate)
    return calls


def sign(url):
    return asyncio.run(blob_sas.sign_blob_url(url))


def test_signs_our_own_unsigned_blob_url(signing_env):
    result = sign(BLOB_URL)

    assert result.startswith(BLOB_URL + "?")
    assert "sig=fake-signature" in result
    signed = signing_env["signatures"][0]
    assert signed["account_name"] == ACCOUNT
    assert signed["container_name"] == "documents"
    assert signed["blob_name"] == "report.txt"


def test_grants_read_only_access_with_an_expiry(signing_env):
    sign(BLOB_URL)

    signed = signing_env["signatures"][0]
    assert signed["permission"].read is True
    assert signed["permission"].write is False
    assert signed["permission"].delete is False
    horizon = datetime.now(timezone.utc) + timedelta(
        hours=blob_sas.DEFAULT_EXPIRY_HOURS
    )
    assert signed["expiry"] <= horizon + timedelta(minutes=1)
    assert signed["expiry"] > datetime.now(timezone.utc)


def test_decodes_percent_encoded_blob_names(signing_env):
    sign(f"https://{ACCOUNT}.blob.core.windows.net/documents/my%20report.txt")

    assert signing_env["signatures"][0]["blob_name"] == "my report.txt"


def test_preserves_a_fragment(signing_env):
    result = sign(BLOB_URL + "#page=4")

    assert result.endswith("#page=4")
    assert "sig=fake-signature" in result


def test_leaves_an_already_signed_url_untouched(signing_env):
    already = BLOB_URL + "?sv=2024-01-01&sig=existing"

    assert sign(already) == already
    assert signing_env["signatures"] == []


def test_leaves_an_external_url_untouched(signing_env):
    external = "https://contoso.blob.core.windows.net/documents/report.txt"

    assert sign(external) == external
    assert signing_env["signatures"] == []


def test_leaves_a_relative_reference_untouched(signing_env):
    assert sign("documents/report.txt") == "documents/report.txt"
    assert signing_env["signatures"] == []


def test_leaves_a_container_only_url_untouched(signing_env):
    container_only = f"https://{ACCOUNT}.blob.core.windows.net/documents"

    assert sign(container_only) == container_only
    assert signing_env["signatures"] == []


def test_returns_empty_input_unchanged(signing_env):
    assert sign("") == ""


def test_leaves_the_url_untouched_when_no_account_is_configured(monkeypatch):
    monkeypatch.setattr(blob_sas, "_storage_account_name", lambda: "")

    assert sign(BLOB_URL) == BLOB_URL


def test_falls_back_to_the_original_url_when_signing_fails(monkeypatch):
    async def failing_key(account):
        raise RuntimeError("no delegation permission")

    monkeypatch.setattr(blob_sas, "_storage_account_name", lambda: ACCOUNT)
    monkeypatch.setattr(blob_sas, "_get_user_delegation_key", failing_key)

    assert sign(BLOB_URL) == BLOB_URL


def test_reuses_a_cached_delegation_key(monkeypatch):
    calls = {"count": 0}

    class FakeService:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_user_delegation_key(self, **kwargs):
            calls["count"] += 1
            return "delegation-key"

    monkeypatch.setattr(blob_sas, "_storage_account_name", lambda: ACCOUNT)
    monkeypatch.setattr("azure.storage.blob.aio.BlobServiceClient", FakeService)
    monkeypatch.setattr(
        "azure.storage.blob.generate_blob_sas",
        lambda **kwargs: "sv=2024-01-01&sig=fake",
    )

    async def sign_twice():
        await blob_sas.sign_blob_url(BLOB_URL)
        await blob_sas.sign_blob_url(BLOB_URL)

    asyncio.run(sign_twice())

    assert calls["count"] == 1
