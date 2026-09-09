"""Optional profile persistence keeps its established best-effort boundary."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from strategies.maf_agent_service_strategy import MafAgentServiceStrategy
from strategies.maf_lite_strategy import MafLiteStrategy
from strategies.maf_plugins.user_profile_memory import UserProfile
from strategies.multimodal_strategy import MultimodalStrategy


@pytest.fixture(params=[MafLiteStrategy, MafAgentServiceStrategy, MultimodalStrategy])
def profile_strategy(request):
    strategy = request.param.__new__(request.param)
    strategy.user_profile_container = "profiles"
    strategy.cosmos = SimpleNamespace(
        get_document=AsyncMock(return_value=None),
        create_document=AsyncMock(return_value={"id": "user_profile_user-1"}),
        update_document=AsyncMock(return_value={"id": "user_profile_user-1"}),
    )
    return strategy


@pytest.mark.parametrize("outcome", ["missing", "valid", "malformed", "unavailable", "cancelled"])
async def test_optional_profile_load_preserves_empty_valid_and_cancelled_outcomes(
    profile_strategy, outcome, caplog,
):
    caplog.set_level("DEBUG")
    marker = "synthetic-private-profile-detail"
    expected = UserProfile(name="Test User", preferences=["concise"])
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    if outcome == "valid":
        profile_strategy.cosmos.get_document.return_value = {"profile_data": expected.model_dump_json()}
    elif outcome == "malformed":
        profile_strategy.cosmos.get_document.return_value = {
            "profile_data": json.dumps({"name": {"detail": marker}}),
        }
    elif outcome in {"unavailable", "cancelled"}:
        profile_strategy.cosmos.get_document.side_effect = failure
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await profile_strategy._load_user_profile("user-1")
        assert raised.value is failure
    else:
        loaded = await profile_strategy._load_user_profile("user-1")
        assert loaded == (expected if outcome == "valid" else UserProfile())
    profile_strategy.cosmos.get_document.assert_awaited_once_with(
        "profiles", "user_profile_user-1")
    profile_strategy.cosmos.create_document.assert_not_awaited()
    profile_strategy.cosmos.update_document.assert_not_awaited()
    assert marker not in caplog.text


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("outcome", ["success", "read_failure", "write_failure", "unconfirmed", "cancelled"])
async def test_optional_profile_save_preserves_write_scope_and_bounded_failure(
    profile_strategy, existing, outcome, caplog,
):
    caplog.set_level("INFO")
    marker = "synthetic-private-profile-detail"
    profile = UserProfile(name="Test User", preferences=["concise"])
    profile_strategy.cosmos.get_document.return_value = {"id": "user_profile_user-1"} if existing else None
    write = (
        profile_strategy.cosmos.update_document if existing
        else profile_strategy.cosmos.create_document
    )
    failure = asyncio.CancelledError(marker) if outcome == "cancelled" else RuntimeError(marker)
    if outcome == "read_failure":
        profile_strategy.cosmos.get_document.side_effect = failure
    elif outcome in {"write_failure", "cancelled"}:
        write.side_effect = failure
    elif outcome == "unconfirmed":
        write.return_value = None
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as raised:
            await profile_strategy._save_user_profile("user-1", profile)
        assert raised.value is failure
    else:
        assert await profile_strategy._save_user_profile("user-1", profile) is None
    profile_strategy.cosmos.get_document.assert_awaited_once_with(
        "profiles", "user_profile_user-1")
    if outcome == "read_failure":
        write.assert_not_awaited()
    else:
        write.assert_awaited_once()
        if existing:
            container, doc = write.await_args.args
        else:
            container, key = write.await_args.args
            assert key == "user_profile_user-1"
            doc = write.await_args.kwargs["body"]
        assert container == "profiles"
        assert doc["id"] == "user_profile_user-1"
        assert UserProfile.model_validate_json(doc["profile_data"]) == profile
        assert isinstance(doc["updated_at"], float)
    assert marker not in caplog.text
    assert ("Saved user profile" in caplog.text) is (outcome == "success")
    if outcome == "unconfirmed":
        assert "not confirmed" in caplog.text
