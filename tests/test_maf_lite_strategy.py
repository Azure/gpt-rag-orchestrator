"""Tests for MafLiteStrategy (src/strategies/maf_lite_strategy.py)."""

import pytest
from unittest.mock import patch, MagicMock

from strategies.agent_strategies import AgentStrategies


class TestMafLiteStrategy:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch("strategies.maf_lite_strategy.get_config", return_value=mock_config):
            yield

    def test_strategy_type(self):
        from strategies.maf_lite_strategy import MafLiteStrategy
        s = MafLiteStrategy()
        assert s.strategy_type == AgentStrategies.MAF_LITE

    def test_prompt_namespace_returns_maf(self):
        from strategies.maf_lite_strategy import MafLiteStrategy
        s = MafLiteStrategy()
        assert s._prompt_namespace() == "maf"

    def test_user_profile_container(self):
        from strategies.maf_lite_strategy import MafLiteStrategy
        s = MafLiteStrategy()
        assert s.user_profile_container == "conversations"

    def test_chat_client_is_openai_type(self):
        from strategies.maf_lite_strategy import MafLiteStrategy
        from connectors.openai_chat_client import OpenAIChatClient

        with patch("strategies.maf_lite_strategy.OpenAIChatClient") as MockClient:
            MockClient.return_value = MagicMock(spec=OpenAIChatClient)
            s = MafLiteStrategy()
            client = s._get_or_create_chat_client()
            MockClient.assert_called_once()
            assert client is MockClient.return_value
