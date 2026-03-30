"""Tests for the OpenAIChatClient adapter (src/connectors/openai_chat_client.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework import ChatMessage, ChatResponse, ChatResponseUpdate


class TestOpenAIChatClient:
    """Verify message conversion, get_response, and streaming."""

    @pytest.fixture()
    def client(self):
        with patch("connectors.openai_chat_client.get_bearer_token_provider") as mock_tp:
            mock_tp.return_value = lambda: "fake-token"
            with patch("connectors.openai_chat_client.AsyncAzureOpenAI") as MockOAI:
                self._mock_oai = MockOAI.return_value
                from connectors.openai_chat_client import OpenAIChatClient

                c = OpenAIChatClient(
                    azure_endpoint="https://fake.openai.azure.com",
                    model_deployment_name="gpt-4o",
                    credential=MagicMock(),
                )
                yield c

    # --- _to_openai_messages ---

    def test_string_message(self, client):
        result = client._to_openai_messages("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_chat_message(self, client):
        msg = ChatMessage(role="system", text="You are helpful.")
        result = client._to_openai_messages(msg)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_sequence_of_messages(self, client):
        msgs = [
            ChatMessage(role="system", text="sys"),
            ChatMessage(role="user", text="hi"),
            "plain string",
        ]
        result = client._to_openai_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[2] == {"role": "user", "content": "plain string"}

    # --- get_response ---

    @pytest.mark.asyncio
    async def test_get_response(self, client):
        choice = MagicMock()
        choice.message.content = "Hello from GPT"
        completion = MagicMock()
        completion.choices = [choice]
        completion.model = "gpt-4o"
        completion.id = "resp-1"
        self._mock_oai.chat.completions.create = AsyncMock(return_value=completion)

        resp = await client.get_response("greet me")
        assert isinstance(resp, ChatResponse)
        assert resp.text == "Hello from GPT"
        self._mock_oai.chat.completions.create.assert_awaited_once()

    # --- get_streaming_response ---

    @pytest.mark.asyncio
    async def test_streaming_response(self, client):
        # Simulate an async iterable of OpenAI stream chunks
        chunks = []
        for text in ["He", "llo", " world"]:
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = text
            c.id = "resp-1"
            c.model = "gpt-4o"
            chunks.append(c)

        async def fake_stream():
            for c_ in chunks:
                yield c_

        self._mock_oai.chat.completions.create = AsyncMock(return_value=fake_stream())

        collected = []
        async for update in client.get_streaming_response("stream me"):
            assert isinstance(update, ChatResponseUpdate)
            collected.append(update.text)

        assert "".join(collected) == "Hello world"

    # --- additional_properties ---

    def test_has_additional_properties(self, client):
        assert hasattr(client, "additional_properties")
        assert isinstance(client.additional_properties, dict)
