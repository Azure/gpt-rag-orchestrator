"""Tests for the OpenAIChatClient adapter (src/connectors/openai_chat_client.py)."""

import pytest
import asyncio
import logging
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework import ChatMessage, ChatResponse, ChatResponseUpdate
from pydantic import BaseModel, TypeAdapter


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

    @pytest.mark.parametrize("stream", [False, True], ids=["response", "stream"])
    @pytest.mark.parametrize("format_kind", [
        "model", "mapping", "unsupported", "unsupported-class", "adapter",
        "invalid-schema", "invalid-ref", "unexpected", "cancelled",
    ])
    async def test_public_response_format_boundary_uses_real_schema_conversion(self, client, stream, format_kind, caplog):
        marker = "synthetic-private-schema-detail"
        failure = asyncio.CancelledError(marker) if format_kind == "cancelled" else RuntimeError(marker)

        class Answer(BaseModel):
            value: str

        class InvalidSchema(BaseModel):
            callback: Callable[[], None]

        class InvalidReference(BaseModel):
            @classmethod
            def model_json_schema(cls, **kwargs):
                return {"$ref": marker, "description": "invalid reference"}

        class UnsupportedClass:
            pass

        class BrokenSchema(BaseModel):
            @classmethod
            def model_json_schema(cls, **kwargs):
                raise failure

        response_format = {
            "model": Answer,
            "mapping": {"type": "json_object"},
            "unsupported": marker,
            "unsupported-class": UnsupportedClass,
            "adapter": TypeAdapter(str),
            "invalid-schema": InvalidSchema,
            "invalid-ref": InvalidReference,
            "unexpected": BrokenSchema,
            "cancelled": BrokenSchema,
        }[format_kind]
        completion = MagicMock()
        completion.choices[0].message.content = "answer"
        completion.model = "synthetic-model"
        completion.id = "synthetic-response"

        async def chunks():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "answer"
            chunk.id = "synthetic-response"
            chunk.model = "synthetic-model"
            yield chunk

        self._mock_oai.chat.completions.create = AsyncMock(return_value=chunks() if stream else completion)

        async def run():
            options = {"response_format": response_format}
            if stream:
                return "".join([part.text async for part in client.get_streaming_response("question", options=options)])
            return (await client.get_response("question", options=options)).text

        if format_kind in {"unexpected", "cancelled"}:
            with pytest.raises(type(failure)) as raised:
                await run()
            assert raised.value is failure
            self._mock_oai.chat.completions.create.assert_not_awaited()
        else:
            assert await run() == "answer"
            params = self._mock_oai.chat.completions.create.await_args.kwargs
            if format_kind == "model":
                assert params["response_format"]["json_schema"]["name"] == "Answer"
                assert params["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
            elif format_kind == "mapping":
                assert params["response_format"] == {"type": "json_object"}
            else:
                assert "response_format" not in params
                assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert marker not in caplog.text
