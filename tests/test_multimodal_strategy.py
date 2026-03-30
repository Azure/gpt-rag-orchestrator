"""Tests for the Multimodal Strategy and its components."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from strategies.agent_strategies import AgentStrategies
from connectors.multimodal_chat_client import MultimodalChatClient, MULTIMODAL_PREFIX


class _AsyncSearchResults:
    def __init__(self, docs):
        self._docs = iter(docs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._docs)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeSearchClient:
    def __init__(self, docs):
        self._docs = docs
        self.search_kwargs = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def search(self, **kwargs):
        self.search_kwargs = kwargs
        return _AsyncSearchResults(self._docs)


# ======================================================================
# MultimodalChatClient tests
# ======================================================================

class TestMultimodalChatClient:
    """Tests for MultimodalChatClient message conversion."""

    def test_plain_text_message_unchanged(self):
        """Regular text messages should pass through unchanged."""
        msg = MagicMock()
        msg.text = "Hello, how are you?"
        msg.role = MagicMock(value="user")

        result = MultimodalChatClient._convert_message(msg)
        assert result == {"role": "user", "content": "Hello, how are you?"}

    def test_multimodal_message_with_text_and_image(self):
        """Messages with MULTIMODAL_PREFIX should be parsed into content arrays."""
        parts = [
            {"type": "text", "text": "Here is a document"},
            {"type": "image_url", "url": "data:image/png;base64,abc123"},
        ]
        msg = MagicMock()
        msg.text = MULTIMODAL_PREFIX + json.dumps(parts)
        msg.role = MagicMock(value="system")

        result = MultimodalChatClient._convert_message(msg)
        assert result["role"] == "system"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Here is a document"
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == "data:image/png;base64,abc123"
        assert result["content"][1]["image_url"]["detail"] == "auto"

    def test_multimodal_message_with_custom_detail(self):
        """Image parts can specify a custom detail level."""
        parts = [
            {"type": "text", "text": "test"},
            {"type": "image_url", "url": "data:image/png;base64,x", "detail": "low"},
        ]
        msg = MagicMock()
        msg.text = MULTIMODAL_PREFIX + json.dumps(parts)
        msg.role = MagicMock(value="system")

        result = MultimodalChatClient._convert_message(msg)
        assert result["content"][1]["image_url"]["detail"] == "low"

    def test_malformed_json_falls_back_to_text(self):
        """If JSON parsing fails, fall back to plain text (minus prefix)."""
        msg = MagicMock()
        msg.text = MULTIMODAL_PREFIX + "not valid json{{"
        msg.role = MagicMock(value="system")

        result = MultimodalChatClient._convert_message(msg)
        assert result["role"] == "system"
        assert isinstance(result["content"], str)
        assert "not valid json" in result["content"]

    def test_to_openai_messages_with_mixed_content(self):
        """_to_openai_messages should handle a mix of plain and multimodal."""
        plain_msg = MagicMock()
        plain_msg.text = "plain text"
        plain_msg.role = MagicMock(value="user")

        mm_parts = [{"type": "text", "text": "context"}]
        mm_msg = MagicMock()
        mm_msg.text = MULTIMODAL_PREFIX + json.dumps(mm_parts)
        mm_msg.role = MagicMock(value="system")

        results = MultimodalChatClient._to_openai_messages([plain_msg, mm_msg])
        assert len(results) == 2
        assert results[0]["content"] == "plain text"
        assert isinstance(results[1]["content"], list)

    def test_to_openai_messages_string_input(self):
        """String input should be wrapped in a user message."""
        results = MultimodalChatClient._to_openai_messages("hello")
        assert results == [{"role": "user", "content": "hello"}]

    def test_multiple_images(self):
        """Multiple images should all be included in the content array."""
        parts = [
            {"type": "text", "text": "doc text"},
            {"type": "image_url", "url": "data:image/png;base64,img1"},
            {"type": "image_url", "url": "data:image/png;base64,img2"},
            {"type": "image_url", "url": "data:image/png;base64,img3"},
        ]
        msg = MagicMock()
        msg.text = MULTIMODAL_PREFIX + json.dumps(parts)
        msg.role = MagicMock(value="system")

        result = MultimodalChatClient._convert_message(msg)
        image_parts = [p for p in result["content"] if p["type"] == "image_url"]
        assert len(image_parts) == 3


class TestMarkdownImageDedup:
    def test_dedup_markdown_images_keeps_first_path_only(self):
        from strategies.multimodal_strategy import _dedup_markdown_images

        text = (
            "Step 1 ![Figure 100.3](documents-images/figure-100.3.png)\n"
            "Step 2 ![Figure 101.2](documents-images/figure-101.2.png)\n"
            "Repeat ![Figure 100.3](documents-images/figure-100.3.png)"
        )

        result = _dedup_markdown_images(text)
        assert result.count("documents-images/figure-100.3.png") == 1
        assert result.count("documents-images/figure-101.2.png") == 1


# ======================================================================
# MultimodalStrategy tests
# ======================================================================

class TestMultimodalStrategy:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch("strategies.multimodal_strategy.get_config", return_value=mock_config):
            yield

    def test_strategy_type(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.strategy_type == AgentStrategies.MULTIMODAL

    def test_prompt_namespace_returns_multimodal(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s._prompt_namespace() == "multimodal"

    def test_user_profile_container(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.user_profile_container == "conversations"

    def test_chat_client_is_multimodal_type(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            MockClient.return_value = MagicMock(spec=MultimodalChatClient)
            s = MultimodalStrategy()
            client = s._get_or_create_chat_client()
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_max_images_default(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.max_images == 10

    def test_max_images_per_doc_default(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.max_images_per_doc == 5

    def test_max_content_chars_default(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.max_content_chars == 4000

    def test_classify_images_default_enabled(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.classify_images is True

    @pytest.mark.asyncio
    async def test_create_search_provider_passes_visual_classifier_callback_when_enabled(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        with (
            patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient,
            patch("strategies.multimodal_strategy.MultimodalSearchContextProvider") as MockProvider,
        ):
            MockClient.return_value = MagicMock(spec=MultimodalChatClient)
            MockProvider.return_value = MagicMock()

            s = MultimodalStrategy()
            s.search_endpoint = "https://search.example.com"
            s.search_index_name = "ragindex"
            s.classify_images = True

            provider = await s._create_search_provider()

            assert provider is MockProvider.return_value
            assert MockProvider.call_args.kwargs["classify_images_fn"] is not None
            assert (
                MockProvider.call_args.kwargs["classify_images_concurrency"]
                == s.image_classification_concurrency
            )

    @pytest.mark.asyncio
    async def test_create_search_provider_omits_visual_classifier_callback_when_disabled(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        with (
            patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient,
            patch("strategies.multimodal_strategy.MultimodalSearchContextProvider") as MockProvider,
        ):
            MockClient.return_value = MagicMock(spec=MultimodalChatClient)
            MockProvider.return_value = MagicMock()

            s = MultimodalStrategy()
            s.search_endpoint = "https://search.example.com"
            s.search_index_name = "ragindex"
            s.classify_images = False

            await s._create_search_provider()

            assert MockProvider.call_args.kwargs["classify_images_fn"] is None

    @pytest.mark.asyncio
    async def test_classify_image_relevance_returns_true_for_keep(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "KEEP"

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            result = await s._classify_image_relevance({
                "query": "How do I rebuild a VW engine step by step?",
                "fig_path": "documents-images/figure-101.2.png",
                "caption": "Valve clearance diagram",
                "local_text": "Adjust the valves before closing the covers",
                "image_base64": "abc123",
            })

            assert result is True

    @pytest.mark.asyncio
    async def test_classify_image_relevance_returns_false_for_skip(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SKIP"

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            result = await s._classify_image_relevance({
                "query": "How do I rebuild a VW engine step by step?",
                "fig_path": "documents-images/figure-69.1.png",
                "caption": "Decorative boot illustration",
                "local_text": "Engine rebuild chapter opener",
                "image_base64": "abc123",
            })

            assert result is False

    @pytest.mark.asyncio
    async def test_classify_image_relevance_returns_false_for_empty(self):
        """Fail-closed: empty classifier result means the image is skipped."""
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].finish_reason = "length"
        mock_response.usage = MagicMock(completion_tokens=0)

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            result = await s._classify_image_relevance({
                "query": "How do I rebuild a VW engine?",
                "fig_path": "documents-images/figure-42.png",
                "caption": "Unknown illustration",
                "local_text": "Some text near the figure",
                "image_base64": "abc123",
            })

            assert result is False


# ======================================================================
# Factory registration test
# ======================================================================

class TestMultimodalFactoryRegistration:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch("strategies.multimodal_strategy.get_config", return_value=mock_config):
            yield

    @pytest.mark.asyncio
    async def test_factory_returns_multimodal_strategy(self):
        from strategies.agent_strategy_factory import AgentStrategyFactory
        from strategies.multimodal_strategy import MultimodalStrategy
        strategy = await AgentStrategyFactory.get_strategy("multimodal")
        assert isinstance(strategy, MultimodalStrategy)
        assert strategy.strategy_type == AgentStrategies.MULTIMODAL


# ======================================================================
# MultimodalSearchContextProvider tests
# ======================================================================

class TestMultimodalSearchContextProvider:
    """Tests for the multimodal search context provider."""

    def test_init_parameters(self):
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
            top_k=5,
            max_images=20,
            max_images_per_doc=4,
        )
        assert provider._top_k == 5
        assert provider._max_images == 20
        assert provider._max_images_per_doc == 4
        assert provider._index_name == "ragindex"

    def test_default_limits(self):
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
        )
        assert provider._max_images == 10
        assert provider._max_images_per_doc == 3

    @pytest.mark.asyncio
    async def test_invoking_with_no_user_messages_returns_empty(self):
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider
        from agent_framework import ChatMessage, Role

        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
        )
        # Pass only a system message — no user text
        msg = ChatMessage(role=Role.SYSTEM, text="system prompt")
        ctx = await provider.invoking(msg)
        assert ctx.messages is None or len(ctx.messages) == 0

    @pytest.mark.asyncio
    async def test_invoking_keeps_image_when_visual_classifier_accepts(self):
        from agent_framework import ChatMessage, Role
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider

        doc = {
            "title": "VW Manual",
            "filepath": "how-to-keep-vw-alive.pdf",
            "content": "Inspect the valves <figure>documents-images/figure-101.2.png</figure> then continue.",
            "relatedImages": [
                "https://acct.blob.core.windows.net/documents-images/figure-101.2.png"
            ],
            "imageCaptions": "[figure-101.2.png]: Valve clearance diagram",
        }
        classify_images_fn = AsyncMock(return_value=True)
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
            classify_images_fn=classify_images_fn,
        )
        provider._download_image_as_base64 = AsyncMock(return_value="abc123")

        with patch(
            "strategies.multimodal_search_context_provider.SearchClient",
            return_value=_FakeSearchClient([doc]),
        ):
            ctx = await provider.invoking(ChatMessage(role=Role.USER, text="How do I fix valves in a VW?"))

        assert ctx.messages
        payload = ctx.messages[0].text
        assert payload.startswith(MULTIMODAL_PREFIX)
        parts = json.loads(payload[len(MULTIMODAL_PREFIX):])
        assert any(part["type"] == "image_url" for part in parts)
        assert any(
            part["type"] == "text" and "documents-images/figure-101.2.png" in part["text"]
            for part in parts
        )
        classify_images_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoking_skips_image_when_visual_classifier_rejects(self):
        from agent_framework import ChatMessage, Role
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider

        doc = {
            "title": "VW Manual",
            "filepath": "how-to-keep-vw-alive.pdf",
            "content": "Inspect the valves <figure>documents-images/figure-101.2.png</figure> then continue.",
            "relatedImages": [
                "https://acct.blob.core.windows.net/documents-images/figure-101.2.png"
            ],
            "imageCaptions": "[figure-101.2.png]: Valve clearance diagram",
        }
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
            classify_images_fn=AsyncMock(return_value=False),
        )
        provider._download_image_as_base64 = AsyncMock(return_value="abc123")

        with patch(
            "strategies.multimodal_search_context_provider.SearchClient",
            return_value=_FakeSearchClient([doc]),
        ):
            ctx = await provider.invoking(ChatMessage(role=Role.USER, text="How do I fix valves in a VW?"))

        assert ctx.messages
        payload = ctx.messages[0].text
        assert not payload.startswith(MULTIMODAL_PREFIX)
        assert "documents-images/figure-101.2.png" not in payload

    @pytest.mark.asyncio
    async def test_invoking_does_not_call_visual_classifier_for_heuristically_irrelevant_figure(self):
        from agent_framework import ChatMessage, Role
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider

        classify_images_fn = AsyncMock(return_value=True)
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
            classify_images_fn=classify_images_fn,
        )
        provider._download_image_as_base64 = AsyncMock(return_value="abc123")
        doc = {
            "title": "VW Manual",
            "filepath": "how-to-keep-vw-alive.pdf",
            "content": "Chapter opener <figure>documents-images/figure-69.1.png</figure> engine notes.",
            "relatedImages": [
                "https://acct.blob.core.windows.net/documents-images/figure-69.1.png"
            ],
            "imageCaptions": "[figure-69.1.png]: Decorative boot illustration used as chapter divider",
        }

        with patch(
            "strategies.multimodal_search_context_provider.SearchClient",
            return_value=_FakeSearchClient([doc]),
        ):
            await provider.invoking(ChatMessage(role=Role.USER, text="How do I rebuild a VW engine step by step?"))

        classify_images_fn.assert_not_awaited()
        provider._download_image_as_base64.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invoking_passes_expected_payload_to_visual_classifier(self):
        from agent_framework import ChatMessage, Role
        from strategies.multimodal_search_context_provider import MultimodalSearchContextProvider

        classify_images_fn = AsyncMock(return_value=True)
        provider = MultimodalSearchContextProvider(
            endpoint="https://search.example.com",
            index_name="ragindex",
            credential=MagicMock(),
            blob_credential=MagicMock(),
            classify_images_fn=classify_images_fn,
        )
        provider._download_image_as_base64 = AsyncMock(return_value="abc123")
        doc = {
            "title": "VW Manual",
            "filepath": "how-to-keep-vw-alive.pdf",
            "content": "Inspect the valves carefully <figure>documents-images/figure-101.2.png</figure> then continue.",
            "relatedImages": [
                "https://acct.blob.core.windows.net/documents-images/figure-101.2.png"
            ],
            "imageCaptions": "[figure-101.2.png]: Valve clearance diagram",
        }

        with patch(
            "strategies.multimodal_search_context_provider.SearchClient",
            return_value=_FakeSearchClient([doc]),
        ):
            await provider.invoking(ChatMessage(role=Role.USER, text="How do I fix valves in a VW?"))

        candidate = classify_images_fn.await_args.args[0]
        assert candidate["query"] == "How do I fix valves in a VW?"
        assert candidate["fig_path"] == "documents-images/figure-101.2.png"
        assert candidate["caption"] == "Valve clearance diagram"
        assert candidate["image_base64"] == "abc123"
        assert "Inspect the valves carefully" in candidate["local_text"]


class TestMultimodalImageFilteringHelpers:
    def test_parse_image_captions_maps_by_basename(self):
        from strategies.multimodal_search_context_provider import _parse_image_captions

        figure_paths = [
            "documents-images/how-to-keep-vw-alive.pdf-figure-100.3.png",
            "documents-images/how-to-keep-vw-alive.pdf-figure-101.2.png",
        ]
        captions = (
            "[how-to-keep-vw-alive.pdf-figure-100.3.png]: Timing marks diagram "
            "[how-to-keep-vw-alive.pdf-figure-101.2.png]: Valve clearance illustration"
        )

        result = _parse_image_captions(captions, figure_paths)
        assert result[figure_paths[0]] == "Timing marks diagram"
        assert result[figure_paths[1]] == "Valve clearance illustration"

    def test_parse_image_captions_falls_back_to_order(self):
        from strategies.multimodal_search_context_provider import _parse_image_captions

        figure_paths = ["fig-1.png", "fig-2.png"]
        captions = "[caption-1]: Exploded carburetor view [caption-2]: Distributor timing diagram"

        result = _parse_image_captions(captions, figure_paths)
        assert result["fig-1.png"] == "Exploded carburetor view"
        assert result["fig-2.png"] == "Distributor timing diagram"

    def test_is_relevant_figure_rejects_decorative_boot_image(self):
        from strategies.multimodal_search_context_provider import _is_relevant_figure

        result = _is_relevant_figure(
            fig_path="documents-images/how-to-keep-vw-alive.pdf-figure-69.1.png",
            caption="Decorative boot illustration used as chapter divider",
            local_text="Quick diagram engine major parts useful as you disassemble",
            query="How do I rebuild a VW engine step by step?",
        )

        assert result is False

    def test_is_relevant_figure_accepts_mechanical_figure(self):
        from strategies.multimodal_search_context_provider import _is_relevant_figure

        result = _is_relevant_figure(
            fig_path="documents-images/how-to-keep-vw-alive.pdf-figure-101.2.png",
            caption="Valve clearance diagram with rocker arm and timing references",
            local_text="Check and adjust valve clearance before final assembly",
            query="How do I fix valves in a VW?",
        )

        assert result is True


# ======================================================================
# _extract_blob_relative_path tests
# ======================================================================

class TestExtractBlobRelativePath:
    """Tests for the blob URL to relative path extraction utility."""

    def test_standard_blob_url(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://staccount.blob.core.windows.net/documents-images/figures/arch-overview.png"
        assert _extract_blob_relative_path(url) == "figures/arch-overview.png"

    def test_nested_path(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://staccount.blob.core.windows.net/documents-images/chapter-01/section-02/fig1.jpg"
        assert _extract_blob_relative_path(url) == "chapter-01/section-02/fig1.jpg"

    def test_file_at_container_root(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://staccount.blob.core.windows.net/documents-images/diagram.png"
        assert _extract_blob_relative_path(url) == "diagram.png"

    def test_no_path_after_container_returns_none(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://staccount.blob.core.windows.net/documents-images"
        assert _extract_blob_relative_path(url) is None

    def test_empty_string_returns_none(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        assert _extract_blob_relative_path("") is None

    def test_url_with_spaces_encoded(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://staccount.blob.core.windows.net/images/my%20diagram.png"
        assert _extract_blob_relative_path(url) == "my%20diagram.png"

    def test_different_container_name(self):
        from strategies.multimodal_search_context_provider import _extract_blob_relative_path
        url = "https://mystg.blob.core.windows.net/my-container/path/to/image.png"
        assert _extract_blob_relative_path(url) == "path/to/image.png"


# ======================================================================
# Intent classification tests
# ======================================================================

class TestIntentClassification:
    """Tests for the LLM-based intent classification."""

    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch("strategies.multimodal_strategy.get_config", return_value=mock_config):
            yield

    @pytest.mark.asyncio
    async def test_classify_greeting(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "GREETING"

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            intent = await s._classify_intent("Hello, how are you?")
            assert intent == "greeting"

    @pytest.mark.asyncio
    async def test_classify_question(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "QUESTION"

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            intent = await s._classify_intent("How do I adjust the valves?")
            assert intent == "question"

    @pytest.mark.asyncio
    async def test_classify_intent_error_defaults_to_question(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(
                side_effect=Exception("API error")
            )
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            intent = await s._classify_intent("anything")
            assert intent == "question"


# ======================================================================
# Post-response image validation guardrail tests
# ======================================================================

class TestValidateResponseImages:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_config):
        with patch("strategies.multimodal_strategy.get_config", return_value=mock_config):
            yield

    @pytest.mark.asyncio
    async def test_no_images_returns_text_unchanged(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        with patch("strategies.multimodal_strategy.MultimodalChatClient"):
            s = MultimodalStrategy()
            text = "This is a plain answer with no images."
            result = await s._validate_response_images(text, "any query")
            assert result == text

    @pytest.mark.asyncio
    async def test_valid_image_is_kept(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "VALID"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(completion_tokens=5)

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {"figures/fig-101.png": "abc123base64"}

            text = "Here is a diagram:\n\n![Figure 101](figures/fig-101.png)\n\nAs shown above."
            result = await s._validate_response_images(text, "how to rebuild engine")
            assert "![Figure 101](figures/fig-101.png)" in result

    @pytest.mark.asyncio
    async def test_invalid_image_is_stripped(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "INVALID"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(completion_tokens=5)

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {"figures/fig-69.png": "decorativeart64"}

            text = "Step one:\n\n![Figure 69](figures/fig-69.png)\n\nContinue with step two."
            result = await s._validate_response_images(text, "how to rebuild engine")
            assert "![Figure 69]" not in result
            assert "Continue with step two." in result

    @pytest.mark.asyncio
    async def test_empty_classifier_result_strips_image(self):
        """Fail-closed: empty model response means the image is stripped."""
        from strategies.multimodal_strategy import MultimodalStrategy

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].finish_reason = "length"
        mock_response.usage = MagicMock(completion_tokens=0)

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {"figures/fig-42.png": "somebase64"}

            text = "![Figure 42](figures/fig-42.png)"
            result = await s._validate_response_images(text, "query")
            assert "![Figure 42]" not in result

    @pytest.mark.asyncio
    async def test_classifier_error_strips_image(self):
        """Fail-closed: exception during validation means the image is stripped."""
        from strategies.multimodal_strategy import MultimodalStrategy

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(
                side_effect=Exception("API timeout")
            )
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {"figures/fig-42.png": "somebase64"}

            text = "Answer: ![Figure 42](figures/fig-42.png) done"
            result = await s._validate_response_images(text, "query")
            assert "![Figure 42]" not in result
            assert "Answer:" in result
            assert "done" in result

    @pytest.mark.asyncio
    async def test_no_base64_data_strips_image(self):
        """If image_data doesn't have the path, strip it (fail-closed)."""
        from strategies.multimodal_strategy import MultimodalStrategy

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            MockClient.return_value = MagicMock()

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {}  # No base64 stored

            text = "![Figure 99](figures/fig-99.png)"
            result = await s._validate_response_images(text, "query")
            assert "![Figure 99]" not in result

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid_keeps_only_valid(self):
        from strategies.multimodal_strategy import MultimodalStrategy

        def _mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            user_msg = messages[-1] if messages else {}
            content_parts = user_msg.get("content", [])
            # Find the image URL to determine which image is being validated
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if "good" in url:
                        resp = MagicMock()
                        resp.choices = [MagicMock()]
                        resp.choices[0].message.content = "VALID"
                        resp.choices[0].finish_reason = "stop"
                        resp.usage = MagicMock(completion_tokens=5)
                        return resp
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "INVALID"
            resp.choices[0].finish_reason = "stop"
            resp.usage = MagicMock(completion_tokens=5)
            return resp

        with patch("strategies.multimodal_strategy.MultimodalChatClient") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance._client.chat.completions.create = AsyncMock(side_effect=_mock_create)
            MockClient.return_value = mock_client_instance

            s = MultimodalStrategy()
            s._search_provider = MagicMock()
            s._search_provider.image_data = {
                "figures/good-diagram.png": "good_base64",
                "figures/bad-cartoon.png": "bad_base64",
            }

            text = (
                "Step 1:\n![Good Diagram](figures/good-diagram.png)\n"
                "Step 2:\n![Bad Cartoon](figures/bad-cartoon.png)\n"
                "Step 3: done"
            )
            result = await s._validate_response_images(text, "rebuild engine")
            assert "![Good Diagram](figures/good-diagram.png)" in result
            assert "![Bad Cartoon]" not in result
            assert "Step 3: done" in result

    def test_validate_response_images_default_enabled(self):
        from strategies.multimodal_strategy import MultimodalStrategy
        s = MultimodalStrategy()
        assert s.validate_response_images is True
