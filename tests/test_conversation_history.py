"""Tests for the Conversation History Management feature.

Covers:
- schemas.py: ConversationMetadata, ConversationListResponse, ConversationDetail
- connectors/cosmosdb.py: partition_key support + standalone query functions
- orchestration/orchestrator.py: principal_id + partition key logic
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestConversationSchemas:
    def test_conversation_metadata_basic(self):
        from schemas import ConversationMetadata
        m = ConversationMetadata(id="abc-123", name="My Chat")
        assert m.id == "abc-123"
        assert m.name == "My Chat"
        assert m.created_at is None
        assert m.last_updated is None

    def test_conversation_metadata_with_alias(self):
        from schemas import ConversationMetadata
        m = ConversationMetadata(id="abc", _ts=1704718800, lastUpdated="2026-01-14T12:34:56")
        assert m.created_at is not None
        assert m.last_updated is not None

    def test_conversation_list_response(self):
        from schemas import ConversationListResponse, ConversationMetadata
        resp = ConversationListResponse(
            conversations=[ConversationMetadata(id="1", name="c1")],
            has_more=False,
            skip=0,
            limit=10,
        )
        assert len(resp.conversations) == 1
        assert resp.has_more is False
        assert resp.skip == 0
        assert resp.limit == 10

    def test_conversation_detail_with_messages(self):
        from schemas import ConversationDetail
        detail = ConversationDetail(
            id="conv-1",
            name="Test",
            principal_id="user-123",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert detail.id == "conv-1"
        assert detail.principal_id == "user-123"
        assert len(detail.messages) == 1

    def test_conversation_detail_empty_messages(self):
        from schemas import ConversationDetail
        detail = ConversationDetail(id="conv-1")
        assert detail.messages == []


# ---------------------------------------------------------------------------
# CosmosDB Client Tests (partition key support)
# ---------------------------------------------------------------------------

class TestCosmosDBPartitionKey:

    async def test_get_document_default_partition_key(self):
        """When no partition_key is provided, key is used as partition_key (backward compat)."""
        from connectors.cosmosdb import CosmosDBClient

        mock_container = MagicMock()
        mock_container.read_item = AsyncMock(return_value={"id": "doc-1"})

        with patch.object(CosmosDBClient, "__init__", lambda self: None):
            client = CosmosDBClient.__new__(CosmosDBClient)
            with patch.object(client, "_get_container", return_value=mock_container):
                result = await client.get_document("container", "doc-1")
                mock_container.read_item.assert_called_once_with(item="doc-1", partition_key="doc-1")
                assert result["id"] == "doc-1"

    async def test_get_document_custom_partition_key(self):
        """When partition_key is provided, it's used instead of key."""
        from connectors.cosmosdb import CosmosDBClient

        mock_container = MagicMock()
        mock_container.read_item = AsyncMock(return_value={"id": "doc-1"})

        with patch.object(CosmosDBClient, "__init__", lambda self: None):
            client = CosmosDBClient.__new__(CosmosDBClient)
            with patch.object(client, "_get_container", return_value=mock_container):
                result = await client.get_document("container", "doc-1", partition_key="user-xyz")
                mock_container.read_item.assert_called_once_with(item="doc-1", partition_key="user-xyz")

    async def test_create_document_with_partition_key(self):
        """create_document should set principal_id and lastUpdated in the body."""
        from connectors.cosmosdb import CosmosDBClient

        mock_container = MagicMock()
        mock_container.create_item = AsyncMock(return_value={"id": "doc-1"})

        with patch.object(CosmosDBClient, "__init__", lambda self: None):
            client = CosmosDBClient.__new__(CosmosDBClient)
            with patch.object(client, "_get_container", return_value=mock_container):
                body = {"name": "Test Conv"}
                await client.create_document("container", "doc-1", body=body, partition_key="user-abc")
                call_args = mock_container.create_item.call_args
                created = call_args.kwargs.get("body")
                assert created["id"] == "doc-1"
                assert created["principal_id"] == "user-abc"
                assert "lastUpdated" in created

    async def test_create_document_without_partition_key(self):
        """create_document without partition_key should not set principal_id."""
        from connectors.cosmosdb import CosmosDBClient

        mock_container = MagicMock()
        mock_container.create_item = AsyncMock(return_value={"id": "doc-1"})

        with patch.object(CosmosDBClient, "__init__", lambda self: None):
            client = CosmosDBClient.__new__(CosmosDBClient)
            with patch.object(client, "_get_container", return_value=mock_container):
                await client.create_document("container", "doc-1")
                call_args = mock_container.create_item.call_args
                created = call_args.kwargs.get("body")
                assert "principal_id" not in created

    async def test_update_document_adds_last_updated(self):
        """update_document should add lastUpdated timestamp."""
        from connectors.cosmosdb import CosmosDBClient

        mock_container = MagicMock()
        mock_container.replace_item = AsyncMock(return_value={"id": "doc-1", "lastUpdated": "2026-01-01"})

        with patch.object(CosmosDBClient, "__init__", lambda self: None):
            client = CosmosDBClient.__new__(CosmosDBClient)
            with patch.object(client, "_get_container", return_value=mock_container):
                doc = {"id": "doc-1", "data": "test"}
                await client.update_document("container", doc)
                assert "lastUpdated" in doc  # modified in-place before the call


# ---------------------------------------------------------------------------
# Standalone CosmosDB Query Function Tests
# ---------------------------------------------------------------------------

class TestCosmosDBStandaloneFunctions:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mock_container = MagicMock()
        self.mock_cfg = MagicMock()
        self.mock_cfg.get.side_effect = lambda key, default=None, **kw: {
            "CONVERSATIONS_DATABASE_CONTAINER": "conversations",
        }.get(key, default)

        self.mock_client = MagicMock()
        self.mock_client.cfg = self.mock_cfg
        self.mock_client._get_container = MagicMock(return_value=self.mock_container)
        yield

    async def test_query_user_conversations_no_filter(self):
        from connectors.cosmosdb import query_user_conversations

        async def fake_iter(*args, **kwargs):
            for doc in [{"id": "c1", "name": "Conv 1"}, {"id": "c2", "name": "Conv 2"}]:
                yield doc

        self.mock_container.query_items = MagicMock(return_value=fake_iter())

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await query_user_conversations("user-1", skip=0, limit=10)
            assert len(result) == 2
            assert result[0]["id"] == "c1"

    async def test_query_user_conversations_with_name_filter(self):
        from connectors.cosmosdb import query_user_conversations

        async def fake_iter(*args, **kwargs):
            for doc in [{"id": "c1", "name": "Budget"}]:
                yield doc

        self.mock_container.query_items = MagicMock(return_value=fake_iter())

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await query_user_conversations("user-1", skip=0, limit=10, name="Budget")
            assert len(result) == 1
            # Verify the query contained the name parameter
            call_args = self.mock_container.query_items.call_args
            params = call_args.kwargs.get("parameters") or call_args[1].get("parameters")
            param_names = [p["name"] for p in params]
            assert "@name" in param_names

    async def test_read_user_conversation_found(self):
        from connectors.cosmosdb import read_user_conversation

        self.mock_container.read_item = AsyncMock(return_value={
            "id": "conv-1", "principal_id": "user-1", "name": "Test"
        })

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await read_user_conversation("conv-1", "user-1")
            assert result is not None
            assert result["id"] == "conv-1"

    async def test_read_user_conversation_soft_deleted(self):
        from connectors.cosmosdb import read_user_conversation

        self.mock_container.read_item = AsyncMock(return_value={
            "id": "conv-1", "isDeleted": True
        })

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await read_user_conversation("conv-1", "user-1")
            assert result is None

    async def test_read_user_conversation_not_found(self):
        from connectors.cosmosdb import read_user_conversation

        self.mock_container.read_item = AsyncMock(side_effect=Exception("Not found"))

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await read_user_conversation("conv-1", "user-1")
            assert result is None

    async def test_update_conversation_name_success(self):
        from connectors.cosmosdb import update_conversation_name

        doc = {"id": "conv-1", "name": "Old Name", "principal_id": "user-1"}
        self.mock_container.read_item = AsyncMock(return_value=doc.copy())
        self.mock_container.replace_item = AsyncMock(return_value={**doc, "name": "New Name"})

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await update_conversation_name("conv-1", "user-1", "New Name")
            assert result is not None
            assert result["name"] == "New Name"

    async def test_update_conversation_name_soft_deleted(self):
        from connectors.cosmosdb import update_conversation_name

        self.mock_container.read_item = AsyncMock(return_value={
            "id": "conv-1", "isDeleted": True
        })

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await update_conversation_name("conv-1", "user-1", "New Name")
            assert result is None

    async def test_soft_delete_conversation_success(self):
        from connectors.cosmosdb import soft_delete_conversation

        doc = {"id": "conv-1", "name": "Test", "principal_id": "user-1"}
        self.mock_container.read_item = AsyncMock(return_value=doc.copy())
        self.mock_container.replace_item = AsyncMock(return_value={
            **doc, "isDeleted": True, "deletedAt": "2026-01-01"
        })

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await soft_delete_conversation("conv-1", "user-1")
            assert result is not None
            assert result["isDeleted"] is True

    async def test_soft_delete_conversation_not_found(self):
        from connectors.cosmosdb import soft_delete_conversation

        self.mock_container.read_item = AsyncMock(side_effect=Exception("Not found"))

        with patch("connectors.cosmosdb.get_cosmosdb_client", return_value=self.mock_client):
            result = await soft_delete_conversation("conv-1", "user-1")
            assert result is None


# ---------------------------------------------------------------------------
# Orchestrator Tests (principal_id + partition key logic)
# ---------------------------------------------------------------------------

class TestOrchestratorConversationHistory:
    @pytest.fixture(autouse=True)
    def _patch(self, patch_dependencies, mock_cosmos, mock_config):
        self.mock_cosmos = mock_cosmos
        self.mock_config = mock_config
        # Also patch get_cosmosdb_client where the orchestrator imports it
        with patch("orchestration.orchestrator.get_cosmosdb_client", return_value=mock_cosmos):
            yield

    async def test_create_with_principal_id(self):
        """Orchestrator.create should set principal_id from user_context."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()
            mock_strategy.initiate_agent_flow = AsyncMock(return_value=iter([]))
            mock_strat.return_value = mock_strategy

            orchestrator = await self._create_orchestrator(
                user_context={"principal_id": "user-abc-123"}
            )
            assert orchestrator.principal_id == "user-abc-123"

    async def test_create_anonymous_principal(self):
        """Orchestrator with empty principal_id should default to 'anonymous'."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()
            mock_strat.return_value = mock_strategy

            orchestrator = await self._create_orchestrator(
                user_context={"principal_id": ""}
            )
            assert orchestrator.principal_id == "anonymous"

    async def test_stream_response_new_conversation_authenticated(self):
        """New conversation for authenticated user should use principal_id as partition key."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()

            async def fake_flow(ask):
                yield "response chunk"

            mock_strategy.initiate_agent_flow = fake_flow
            mock_strategy.conversation = {}
            mock_strat.return_value = mock_strategy

            self.mock_cosmos.create_document = AsyncMock(return_value={"id": "new-id"})
            self.mock_cosmos.update_document = AsyncMock(return_value={})

            orchestrator = await self._create_orchestrator(
                user_context={"principal_id": "user-xyz"}
            )

            chunks = []
            async for chunk in orchestrator.stream_response("Hello"):
                chunks.append(chunk)

            # Should have called create_document with partition_key=principal_id
            self.mock_cosmos.create_document.assert_called_once()
            call_kwargs = self.mock_cosmos.create_document.call_args.kwargs
            assert call_kwargs.get("partition_key") == "user-xyz"

    async def test_stream_response_new_conversation_anonymous(self):
        """New conversation for anonymous user should use anonymous-{id} as partition key."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()

            async def fake_flow(ask):
                yield "chunk"

            mock_strategy.initiate_agent_flow = fake_flow
            mock_strategy.conversation = {}
            mock_strat.return_value = mock_strategy

            self.mock_cosmos.create_document = AsyncMock(return_value={"id": "new-id"})
            self.mock_cosmos.update_document = AsyncMock(return_value={})

            orchestrator = await self._create_orchestrator(
                user_context={"principal_id": "anonymous"}
            )

            chunks = []
            async for chunk in orchestrator.stream_response("Hello"):
                chunks.append(chunk)

            # Verify partition_key starts with "anonymous-"
            call_kwargs = self.mock_cosmos.create_document.call_args.kwargs
            pk = call_kwargs.get("partition_key", "")
            assert pk.startswith("anonymous-")

    async def test_stream_response_existing_conversation(self):
        """Loading an existing conversation should use the correct partition key."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()

            async def fake_flow(ask):
                yield "chunk"

            mock_strategy.initiate_agent_flow = fake_flow
            mock_strategy.conversation = {}
            mock_strat.return_value = mock_strategy

            existing_conv = {"id": "conv-123", "principal_id": "user-xyz", "name": "Test"}
            self.mock_cosmos.get_document = AsyncMock(return_value=existing_conv)
            self.mock_cosmos.update_document = AsyncMock(return_value=existing_conv)

            orchestrator = await self._create_orchestrator(
                conversation_id="conv-123",
                user_context={"principal_id": "user-xyz"},
            )

            chunks = []
            async for chunk in orchestrator.stream_response("Continue"):
                chunks.append(chunk)

            # Should have called get_document with partition_key
            self.mock_cosmos.get_document.assert_called_once()
            call_kwargs = self.mock_cosmos.get_document.call_args.kwargs
            assert call_kwargs.get("partition_key") == "user-xyz"

    async def test_save_feedback_with_partition_key(self):
        """save_feedback should use partition key when reading conversation."""
        with patch("orchestration.orchestrator.AgentStrategyFactory.get_strategy", new_callable=AsyncMock) as mock_strat:
            mock_strategy = MagicMock()
            mock_strat.return_value = mock_strategy

            existing_conv = {"id": "conv-123", "principal_id": "user-abc", "name": "Test"}
            self.mock_cosmos.get_document = AsyncMock(return_value=existing_conv)
            self.mock_cosmos.update_document = AsyncMock(return_value=existing_conv)

            orchestrator = await self._create_orchestrator(
                conversation_id="conv-123",
                user_context={"principal_id": "user-abc"},
            )

            feedback = {"is_positive": True, "feedback_text": "Great!"}
            await orchestrator.save_feedback(feedback)

            # Verify get_document was called with partition_key
            call_kwargs = self.mock_cosmos.get_document.call_args.kwargs
            assert call_kwargs.get("partition_key") == "user-abc"

    async def _create_orchestrator(self, conversation_id=None, user_context=None):
        from orchestration.orchestrator import Orchestrator
        return await Orchestrator.create(
            conversation_id=conversation_id,
            user_context=user_context or {},
        )
