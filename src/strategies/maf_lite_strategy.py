"""
Microsoft Agent Framework (MAF) Lite Strategy.

This strategy uses Microsoft Agent Framework with a direct Azure OpenAI model
connection — no Azure AI Foundry Agent Service V2 dependency.  It provides:
- Memory persistence for user profile (across sessions)
- Optional agentic search over documents
- Extensible context providers for custom capabilities
- Local conversation history (no server-side threads)
"""

import logging
import time
from typing import Optional

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure",
]:
    _logger = logging.getLogger(_azure_logger)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False
    _logger.disabled = True
    _logger.handlers.clear()

from agent_framework import ChatAgent, ChatMessage

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from .composite_context_provider import CompositeContextProvider
from .search_context_provider import SearchContextProvider
from .maf_plugins import UserProfile, UserProfileMemory
from connectors.openai_chat_client import OpenAIChatClient
from dependencies import get_config


class MafLiteStrategy(BaseAgentStrategy):
    """Agent strategy using Microsoft Agent Framework with direct Azure OpenAI.

    Unlike ``MafAgentServiceStrategy`` this strategy does NOT require Azure AI
    Foundry Agent Service V2.  The ``ChatAgent`` talks to the model deployment
    directly via :class:`OpenAIChatClient`, while still supporting user-profile
    memory and optional agentic search.
    """

    AGENT_INSTRUCTIONS = (
        "You are a helpful AI assistant. Your role is to assist users with their "
        "questions and tasks.\n\n"
        "Your capabilities:\n"
        "1. **Conversation**: Engage in helpful, informative conversations\n"
        "2. **Profile Awareness**: Remember user information to provide personalized assistance\n"
        "3. **Knowledge Search**: Search your knowledge base when relevant to answer questions\n\n"
        "Guidelines:\n"
        "- Provide clear, helpful, and accurate responses\n"
        "- Ask clarifying questions when needed\n"
        "- Be concise but thorough in your explanations"
    )

    def __init__(self):
        super().__init__()
        logging.debug("[MafLiteStrategy] Initializing...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.MAF_LITE

        if not hasattr(self, "credential") or self.credential is None:
            self.credential = cfg.aiocredential
            logging.debug("[MafLiteStrategy] Using credential from AppConfigClient")

        # Sync credential for OpenAI client (get_bearer_token_provider is sync)
        self._sync_credential = cfg.credential

        # Model endpoint — needed for direct OpenAI calls
        self.model_endpoint = cfg.get("AI_FOUNDRY_ACCOUNT_ENDPOINT")
        self.openai_api_version = cfg.get("OPENAI_API_VERSION", "2025-04-01-preview")

        # Azure AI Search configuration for retrieval
        self.search_endpoint = cfg.get_value("SEARCH_SERVICE_QUERY_ENDPOINT", allow_none=True)
        self.search_index_name = cfg.get_value("SEARCH_RAG_INDEX_NAME", allow_none=True)
        self.search_top_k = int(cfg.get("SEARCH_RAGINDEX_TOP_K", 3))
        self.semantic_search_config = cfg.get_value("SEARCH_SEMANTIC_SEARCH_CONFIG", allow_none=True)

        # Embedding configuration for hybrid (keyword + vector) search
        self.embedding_deployment = cfg.get_value("EMBEDDING_DEPLOYMENT_NAME", allow_none=True)

        # User profiles stored in the conversations container (no extra container needed)
        self.user_profile_container = cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")

        # Runtime state
        self._chat_client: Optional[OpenAIChatClient] = None
        self._user_memory: Optional[UserProfileMemory] = None
        self._search_provider: Optional[SearchContextProvider] = None

        logging.debug("[MafLiteStrategy] Initialized")

    # ------------------------------------------------------------------
    # Prompt namespace override — share prompts with MafAgentServiceStrategy
    # ------------------------------------------------------------------
    def _prompt_namespace(self) -> str:
        return "maf"

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------
    def _get_or_create_chat_client(self) -> OpenAIChatClient:
        if self._chat_client is None:
            logging.debug(
                "[MafLiteStrategy] Creating OpenAIChatClient "
                f"endpoint={self.model_endpoint} model={self.model_name}"
            )
            self._chat_client = OpenAIChatClient(
                azure_endpoint=self.model_endpoint,
                model_deployment_name=self.model_name,
                credential=self._sync_credential,
                api_version=self.openai_api_version,
            )
        return self._chat_client

    # ------------------------------------------------------------------
    # User profile persistence
    # ------------------------------------------------------------------
    async def _load_user_profile(self, user_id: str) -> UserProfile:
        profile_key = f"user_profile_{user_id}"
        try:
            doc = await self.cosmos.get_document(self.user_profile_container, profile_key)
            if doc and "profile_data" in doc:
                return UserProfile.model_validate_json(doc["profile_data"])
        except Exception as e:
            logging.debug(f"[MafLiteStrategy] No existing user profile found: {e}")
        return UserProfile()

    async def _save_user_profile(self, user_id: str, profile: UserProfile):
        profile_key = f"user_profile_{user_id}"
        try:
            doc = {
                "id": profile_key,
                "profile_data": profile.model_dump_json(),
                "updated_at": time.time(),
            }
            existing = await self.cosmos.get_document(self.user_profile_container, profile_key)
            if existing:
                await self.cosmos.update_document(self.user_profile_container, doc)
            else:
                await self.cosmos.create_document(self.user_profile_container, profile_key, body=doc)
            logging.info(f"[MafLiteStrategy] Saved user profile for {user_id}")
        except Exception as e:
            logging.error(f"[MafLiteStrategy] Failed to save user profile: {e}")

    # ------------------------------------------------------------------
    # Search provider (optional agentic retrieval)
    # ------------------------------------------------------------------
    async def _create_search_provider(self) -> Optional[SearchContextProvider]:
        if not self.search_endpoint:
            logging.debug("[MafLiteStrategy] No search endpoint configured, skipping search")
            return None
        if not self.search_index_name:
            logging.warning("[MafLiteStrategy] No search index name configured, skipping search")
            return None
        try:
            # Build async embed function for hybrid search
            embed_fn = None
            if self.embedding_deployment:
                chat_client = self._get_or_create_chat_client()
                async def _embed(text: str) -> list[float]:
                    resp = await chat_client._client.embeddings.create(
                        input=text, model=self.embedding_deployment
                    )
                    return resp.data[0].embedding
                embed_fn = _embed

            provider = SearchContextProvider(
                endpoint=self.search_endpoint,
                credential=self.credential,
                index_name=self.search_index_name,
                top_k=self.search_top_k,
                semantic_configuration_name=self.semantic_search_config,
                embed_fn=embed_fn,
            )
            logging.info(
                "[MafLiteStrategy] SearchContextProvider created (index=%s, top_k=%d)",
                self.search_index_name, self.search_top_k,
            )
            return provider
        except Exception as e:
            logging.error(f"[MafLiteStrategy] Failed to create search provider: {e}")
            return None

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------
    def _build_session_summary(self) -> str:
        parts = []
        if self._user_memory and self._user_memory.has_minimum_context():
            parts.append("**Your Profile:**")
            parts.append(self._user_memory._build_profile_summary())
        else:
            parts.append("**Your Profile:** Not yet configured.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main agent flow
    # ------------------------------------------------------------------
    async def initiate_agent_flow(self, user_message: str):
        """Run the conversational agent with direct Azure OpenAI model access."""
        flow_start = time.time()
        logging.info("[MafLiteStrategy] === Flow start === question=%r", user_message[:120])

        conv = self.conversation
        is_new_session = not conv.get("session_initialized", False)
        user_id = conv.get("user_id", "default_user")

        try:
            chat_client = self._get_or_create_chat_client()

            # Load or initialise user-profile memory
            if self._user_memory is None:
                t0 = time.time()
                user_profile = await self._load_user_profile(user_id)
                self._user_memory = UserProfileMemory(
                    chat_client=chat_client,
                    user_profile=user_profile,
                )
                logging.info("[MafLiteStrategy] user_profile_load: %.2fs (user=%s)", time.time() - t0, user_id)

            # Initialize search provider if not done
            if self._search_provider is None:
                t0 = time.time()
                self._search_provider = await self._create_search_provider()
                logging.info("[MafLiteStrategy] search_provider_init: %.2fs (hybrid=%s)", time.time() - t0, bool(self.embedding_deployment))

            # Build context providers
            context_providers = [self._user_memory]
            if self._search_provider:
                context_providers.append(self._search_provider)
            else:
                logging.warning("[MafLiteStrategy] No search provider — agent will answer without grounding")
            logging.info("[MafLiteStrategy] context_providers: %d", len(context_providers))

            # Read base instructions
            base_instructions = await self._read_prompt("main")
            instructions = base_instructions if base_instructions else self.AGENT_INSTRUCTIONS

            # Create agent and stream — no server-side thread needed
            async with ChatAgent(
                chat_client=chat_client,
                instructions=instructions,
                context_provider=CompositeContextProvider(context_providers),
            ) as agent:

                thread = agent.get_new_thread()

                # Session welcome with existing profile
                if is_new_session and self._user_memory.has_minimum_context():
                    conv["session_initialized"] = True
                    session_summary = self._build_session_summary()
                    yield f"Welcome back! Here's what I remember:\n\n{session_summary}\n\n---\n\n"
                elif is_new_session:
                    conv["session_initialized"] = True

                # Build message list with conversation history
                history = conv.get("messages", [])
                input_messages: list[ChatMessage] = []
                for msg in history[-10:]:
                    role = msg.get("role", "user")
                    text = msg.get("text") or msg.get("content") or ""
                    if text:
                        input_messages.append(ChatMessage(role=role, text=text))
                input_messages.append(ChatMessage(role="user", text=user_message))
                logging.info("[MafLiteStrategy] history_messages: %d (total input: %d)", len(history), len(input_messages))

                # Stream the agent response
                stream_start = time.time()
                full_response = ""
                async for chunk in agent.run_stream(input_messages, thread=thread):
                    if chunk.text:
                        full_response += chunk.text
                        yield chunk.text
                logging.info("[MafLiteStrategy] agent_stream: %.2fs (response_len=%d)", time.time() - stream_start, len(full_response))

                # Persist conversation history locally
                if "messages" not in conv:
                    conv["messages"] = []
                conv["messages"].append({"role": "user", "text": user_message})
                conv["messages"].append({"role": "assistant", "text": full_response})

            # Save updated profile
            t0 = time.time()
            await self._save_user_profile(user_id, self._user_memory.user_profile)
            logging.info("[MafLiteStrategy] user_profile_save: %.2fs", time.time() - t0)

            logging.info("[MafLiteStrategy] === Flow done === total: %.2fs", time.time() - flow_start)

        except Exception as e:
            logging.error(f"[MafLiteStrategy] Agent flow failed: {e}", exc_info=True)
            yield f"I encountered an error processing your request: {str(e)}. Please try again."

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    async def clear_session(self):
        conv = self.conversation
        conv["session_initialized"] = False
        conv["messages"] = []
        self._user_memory = None
        logging.info("[MafLiteStrategy] Session cleared")
