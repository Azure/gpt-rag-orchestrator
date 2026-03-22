"""
Microsoft Agent Framework (MAF) + Agent Service V2 Strategy.

This strategy implements a conversational agent using Microsoft Agent Framework
with Azure AI Foundry Agent Service V2 as the backend. It provides:
- Memory persistence for user profile (across sessions)
- Optional agentic search over documents via Agent Service V2
- Extensible context providers for custom capabilities
- Server-side thread management via Agent Service V2
"""

import logging
import time
from typing import Optional

# Suppress Azure SDK HTTP logging BEFORE importing azure packages
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    _logger = logging.getLogger(_azure_logger)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False
    _logger.disabled = True
    _logger.handlers.clear()

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from .composite_context_provider import CompositeContextProvider
from .search_context_provider import SearchContextProvider
from .maf_plugins import UserProfile, UserProfileMemory
from dependencies import get_config


# ============================================================================
# Main MAF Strategy
# ============================================================================

class MafAgentServiceStrategy(BaseAgentStrategy):
    """
    Agent strategy using Microsoft Agent Framework + Azure AI Foundry Agent Service V2.

    This strategy serves as a blank canvas for building custom agent capabilities:
    1. Maintaining persistent memory of the user's profile
    2. Optional agentic search over documents (via Agent Service V2)
    3. Extensible context providers for custom functionality

    Uses AzureAIAgentClient which communicates with Azure AI Foundry Agent Service V2
    for thread management and model inference.
    """

    AGENT_INSTRUCTIONS = """You are a helpful AI assistant. Your role is to assist users with their
questions and tasks.

Your capabilities:
1. **Conversation**: Engage in helpful, informative conversations
2. **Profile Awareness**: Remember user information to provide personalized assistance
3. **Knowledge Search**: Search your knowledge base when relevant to answer questions

Guidelines:
- Provide clear, helpful, and accurate responses
- Ask clarifying questions when needed
- Be concise but thorough in your explanations"""

    def __init__(self):
        """Initialize the MAF + Agent Service V2 strategy."""
        super().__init__()

        logging.debug("[MafAgentServiceStrategy] Initializing...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.MAF_AGENT_SERVICE

        # Ensure credential is set (use config's async credential as fallback)
        if not hasattr(self, 'credential') or self.credential is None:
            self.credential = cfg.aiocredential
            logging.debug("[MafAgentServiceStrategy] Using credential from AppConfigClient")

        # Azure AI Search configuration for retrieval
        self.search_endpoint = cfg.get_value("SEARCH_SERVICE_QUERY_ENDPOINT", allow_none=True)
        self.search_index_name = cfg.get_value("SEARCH_RAG_INDEX_NAME", allow_none=True)
        self.search_top_k = int(cfg.get("SEARCH_RAGINDEX_TOP_K", 3))
        self.semantic_search_config = cfg.get_value("SEARCH_SEMANTIC_SEARCH_CONFIG", allow_none=True)

        # User profiles stored in the conversations container (no extra container needed)
        self.user_profile_container = cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")

        # Runtime state
        self._agent: Optional[ChatAgent] = None
        self._search_provider: Optional[SearchContextProvider] = None

        logging.debug("[MafAgentServiceStrategy] Initialized")

    def _prompt_namespace(self) -> str:
        """Share prompts directory with MafLiteStrategy."""
        return "maf"

    async def _load_user_profile(self, user_id: str) -> UserProfile:
        """Load user profile from CosmosDB or return empty profile."""
        profile_key = f"user_profile_{user_id}"
        try:
            doc = await self.cosmos.get_document(self.user_profile_container, profile_key)
            if doc and "profile_data" in doc:
                return UserProfile.model_validate_json(doc["profile_data"])
        except Exception as e:
            logging.debug(f"[MafAgentServiceStrategy] No existing user profile found: {e}")
        return UserProfile()

    async def _save_user_profile(self, user_id: str, profile: UserProfile):
        """Save user profile to CosmosDB."""
        profile_key = f"user_profile_{user_id}"
        try:
            doc = {
                "id": profile_key,
                "profile_data": profile.model_dump_json(),
                "updated_at": time.time()
            }
            existing = await self.cosmos.get_document(self.user_profile_container, profile_key)
            if existing:
                await self.cosmos.update_document(self.user_profile_container, doc)
            else:
                await self.cosmos.create_document(self.user_profile_container, profile_key, body=doc)
            logging.info(f"[MafAgentServiceStrategy] Saved user profile for {user_id}")
        except Exception as e:
            logging.error(f"[MafAgentServiceStrategy] Failed to save user profile: {e}")

    async def _create_search_provider(self) -> Optional[SearchContextProvider]:
        """Create the search context provider for retrieval."""
        if not self.search_endpoint:
            logging.debug("[MafAgentServiceStrategy] No search endpoint configured, skipping search")
            return None
        if not self.search_index_name:
            logging.warning("[MafAgentServiceStrategy] No search index name configured, skipping search")
            return None

        try:
            provider = SearchContextProvider(
                endpoint=self.search_endpoint,
                credential=self.credential,
                index_name=self.search_index_name,
                top_k=self.search_top_k,
                semantic_configuration_name=self.semantic_search_config,
            )
            logging.info(
                "[MafAgentServiceStrategy] SearchContextProvider created (index=%s)",
                self.search_index_name,
            )
            return provider

        except Exception as e:
            logging.error(f"[MafAgentServiceStrategy] Failed to create search provider: {e}")
            return None

    def _build_session_summary(self, user_memory: UserProfileMemory) -> str:
        """Build a summary of loaded profiles for session start."""
        parts = []

        # User profile summary
        if user_memory.has_minimum_context():
            parts.append("**Your Profile:**")
            parts.append(user_memory._build_profile_summary())
        else:
            parts.append("**Your Profile:** Not yet configured.")

        return "\n".join(parts)

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiate the agent flow for a conversational interaction.

        Steps:
        1. Initialize/load memories for user profile
        2. Create agent with context providers
        3. If first message, provide session summary
        4. Process user message and stream response
        5. Save updated profile
        """
        flow_start = time.time()
        logging.debug(f"[MafAgentServiceStrategy] initiate_agent_flow called with: {user_message!r}")

        conv = self.conversation
        is_new_session = not conv.get("session_initialized", False)

        # Get user ID from conversation context
        user_id = conv.get("user_id", "default_user")

        try:
            t0 = time.time()
            user_profile = await self._load_user_profile(user_id)
            logging.info("[MafAgentServiceStrategy] user_profile_load: %.2fs (user=%s)", time.time() - t0, user_id)

            # Initialize search provider if not done
            if self._search_provider is None:
                self._search_provider = await self._create_search_provider()

            # Read base instructions
            base_instructions = await self._read_prompt("main")
            instructions = base_instructions if base_instructions else self.AGENT_INSTRUCTIONS

            user_memory: UserProfileMemory | None = None

            # Use request-scoped AzureAIAgentClient to guarantee async cleanup.
            async with AzureAIAgentClient(
                project_endpoint=self.project_endpoint,
                model_deployment_name=self.model_name,
                credential=self.credential,
            ) as client:
                user_memory = UserProfileMemory(
                    chat_client=client,
                    user_profile=user_profile,
                )

                context_providers = [user_memory]
                if self._search_provider:
                    context_providers.append(self._search_provider)

                async with ChatAgent(
                    chat_client=client,
                    instructions=instructions,
                    context_provider=CompositeContextProvider(context_providers),
                ) as agent:

                    # Get or create thread
                    thread_id = conv.get("thread_id")
                    if thread_id:
                        # Resume existing thread
                        thread = agent.get_new_thread(service_thread_id=thread_id)
                    else:
                        # Create new thread
                        thread = agent.get_new_thread()
                        # service_thread_id may be None until first run; we'll update after
                        if thread.service_thread_id:
                            conv["thread_id"] = thread.service_thread_id

                    # If new session with existing profile, provide summary
                    if is_new_session and user_memory.has_minimum_context():
                        conv["session_initialized"] = True
                        session_summary = self._build_session_summary(user_memory)
                        yield f"Welcome back! Here's what I remember:\n\n{session_summary}\n\n---\n\n"
                    elif is_new_session:
                        conv["session_initialized"] = True

                    # Stream the agent response
                    full_response = ""
                    async for chunk in agent.run_stream(user_message, thread=thread):
                        if chunk.text:
                            full_response += chunk.text
                            yield chunk.text

                    # Capture thread_id if it was set during the run
                    if not conv.get("thread_id") and thread.service_thread_id:
                        conv["thread_id"] = thread.service_thread_id

                    # Store in conversation history
                    if "messages" not in conv:
                        conv["messages"] = []
                    conv["messages"].append({"role": "user", "text": user_message})
                    conv["messages"].append({"role": "assistant", "text": full_response})

            if user_memory is not None:
                await self._save_user_profile(user_id, user_memory.user_profile)

            logging.info(f"[MafAgentServiceStrategy] Flow completed in {round(time.time() - flow_start, 2)}s")

        except Exception as e:
            logging.error(f"[MafAgentServiceStrategy] Agent flow failed: {e}", exc_info=True)
            yield f"I encountered an error processing your request: {str(e)}. Please try again."

    async def clear_session(self):
        """Clear the current session state (but preserve persisted profile)."""
        conv = self.conversation
        conv["session_initialized"] = False
        conv["thread_id"] = None
        conv["messages"] = []

        self._agent = None

        logging.info("[MafAgentServiceStrategy] Session cleared")
