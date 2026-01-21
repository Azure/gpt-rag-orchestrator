"""
Negotiation Preparation Strategy using Microsoft Agent Framework with Azure AI Foundry.

This strategy implements a negotiation preparation system with:
- Memory persistence for user negotiation profile (across sessions)
- Memory persistence for target buyer profile (across sessions)
- Agentic search over negotiation tactics and strategies documents
- Strategic question answering and negotiation advice
"""

import logging
import time
from collections.abc import MutableSequence, Sequence
from typing import Any, Optional, Dict, List

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

from pydantic import BaseModel, Field
from agent_framework import ContextProvider, Context, ChatAgent, ChatClientProtocol, ChatMessage, ChatOptions
from agent_framework.azure import AzureAIAgentClient, AzureAISearchContextProvider

from .base_agent_strategy import BaseAgentStrategy
from .agent_strategies import AgentStrategies
from dependencies import get_config


# ============================================================================
# Pydantic Models for Structured Memory
# ============================================================================

class UserNegotiationProfile(BaseModel):
    """User's negotiation profile - persisted across sessions."""
    name: Optional[str] = Field(default=None, description="User's name")
    role: Optional[str] = Field(default=None, description="User's role/title (e.g., Account Executive, Sales Manager)")
    company: Optional[str] = Field(default=None, description="User's company name")
    negotiation_style: Optional[str] = Field(default=None, description="User's preferred negotiation style")
    strengths: List[str] = Field(default_factory=list, description="User's negotiation strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas the user wants to improve")
    past_experiences: List[str] = Field(default_factory=list, description="Notable past negotiation experiences")
    goals: List[str] = Field(default_factory=list, description="User's negotiation goals")

class BuyerProfile(BaseModel):
    """Target buyer's profile - persisted across sessions."""
    name: Optional[str] = Field(default=None, description="Buyer's name")
    title: Optional[str] = Field(default=None, description="Buyer's title (e.g., Senior Director, CFO)")
    company: Optional[str] = Field(default=None, description="Buyer's company")
    negotiation_style: Optional[str] = Field(default=None, description="Observed negotiation style")
    personality_traits: List[str] = Field(default_factory=list, description="Personality traits (e.g., aggressive, analytical)")
    communication_patterns: List[str] = Field(default_factory=list, description="Communication patterns (e.g., tends to over-talk)")
    known_priorities: List[str] = Field(default_factory=list, description="Known priorities and concerns")
    past_interactions: List[str] = Field(default_factory=list, description="Notes from past interactions")
    decision_making_style: Optional[str] = Field(default=None, description="How they make decisions")

class ExtractedUserInfo(BaseModel):
    """Structured extraction of user profile information from conversation."""
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None
    negotiation_style: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    past_experiences: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)

class ExtractedBuyerInfo(BaseModel):
    """Structured extraction of buyer profile information from conversation."""
    name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    negotiation_style: Optional[str] = None
    personality_traits: List[str] = Field(default_factory=list)
    communication_patterns: List[str] = Field(default_factory=list)
    known_priorities: List[str] = Field(default_factory=list)
    past_interactions: List[str] = Field(default_factory=list)
    decision_making_style: Optional[str] = None


# ============================================================================
# Memory Context Providers
# ============================================================================

class UserProfileMemory(ContextProvider):
    """
    Context provider that maintains the user's negotiation profile.

    Extracts user information from conversations and provides context
    about the user to guide the agent's responses.
    """

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        user_profile: Optional[UserNegotiationProfile] = None,
        **kwargs: Any
    ):
        self._chat_client = chat_client
        if user_profile:
            self.user_profile = user_profile
        elif kwargs:
            self.user_profile = UserNegotiationProfile.model_validate(kwargs)
        else:
            self.user_profile = UserNegotiationProfile()

    def has_minimum_context(self) -> bool:
        """Check if we have minimum required user profile information."""
        return bool(self.user_profile.name and self.user_profile.role)

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        _response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        _invoke_exception: Exception | None = None,
        **_kwargs: Any,
    ) -> None:
        """Extract user profile information from messages after each agent call."""
        messages_list = [request_messages] if isinstance(request_messages, ChatMessage) else list(request_messages)
        user_messages = [msg for msg in messages_list if msg.role.value == "user"]

        if not user_messages:
            return

        try:
            result = await self._chat_client.get_response(
                messages=messages_list,
                chat_options=ChatOptions(
                    instructions=(
                        "Extract any information about the USER (the seller/salesperson) from the conversation. "
                        "Look for: their name, role/title, company, negotiation style preferences, "
                        "strengths, areas they want to improve, past experiences, and goals. "
                        "Only extract information that is explicitly stated about the user themselves. "
                        "Return nulls/empty lists for fields not mentioned."
                    ),
                    response_format=ExtractedUserInfo,
                ),
            )

            if result.value and isinstance(result.value, ExtractedUserInfo):
                extracted = result.value
                # Update profile with new information (don't overwrite with None)
                if extracted.name:
                    self.user_profile.name = extracted.name
                if extracted.role:
                    self.user_profile.role = extracted.role
                if extracted.company:
                    self.user_profile.company = extracted.company
                if extracted.negotiation_style:
                    self.user_profile.negotiation_style = extracted.negotiation_style
                if extracted.strengths:
                    self.user_profile.strengths.extend(
                        s for s in extracted.strengths if s not in self.user_profile.strengths
                    )
                if extracted.areas_for_improvement:
                    self.user_profile.areas_for_improvement.extend(
                        a for a in extracted.areas_for_improvement
                        if a not in self.user_profile.areas_for_improvement
                    )
                if extracted.past_experiences:
                    self.user_profile.past_experiences.extend(
                        e for e in extracted.past_experiences
                        if e not in self.user_profile.past_experiences
                    )
                if extracted.goals:
                    self.user_profile.goals.extend(
                        g for g in extracted.goals if g not in self.user_profile.goals
                    )

                logging.debug(f"[UserProfileMemory] Updated user profile: {self.user_profile}")

        except Exception as e:
            logging.warning(f"[UserProfileMemory] Failed to extract user info: {e}")

    async def invoking(
        self,
        _messages: ChatMessage | MutableSequence[ChatMessage],
        **_kwargs: Any
    ) -> Context:
        """Provide user profile context before each agent call."""
        instructions: List[str] = []

        if not self.has_minimum_context():
            instructions.append(
                "The user's negotiation profile is incomplete. "
                "Politely ask them to share: their name, role/title, and any relevant background "
                "about their negotiation experience or style preferences."
            )
        else:
            profile_summary = self._build_profile_summary()
            instructions.append(f"User Profile:\n{profile_summary}")

        return Context(instructions="\n".join(instructions))

    def _build_profile_summary(self) -> str:
        """Build a formatted summary of the user profile."""
        parts = []
        p = self.user_profile

        if p.name:
            parts.append(f"- Name: {p.name}")
        if p.role:
            parts.append(f"- Role: {p.role}")
        if p.company:
            parts.append(f"- Company: {p.company}")
        if p.negotiation_style:
            parts.append(f"- Negotiation Style: {p.negotiation_style}")
        if p.strengths:
            parts.append(f"- Strengths: {', '.join(p.strengths)}")
        if p.areas_for_improvement:
            parts.append(f"- Areas to Improve: {', '.join(p.areas_for_improvement)}")
        if p.goals:
            parts.append(f"- Goals: {', '.join(p.goals)}")
        if p.past_experiences:
            parts.append(f"- Past Experiences: {'; '.join(p.past_experiences[:3])}")  # Limit to 3

        return "\n".join(parts) if parts else "No user profile information available."

    def serialize(self) -> str:
        """Serialize the user profile for persistence."""
        return self.user_profile.model_dump_json()

    @classmethod
    def deserialize(cls, data: str, chat_client: ChatClientProtocol) -> "UserProfileMemory":
        """Deserialize a user profile from stored data."""
        profile = UserNegotiationProfile.model_validate_json(data)
        return cls(chat_client=chat_client, user_profile=profile)


class BuyerProfileMemory(ContextProvider):
    """
    Context provider that maintains the target buyer's profile.

    Extracts buyer information from conversations and provides context
    about the buyer to guide negotiation strategy recommendations.
    """

    def __init__(
        self,
        chat_client: ChatClientProtocol,
        buyer_profile: Optional[BuyerProfile] = None,
        buyer_id: Optional[str] = None,
        **kwargs: Any
    ):
        self._chat_client = chat_client
        self.buyer_id = buyer_id  # Unique identifier for this buyer
        if buyer_profile:
            self.buyer_profile = buyer_profile
        elif kwargs:
            self.buyer_profile = BuyerProfile.model_validate(kwargs)
        else:
            self.buyer_profile = BuyerProfile()

    def has_minimum_context(self) -> bool:
        """Check if we have minimum required buyer profile information."""
        return bool(self.buyer_profile.name or self.buyer_profile.title)

    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        _response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        _invoke_exception: Exception | None = None,
        **_kwargs: Any,
    ) -> None:
        """Extract buyer profile information from messages after each agent call."""
        messages_list = [request_messages] if isinstance(request_messages, ChatMessage) else list(request_messages)
        user_messages = [msg for msg in messages_list if msg.role.value == "user"]

        if not user_messages:
            return

        try:
            result = await self._chat_client.get_response(
                messages=messages_list,
                chat_options=ChatOptions(
                    instructions=(
                        "Extract any information about the TARGET BUYER (the person the user will negotiate with) "
                        "from the conversation. Look for: buyer's name, title/role, company, "
                        "negotiation style, personality traits (e.g., aggressive, analytical), "
                        "communication patterns (e.g., tends to over-talk, interrupts), "
                        "known priorities, past interactions, and decision-making style. "
                        "Only extract information explicitly stated about the buyer. "
                        "Return nulls/empty lists for fields not mentioned."
                    ),
                    response_format=ExtractedBuyerInfo,
                ),
            )

            if result.value and isinstance(result.value, ExtractedBuyerInfo):
                extracted = result.value
                # Update profile with new information
                if extracted.name:
                    self.buyer_profile.name = extracted.name
                if extracted.title:
                    self.buyer_profile.title = extracted.title
                if extracted.company:
                    self.buyer_profile.company = extracted.company
                if extracted.negotiation_style:
                    self.buyer_profile.negotiation_style = extracted.negotiation_style
                if extracted.decision_making_style:
                    self.buyer_profile.decision_making_style = extracted.decision_making_style
                if extracted.personality_traits:
                    self.buyer_profile.personality_traits.extend(
                        t for t in extracted.personality_traits
                        if t not in self.buyer_profile.personality_traits
                    )
                if extracted.communication_patterns:
                    self.buyer_profile.communication_patterns.extend(
                        p for p in extracted.communication_patterns
                        if p not in self.buyer_profile.communication_patterns
                    )
                if extracted.known_priorities:
                    self.buyer_profile.known_priorities.extend(
                        p for p in extracted.known_priorities
                        if p not in self.buyer_profile.known_priorities
                    )
                if extracted.past_interactions:
                    self.buyer_profile.past_interactions.extend(
                        i for i in extracted.past_interactions
                        if i not in self.buyer_profile.past_interactions
                    )

                logging.debug(f"[BuyerProfileMemory] Updated buyer profile: {self.buyer_profile}")

        except Exception as e:
            logging.warning(f"[BuyerProfileMemory] Failed to extract buyer info: {e}")

    async def invoking(
        self,
        _messages: ChatMessage | MutableSequence[ChatMessage],
        **_kwargs: Any
    ) -> Context:
        """Provide buyer profile context before each agent call."""
        instructions: List[str] = []

        if not self.has_minimum_context():
            instructions.append(
                "The target buyer's profile is incomplete. "
                "Ask the user to share information about the buyer they'll be negotiating with: "
                "their name, title/role, company, and any known personality traits, "
                "communication style, or past interaction notes."
            )
        else:
            profile_summary = self._build_profile_summary()
            instructions.append(f"Target Buyer Profile:\n{profile_summary}")

        return Context(instructions="\n".join(instructions))

    def _build_profile_summary(self) -> str:
        """Build a formatted summary of the buyer profile."""
        parts = []
        b = self.buyer_profile

        if b.name:
            parts.append(f"- Name: {b.name}")
        if b.title:
            parts.append(f"- Title: {b.title}")
        if b.company:
            parts.append(f"- Company: {b.company}")
        if b.negotiation_style:
            parts.append(f"- Negotiation Style: {b.negotiation_style}")
        if b.personality_traits:
            parts.append(f"- Personality Traits: {', '.join(b.personality_traits)}")
        if b.communication_patterns:
            parts.append(f"- Communication Patterns: {', '.join(b.communication_patterns)}")
        if b.decision_making_style:
            parts.append(f"- Decision Making: {b.decision_making_style}")
        if b.known_priorities:
            parts.append(f"- Known Priorities: {', '.join(b.known_priorities)}")
        if b.past_interactions:
            parts.append(f"- Past Interactions: {'; '.join(b.past_interactions[:3])}")

        return "\n".join(parts) if parts else "No buyer profile information available."

    def serialize(self) -> str:
        """Serialize the buyer profile for persistence."""
        return self.buyer_profile.model_dump_json()

    @classmethod
    def deserialize(cls, data: str, chat_client: ChatClientProtocol, buyer_id: str = None) -> "BuyerProfileMemory":
        """Deserialize a buyer profile from stored data."""
        profile = BuyerProfile.model_validate_json(data)
        return cls(chat_client=chat_client, buyer_profile=profile, buyer_id=buyer_id)


# ============================================================================
# Main Negotiation Strategy
# ============================================================================

class NegotiationStrategy(BaseAgentStrategy):
    """
    Negotiation preparation strategy using Microsoft Agent Framework.

    This strategy helps users prepare for negotiations by:
    1. Maintaining persistent memory of the user's negotiation profile
    2. Maintaining persistent memory of target buyer profiles
    3. Providing strategic advice using agentic search over negotiation documents
    4. Loading and summarizing profiles at session start
    """

    AGENT_INSTRUCTIONS = """You are an expert negotiation coach and strategist. Your role is to help
the user (a seller/salesperson) prepare for and excel in negotiations with their target buyers.

Your capabilities:
1. **Profile Management**: Help users build and maintain their negotiation profile and buyer profiles
2. **Strategic Advice**: Provide tailored negotiation strategies based on user and buyer profiles
3. **Tactics & Techniques**: Search and recommend specific negotiation tactics from your knowledge base
4. **Preparation Guidance**: Help users prepare talking points, anticipate objections, and plan responses

Guidelines:
- Always consider both the user's profile and the buyer's profile when giving advice
- Provide specific, actionable recommendations rather than generic advice
- Reference proven negotiation frameworks and techniques when appropriate
- Help the user anticipate the buyer's likely moves and prepare counter-strategies
- Be encouraging but realistic about challenges

When profiles are incomplete, gather the necessary information before providing strategic advice.
When answering strategic questions, search your knowledge base for relevant tactics and strategies."""

    def __init__(self):
        """Initialize the negotiation strategy."""
        super().__init__()

        logging.debug("[NegotiationStrategy] Initializing...")

        cfg = get_config()
        self.strategy_type = AgentStrategies.NEGOTIATION

        # Ensure credential is set (use config's async credential as fallback)
        if not hasattr(self, 'credential') or self.credential is None:
            self.credential = cfg.aiocredential
            logging.debug("[NegotiationStrategy] Using credential from AppConfigClient")

        # Azure AI Search configuration for agentic retrieval
        self.search_endpoint = cfg.get("SEARCH_SERVICE_QUERY_ENDPOINT")
        self.search_knowledge_base = cfg.get("NEGOTIATION_KNOWLEDGE_BASE", "negotiation-tactics")
        self.search_index_name = cfg.get("NEGOTIATION_SEARCH_INDEX", 'ragindex-z25ekhr3dyuju')

        # Memory storage keys (in CosmosDB)
        self.user_profile_container = "negotiation_user_profiles"
        self.buyer_profile_container = "negotiation_buyer_profiles"

        # Runtime state
        self._agent: Optional[ChatAgent] = None
        self._chat_client: Optional[AzureAIAgentClient] = None
        self._user_memory: Optional[UserProfileMemory] = None
        self._buyer_memory: Optional[BuyerProfileMemory] = None
        self._search_provider: Optional[AzureAISearchContextProvider] = None

        logging.debug("[NegotiationStrategy] Initialized")

    async def _get_or_create_chat_client(self) -> AzureAIAgentClient:
        """Get or create the Azure AI Agent client."""
        if self._chat_client is None:
            logging.debug(f"[NegotiationStrategy] Creating AzureAIAgentClient with endpoint: {self.project_endpoint}")
            logging.debug(f"[NegotiationStrategy] Credential type: {type(self.credential)}")
            self._chat_client = AzureAIAgentClient(
                project_endpoint=self.project_endpoint,
                model_deployment_name=self.model_name,
                credential=self.credential,
            )
        return self._chat_client

    async def _load_user_profile(self, user_id: str) -> UserNegotiationProfile:
        """Load user profile from CosmosDB or return empty profile."""
        try:
            doc = await self.cosmos.get_document(self.user_profile_container, user_id)
            if doc and "profile_data" in doc:
                return UserNegotiationProfile.model_validate_json(doc["profile_data"])
        except Exception as e:
            logging.debug(f"[NegotiationStrategy] No existing user profile found: {e}")
        return UserNegotiationProfile()

    async def _save_user_profile(self, user_id: str, profile: UserNegotiationProfile):
        """Save user profile to CosmosDB."""
        try:
            doc = {
                "id": user_id,
                "profile_data": profile.model_dump_json(),
                "updated_at": time.time()
            }
            await self.cosmos.upsert_document(self.user_profile_container, doc)
            logging.debug(f"[NegotiationStrategy] Saved user profile for {user_id}")
        except Exception as e:
            logging.error(f"[NegotiationStrategy] Failed to save user profile: {e}")

    async def _load_buyer_profile(self, buyer_id: str) -> BuyerProfile:
        """Load buyer profile from CosmosDB or return empty profile."""
        try:
            doc = await self.cosmos.get_document(self.buyer_profile_container, buyer_id)
            if doc and "profile_data" in doc:
                return BuyerProfile.model_validate_json(doc["profile_data"])
        except Exception as e:
            logging.debug(f"[NegotiationStrategy] No existing buyer profile found: {e}")
        return BuyerProfile()

    async def _save_buyer_profile(self, buyer_id: str, profile: BuyerProfile):
        """Save buyer profile to CosmosDB."""
        try:
            doc = {
                "id": buyer_id,
                "profile_data": profile.model_dump_json(),
                "updated_at": time.time()
            }
            await self.cosmos.upsert_document(self.buyer_profile_container, doc)
            logging.debug(f"[NegotiationStrategy] Saved buyer profile for {buyer_id}")
        except Exception as e:
            logging.error(f"[NegotiationStrategy] Failed to save buyer profile: {e}")

    async def _create_search_provider(self) -> Optional[AzureAISearchContextProvider]:
        """Create the Azure AI Search context provider for agentic retrieval."""
        if not self.search_endpoint:
            logging.warning("[NegotiationStrategy] No search endpoint configured, skipping agentic search")
            return None

        try:
            search_config = {
                "endpoint": self.search_endpoint,
                "credential": self.credential,
                "mode": "agentic",
                "retrieval_reasoning_effort": "medium",
            }

            # Use knowledge base or index name based on configuration
            if self.search_knowledge_base:
                search_config["knowledge_base_name"] = self.search_knowledge_base
            elif self.search_index_name:
                search_config["index_name"] = self.search_index_name

            return AzureAISearchContextProvider(**search_config)

        except Exception as e:
            logging.error(f"[NegotiationStrategy] Failed to create search provider: {e}")
            return None

    def _build_session_summary(self) -> str:
        """Build a summary of loaded profiles for session start."""
        parts = []

        # User profile summary
        if self._user_memory and self._user_memory.has_minimum_context():
            parts.append("**Your Negotiation Profile:**")
            parts.append(self._user_memory._build_profile_summary())
        else:
            parts.append("**Your Negotiation Profile:** Not yet configured. Please share some information about yourself.")

        parts.append("")

        # Buyer profile summary
        if self._buyer_memory and self._buyer_memory.has_minimum_context():
            parts.append("**Target Buyer Profile:**")
            parts.append(self._buyer_memory._build_profile_summary())
        else:
            parts.append("**Target Buyer Profile:** Not yet configured. Please share information about the buyer you'll be negotiating with.")

        return "\n".join(parts)

    async def initiate_agent_flow(self, user_message: str):
        """
        Initiate the negotiation agent flow.

        Steps:
        1. Initialize/load memories for user and buyer profiles
        2. Create agent with context providers
        3. If first message, provide session summary
        4. Process user message and stream response
        5. Save updated profiles
        """
        flow_start = time.time()
        logging.debug(f"[NegotiationStrategy] initiate_agent_flow called with: {user_message!r}")

        conv = self.conversation
        is_new_session = not conv.get("session_initialized", False)

        # Get user and buyer IDs from conversation context
        user_id = conv.get("user_id", "default_user")
        buyer_id = conv.get("buyer_id", "default_buyer")

        try:
            # Initialize chat client
            chat_client = await self._get_or_create_chat_client()

            # Load or initialize user profile memory
            if self._user_memory is None:
                user_profile = await self._load_user_profile(user_id)
                self._user_memory = UserProfileMemory(
                    chat_client=chat_client,
                    user_profile=user_profile
                )
                logging.info(f"[NegotiationStrategy] Loaded user profile for {user_id}")

            # Load or initialize buyer profile memory
            if self._buyer_memory is None:
                buyer_profile = await self._load_buyer_profile(buyer_id)
                self._buyer_memory = BuyerProfileMemory(
                    chat_client=chat_client,
                    buyer_profile=buyer_profile,
                    buyer_id=buyer_id
                )
                logging.info(f"[NegotiationStrategy] Loaded buyer profile for {buyer_id}")

            # Initialize search provider if not done
            if self._search_provider is None:
                self._search_provider = await self._create_search_provider()

            # Build context providers list
            context_providers = [self._user_memory, self._buyer_memory]
            if self._search_provider:
                context_providers.append(self._search_provider)

            # Read base instructions
            base_instructions = await self._read_prompt("negotiation_coach")
            instructions = base_instructions if base_instructions else self.AGENT_INSTRUCTIONS

            # Create or reuse agent
            async with AzureAIAgentClient(
                project_endpoint=self.project_endpoint,
                model_deployment_name=self.model_name,
                credential=self.credential,
            ) as client:
                async with ChatAgent(
                    chat_client=client,
                    instructions=instructions,
                    context_providers=context_providers,
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

                    # If new session, provide summary first
                    if is_new_session:
                        conv["session_initialized"] = True
                        session_summary = self._build_session_summary()
                        yield f"Welcome back! Here's what I remember:\n\n{session_summary}\n\n---\n\n"

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

            # Save updated profiles
            await self._save_user_profile(user_id, self._user_memory.user_profile)
            await self._save_buyer_profile(buyer_id, self._buyer_memory.buyer_profile)

            logging.info(f"[NegotiationStrategy] Flow completed in {round(time.time() - flow_start, 2)}s")

        except Exception as e:
            logging.error(f"[NegotiationStrategy] Agent flow failed: {e}", exc_info=True)
            yield f"I encountered an error processing your request: {str(e)}. Please try again."

    async def set_buyer_context(self, buyer_id: str):
        """
        Switch to a different buyer context.

        Call this when the user wants to prepare for a negotiation
        with a different buyer.
        """
        conv = self.conversation
        conv["buyer_id"] = buyer_id

        # Reset buyer memory to load new profile
        self._buyer_memory = None

        logging.info(f"[NegotiationStrategy] Switched buyer context to: {buyer_id}")

    async def get_buyer_list(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get list of all buyer profiles for a user.

        Returns list of buyer summaries with id, name, title, company.
        """
        try:
            # Query CosmosDB for all buyer profiles
            # This assumes a query method exists - adjust based on actual CosmosDB client
            query = f"SELECT c.id, c.profile_data FROM c WHERE c.user_id = '{user_id}'"
            results = await self.cosmos.query_documents(self.buyer_profile_container, query)

            buyers = []
            for doc in results:
                profile = BuyerProfile.model_validate_json(doc.get("profile_data", "{}"))
                buyers.append({
                    "id": doc["id"],
                    "name": profile.name,
                    "title": profile.title,
                    "company": profile.company
                })
            return buyers

        except Exception as e:
            logging.error(f"[NegotiationStrategy] Failed to get buyer list: {e}")
            return []

    async def clear_session(self):
        """Clear the current session state (but preserve persisted profiles)."""
        conv = self.conversation
        conv["session_initialized"] = False
        conv["thread_id"] = None
        conv["messages"] = []

        self._agent = None
        self._user_memory = None
        self._buyer_memory = None

        logging.info("[NegotiationStrategy] Session cleared")
