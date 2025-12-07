"""
Response Generator Component

This module provides the ResponseGenerator class for generating streaming LLM responses
with organization context, conversation history, and category-specific prompts.

Responsibilities:
- Build system prompts with organization context
- Build user prompts with augmented queries
- Stream responses from Claude with extended thinking
- Sanitize responses (remove storage URLs)
"""

import logging
from typing import Dict, Any

from langsmith import traceable
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from shared.prompts import (
    MARKETING_ANSWER_PROMPT,
    CREATIVE_BRIEF_PROMPT,
    MARKETING_PLAN_PROMPT,
    BRAND_POSITION_STATEMENT_PROMPT,
    CREATIVE_COPYWRITER_PROMPT,
    FA_HELPDESK_PROMPT,
)
from shared.util import get_verbosity_instruction

from .models import ConversationState
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates streaming LLM responses with context.

    Responsibilities:
    - Build system prompts with organization context
    - Build user prompts with augmented queries
    - Stream responses from Claude
    - Sanitize responses (remove storage URLs)
    """

    def __init__(
        self,
        claude_llm: ChatAnthropic,
        organization_data: Dict[str, Any],
        storage_url: str,
    ):
        """
        Initialize ResponseGenerator.

        Args:
            claude_llm: Anthropic Claude LLM instance
            organization_data: Organization information
            storage_url: Azure Storage URL for sanitization
        """
        self.claude_llm = claude_llm
        self.organization_data = organization_data
        self.storage_url = storage_url
        logger.info("[ResponseGenerator] Initialized")

    def build_system_prompt(
        self,
        state: ConversationState,
        context_builder: ContextBuilder,
        conversation_history: str,
        user_settings: Dict[str, Any],
    ) -> str:
        """
        Build system prompt with organization context and category-specific prompts.

        Includes:
        - Organization context (segments, brand, industry)
        - Conversation summary (condensed context from previous turns)
        - Retrieved context documents with citations
        - Conversation history
        - Category-specific prompts based on query category
        - Verbosity instructions based on user settings

        Args:
            state: Current conversation state
            context_builder: ContextBuilder instance
            conversation_history: Formatted conversation history
            user_settings: User preferences

        Returns:
            Complete system prompt
        """
        logger.debug("[ResponseGenerator] Building system prompt")

        # base prompt
        system_prompt = MARKETING_ANSWER_PROMPT

        # Add organization context
        org_context = context_builder.build_organization_context()
        system_prompt += f"\n\n{org_context}"

        if state.conversation_summary:
            system_prompt += f"""
                <----------- CONVERSATION SUMMARY ------------>
                The following is a summary of the conversation so far:
                {state.conversation_summary}
                <----------- END OF CONVERSATION SUMMARY ------------>
                """
            logger.debug(
                "[ResponseGenerator] Added conversation summary to system prompt"
            )

        # Add conversation history if available
        if conversation_history:
            system_prompt += f"""
                <----------- PROVIDED CHAT HISTORY ------------>
                {conversation_history}
                <----------- END OF PROVIDED CHAT HISTORY ------------>
                """
            logger.debug(
                "[ResponseGenerator] Added conversation history to system prompt"
            )

        # Add retrieved context documents if available
        if state.context_docs:
            context_str = "\n\n".join(str(doc) for doc in state.context_docs)
            system_prompt += f"""
                <----------- PROVIDED CONTEXT ------------>
                {context_str}
                <----------- END OF PROVIDED CONTEXT ------------>
                """
            logger.debug(
                f"[ResponseGenerator] Added {len(state.context_docs)} context documents to system prompt"
            )

        # Add category-specific prompt based on query category
        category_prompts = {
            "Creative Brief": CREATIVE_BRIEF_PROMPT,
            "Marketing Plan": MARKETING_PLAN_PROMPT,
            "Brand Positioning Statement": BRAND_POSITION_STATEMENT_PROMPT,
            "Creative Copywriter": CREATIVE_COPYWRITER_PROMPT,
            "Help Desk": FA_HELPDESK_PROMPT,
        }

        if state.query_category in category_prompts:
            category_prompt = category_prompts[state.query_category]
            system_prompt += f"""
                <----------- CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                {category_prompt}
                <----------- END OF CATEGORY-SPECIFIC INSTRUCTIONS ------------>
                """
            logger.debug(
                f"[ResponseGenerator] Added category-specific prompt for: {state.query_category}"
            )

        # Add verbosity instructions based on user settings
        verbosity_instruction = get_verbosity_instruction(user_settings)
        system_prompt += f"""
            <----------- VERBOSITY INSTRUCTIONS ------------>
            {verbosity_instruction}
            <----------- END OF VERBOSITY INSTRUCTIONS ------------>
            """
        logger.debug(
            "[ResponseGenerator] Added verbosity instructions to system prompt"
        )

        logger.info(
            f"[ResponseGenerator] Built system prompt with {len(system_prompt)} characters"
        )
        return system_prompt

    def build_user_prompt(
        self, state: ConversationState, user_settings: Dict[str, Any]
    ) -> str:
        """
        Build user prompt with original question and augmented query.

        Includes original question and augmented query based on detail_level setting:
        - "detailed": Include augmented query
        - "brief" or "balanced": Exclude augmented query

        Args:
            state: Current conversation state
            user_settings: User preferences

        Returns:
            Complete user prompt
        """
        logger.debug("[ResponseGenerator] Building user prompt")

        user_prompt = state.question

        # Check detail_level setting
        detail_level = user_settings.get("detail_level", "balanced")

        # Include augmented query only for "detailed" setting
        if detail_level == "detailed" and state.augmented_query:
            user_prompt += f"\n\nAugmented Query (with historical context): {state.augmented_query}"
            logger.debug(
                "[ResponseGenerator] Included augmented query in user prompt (detail_level: detailed)"
            )
        else:
            logger.debug(
                f"[ResponseGenerator] Excluded augmented query (detail_level: {detail_level})"
            )

        logger.info(
            f"[ResponseGenerator] Built user prompt with {len(user_prompt)} characters"
        )
        return user_prompt

    @traceable(run_type="llm", name="claude_generate_response")
    async def generate_streaming_response(self, system_prompt: str, user_prompt: str):
        """
        Generate streaming response from Claude with extended thinking.

        Uses Anthropic Claude (claude-sonnet-4-5-20250929) for streaming.
        Enables extended thinking but only yields text content (reasoning blocks are hidden).
        Temperature is fixed at 1.0 (set in LLM init) as required for extended thinking.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Yields:
            Response text tokens
        """
        logger.info("[ResponseGenerator] Starting streaming response generation")

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            logger.debug(
                "[ResponseGenerator] Invoking Claude with extended thinking and streaming enabled"
            )
            # Don't pass temperature to astream() - already set in LLM init (must be 1.0 for thinking)
            async for chunk in self.claude_llm.astream(messages):
                if hasattr(chunk, "content"):
                    # Content can be a string or a list of content blocks
                    if isinstance(chunk.content, str) and chunk.content:
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        # Handle content blocks (reasoning/thinking, text, etc.)
                        for block in chunk.content:
                            if isinstance(block, dict):
                                block_type = block.get("type")

                                # Only yield text blocks (skip reasoning blocks)
                                if block_type == "text":
                                    text_content = block.get("text", "")
                                    if text_content:
                                        yield text_content
                            elif isinstance(block, str) and block:
                                yield block

            logger.info("[ResponseGenerator] Completed streaming response generation")

        except Exception as e:
            logger.error(
                f"[ResponseGenerator] Error during streaming response generation: {e}"
            )
            error_message = "I apologize, but I encountered an error while generating the response. Please try again."
            yield error_message
            
