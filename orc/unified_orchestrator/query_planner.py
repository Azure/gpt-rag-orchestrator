"""
Query Planning Component

This module handles query processing operations including rewriting, augmentation,
and categorization for the unified orchestrator.

The QueryPlanner class uses an LLM to:
- Rewrite queries with organization context and segment aliases
- Augment queries with conversation history
- Categorize queries into marketing categories
"""

import logging
from typing import Dict, Any

from langsmith import traceable
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from shared.prompts import (
    QUERY_REWRITING_PROMPT,
    AUGMENTED_QUERY_PROMPT,
)
from .models import ConversationState
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class QueryPlanner:
    """
    Handles query processing: rewriting, augmentation, and categorization.

    Responsibilities:
    - Rewrite queries with organization context
    - Augment queries with conversation history
    - Categorize queries into proper marketing categories
    """

    def __init__(self, llm: AzureChatOpenAI, organization_data: Dict[str, Any]):
        """
        Initialize QueryPlanner.

        Args:
            llm: Azure OpenAI LLM instance for planning tasks
            organization_data: Organization information
        """
        self.llm = llm
        self.organization_data = organization_data
        logger.info("[QueryPlanner] Initialized")

    @traceable(run_type="llm", name="query_rewrite")
    async def rewrite_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Rewrite query with organization context and segment aliases.

        Uses llm planning model to rewrite the user's query with:
        - Organization context (segments, brand, industry)
        - Conversation history for context
        - Segment alias mappings to use official terminology

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance

        Returns:
            Dictionary with rewritten_query
        """
        logger.info(f"[QueryPlanner] Rewriting query: {state.question[:50]}...")

        question = state.question
        history = conversation_data.get("history", [])
        logger.info(
            f"[QueryPlanner] Retrieved {len(history)} messages from conversation history"
        )

        # Build system prompt with organization context including history
        system_prompt = f"""
        {QUERY_REWRITING_PROMPT}
        {context_builder.build_organization_context(history)}"""

        # Build user prompt
        user_prompt = f"""
        Please rewrite the query to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        Original query: 
        <Original query>
        {question}
        </Original query>

        THE HISTORICAL CONVERSATION IS VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, YOU MUST TAKE THAT INTO ACCOUNT.
        Please also consider the line of business/industry of my company when rewriting the query if relevant. However, do not completely alter the query's original intention. Do not be verbose. 
        if the question is a very casual/conversational one, do not rewrite, return it as it is
        """

        logger.info("[QueryPlanner] Sending query rewrite request to LLM")

        try:
            rewritten_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            rewritten_query = rewritten_response.content
            logger.info(
                f"[QueryPlanner] Successfully rewrote query: '{rewritten_query[:100]}...'"
            )
        except Exception as e:
            logger.error(f"[QueryPlanner] Error rewriting query: {e}")
            rewritten_query = question

        return {"rewritten_query": rewritten_query}

    @traceable(run_type="llm", name="query_augment")
    async def augment_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Augment query with conversation history.

        Uses gpt-4.1 to augment the query with historical context for improved
        understanding. Handles casual/conversational queries by returning them as-is.

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance for formatting history

        Returns:
            Dictionary with augmented_query
        """
        logger.info(f"[QueryPlanner] Augmenting query: {state.question[:50]}...")

        question = state.question
        history = conversation_data.get("history", [])
        formatted_history = context_builder.format_conversation_history(history)

        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {formatted_history}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text format only, no additional text, explanations, or formatting.
        """

        logger.info("[QueryPlanner] Sending augmented query request to LLM")

        try:
            augmented_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                    HumanMessage(content=augmented_query_prompt),
                ]
            )
            augmented_query = augmented_response.content
            logger.info(
                f"[QueryPlanner] Successfully augmented query: '{augmented_query[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[QueryPlanner] Failed to augment query, using original question: {e}"
            )
            augmented_query = question

        return {"augmented_query": augmented_query}

    @traceable(run_type="llm", name="query_categorize")
    async def categorize_query(
        self,
        state: ConversationState,
        conversation_data: Dict[str, Any],
        context_builder: ContextBuilder,
    ) -> Dict[str, str]:
        """
        Categorize query into marketing categories.

        Uses Claude with conversation history for context to classify queries into:
        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - Help Desk
        - General

        Args:
            state: Current conversation state
            conversation_data: Conversation history
            context_builder: ContextBuilder instance for formatting history

        Returns:
            Dictionary with query_category
        """
        logger.info(f"[QueryPlanner] Categorizing query: {state.question[:50]}...")

        history = conversation_data.get("history", [])
        formatted_history = context_builder.format_conversation_history(history)

        # Build categorization prompt
        category_prompt = f"""
            You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

            - Creative Brief
            - Marketing Plan
            - Help Desk
            - Brand Positioning Statement
            - Creative Copywriter
            - General

            Use both the current question and the historical conversation context to make an informed decision. 
            Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

            To help you make an accurate decision, consider these cues for each category:

            - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
            - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
            - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
            - **Creative Copywriter**: Use this category when users ask meta-questions about your fundamental functions, purpose, or identity. This applies to inquiries such as "What can you do?", "What are your capabilities?", or "How do you work?". This category is distinct from standard user commands or requests for technical support.
            - **Help Desk**: Use this category when users ask "What can you do?", "What are your capabilities?", or similar questions about your functions.
            - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

            If the question or context is not clearly related to any of the above categories, always return "General".

            ----------------------------------------
            User's Question:
            {state.question}
            ----------------------------------------
            Conversation History:
            {formatted_history}
            ----------------------------------------

            Reply with **only** the exact category name — no additional text, explanations, or formatting.
            """

        logger.info("[QueryPlanner] Sending categorization request to LLM")

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=category_prompt),
                    HumanMessage(content=state.question),
                ]
            )
            query_category = response.content.strip()
            logger.info(f"[QueryPlanner] Complete: Category = '{query_category}'")
        except Exception as e:
            logger.error(f"[QueryPlanner] Error categorizing query: {e}")
            query_category = "General"

        return {"query_category": query_category}
