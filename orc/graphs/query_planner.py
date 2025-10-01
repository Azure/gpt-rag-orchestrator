import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from orc.graphs.utils import (
    clean_chat_history_for_llm,
    extract_last_mcp_tool_from_history,
    extract_thread_id_from_history,
)
from shared.prompts import (
    MARKETING_ORC_PROMPT,
    QUERY_REWRITING_PROMPT,
    AUGMENTED_QUERY_PROMPT,
)


logger = logging.getLogger(__name__)


class QueryPlanner:
    """Handles query rewriting, augmentation, categorization, and routing."""

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self.llm = llm

    async def _llm_invoke(
        self, messages: List[SystemMessage | HumanMessage | AIMessage], **kwargs
    ):
        return await self.llm.ainvoke(messages, **kwargs)

    async def rewrite(
        self,
        state,
        conversation_data: Dict[str, Any],
        context_builder,
    ) -> dict:
        question = state.question
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Rewrite] Retrieved {len(history)} messages from conversation history"
        )

        system_prompt = f"{QUERY_REWRITING_PROMPT}\n\n{context_builder.build_organization_context_prompt(history)}"

        prompt = f"""Original Question: 
        <-------------------------------->
        ```
        {question}. 
        ```
        <-------------------------------->

        Please rewrite the question to be used for searching the database. Make sure to follow the alias mapping instructions at all cost.
        ALSO, THE HISTORICAL CONVERSATION CONTEXT IS VERY VERY IMPORTANT TO THE USER'S FOLLOW UP QUESTIONS, $10,000 WILL BE DEDUCTED FROM YOUR ACCOUNT IF YOU DO NOT USE THE HISTORICAL CONVERSATION CONTEXT.
        Please also consider the line of business/industry of my company when rewriting the query. Don't be too verbose. 

        if the question is a very casual/conversational one, do not rewrite, return it as it is
        """

        logger.info("[Query Rewrite] Sending async query rewrite request to LLM")
        rewritten_response = await self._llm_invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        )
        logger.info(
            f"[Query Rewrite] Successfully rewrote query: '{rewritten_response.content[:100]}...'"
        )

        # Augment the query with the historical conversation context
        augmented_query_prompt = f""" 
        Augment the query with the historical conversation context. If the query is a very casual/conversational one, do not augment, return it as it is.
        
        Here is the historical conversation context if available:
        <context>
        {clean_chat_history_for_llm(history)}
        </context>

        Here is the query to augment:
        <query>
        {question}
        </query>

        Return the augmented query in text format only, no additional text, explanations, or formatting.
        
        """
        logger.info(
            f"[Query Augment] Sending async augmented query request to LLM {augmented_query_prompt[:100]}..."
        )
        try:
            augmented_query = await self._llm_invoke(
                [
                    SystemMessage(content=AUGMENTED_QUERY_PROMPT),
                    HumanMessage(content=augmented_query_prompt),
                ]
            )
            logger.info(
                f"[Query Augment] Successfully augmented query: '{augmented_query.content[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[Query Augment] Failed to augment query, using original question: {e}"
            )
            augmented_query = question

        existing_thread_id = extract_thread_id_from_history(history)
        existing_last_mcp_tool = extract_last_mcp_tool_from_history(history)
        logger.info(
            f"[Query Rewrite] Initialized code thread_id from history: {existing_thread_id}"
        )
        logger.info(
            f"[Query Rewrite] Retrieved last_mcp_tool_used from history: {existing_last_mcp_tool}"
        )

        return {
            "rewritten_query": rewritten_response.content,
            "augmented_query": (
                augmented_query.content
                if hasattr(augmented_query, "content")
                else augmented_query
            ),
            "messages": state.messages + [HumanMessage(content=question)],
            "code_thread_id": existing_thread_id,
            "last_mcp_tool_used": existing_last_mcp_tool,
        }

    async def categorize(self, state, conversation_data: Dict[str, Any]) -> dict:
        history = conversation_data.get("history", [])
        logger.info(
            f"[Query Categorization] Retrieved {len(history)} conversation history messages for context"
        )

        category_prompt = f"""
        You are a senior marketing strategist. Your task is to classify the user's question into one of the following categories:

        - Creative Brief
        - Marketing Plan
        - Brand Positioning Statement
        - Creative Copywriter
        - General

        Use both the current question and the historical conversation context to make an informed decision. 
        Context is crucial, as users may refer to previous topics, provide follow-ups, or respond to earlier prompts. 

        To help you make an accurate decision, consider these cues for each category:

        - **Creative Brief**: Look for project kickoffs, campaign overviews, client objectives, audience targeting, timelines, deliverables, or communication goals.
        - **Marketing Plan**: Look for references to strategy, goals, budget, channels, timelines, performance metrics, or ROI.
        - **Brand Positioning Statement**: Watch for messages about defining brand essence, values, personality, competitive differentiation, or target audience perception.
        - **Creative Copywriter**: Use this category when the user asks for help creating or refining marketing text. This includes taglines, headlines, ad copy, email subject lines, social captions, website copy, or product descriptions. Trigger this if the user is brainstorming, writing, or editing text with a creative, promotional purpose.
        - **General**: If the input lacks context, doesn't relate to marketing deliverables, or is unclear or unrelated to the above.

        If the question or context is not clearly related to any of the above categories, always return "General".

        ----------------------------------------
        User's Question:
        {state.question}
        ----------------------------------------
        Conversation History:
        {clean_chat_history_for_llm(history)}
        ----------------------------------------

        Reply with **only** the exact category name — no additional text, explanations, or formatting.
        """

        logger.info("[Query Categorization] Sending async categorization request to LLM")
        response = await self._llm_invoke(
            [SystemMessage(content=category_prompt), HumanMessage(content=state.question)]
        )
        logger.info(f"[Query Categorization] Categorized query as: '{response.content}'")
        return {"query_category": response.content}

    async def route(self, state) -> dict:
        logger.info(
            f"[Query Routing] Determining routing decision for query: '{state.rewritten_query[:100]}...'"
        )
        human_prompt = f"""
        How should I categorize this question:
        - Original Question: {state.question}
        - Rewritten Question: {state.rewritten_query}
        
        Answer yes/no.
        """
        logger.info("[Query Routing] Sending routing decision request to LLM")
        response = await self._llm_invoke(
            [SystemMessage(content=MARKETING_ORC_PROMPT), HumanMessage(content=human_prompt)]
        )
        llm_suggests_retrieval = response.content.lower().startswith("y")
        logger.info(
            f"[Query Routing] LLM assessment - Not a casual/conversational question, proceed to retrieve documents: {llm_suggests_retrieval}"
        )
        return {"requires_retrieval": llm_suggests_retrieval, "query_category": "General"}

