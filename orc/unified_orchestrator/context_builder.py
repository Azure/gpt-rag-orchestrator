"""
ContextBuilder Component

This module provides the ContextBuilder class for formatting organization data
and conversation history into context prompts for LLM consumption.

Responsibilities:
- Format organization context (segments, brand, industry)
- Clean and format conversation history
- Extract context from tool results
- Extract metadata (blob URLs, thread IDs)
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds context prompts from organization data and conversation history.

    Responsibilities:
    - Format organization context (segments, brand, industry)
    - Clean and format conversation history
    - Extract context from tool results
    - Extract metadata (blob URLs, thread IDs)
    """

    def __init__(self, organization_data: Dict[str, Any]):
        """
        Initialize ContextBuilder.

        Args:
            organization_data: Organization information from Cosmos DB
        """
        self.organization_data = organization_data or {}
        logger.info("[ContextBuilder] Initialized")

    def build_organization_context(
        self, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build organization context string for prompts.

        Formats segment aliases, brand information, industry information,
        and optionally conversation history from organization data.

        Args:
            history: Optional list of conversation history messages

        Returns:
            Formatted organization context string with sections for:
            - Historical conversation context (if provided)
            - Segment alias mappings (segmentSynonyms)
            - Brand information (brandInformation)
            - Industry information (industryInformation)
        """
        logger.debug("[ContextBuilder] Building organization context")

        segment_synonyms = self.organization_data.get("segmentSynonyms", "")
        brand_info = self.organization_data.get("brandInformation", "")
        industry_info = self.organization_data.get("industryInformation", "")

        logger.debug(
            f"[ContextBuilder] Org data present - "
            f"segments: {bool(segment_synonyms)}, "
            f"brand: {bool(brand_info)}, "
            f"industry: {bool(industry_info)}, "
            f"history: {len(history) if history else 0} messages"
        )

        context = ""

        # Add conversation history if provided
        if history:
            formatted_history = self.format_conversation_history(history)
            if formatted_history:
                context += f"""
                <----------- HISTORICAL CONVERSATION CONTEXT ------------>
                {formatted_history}
                <----------- END OF HISTORICAL CONVERSATION CONTEXT ------------>
                """
                logger.debug(
                    "[ContextBuilder] Added conversation history to organization context"
                )

        context += f"""
            <----------- PROVIDED SEGMENT ALIAS (VERY CRITICAL, MUST FOLLOW) ------------>
            Alias to segment mappings typically look like this (Official Name -> Alias):
            A -> B

            This mapping is mostly used in consumer segmentation context.

            Critical Rule – Contextual Consistency with Alias Mapping:
            • Always check whether the segment reference in the historical conversation is an alias (B). For example, historical conversation may mention "B" segment, but whenever you read the context in order to rewrite the query, you must map it to the official segment name "A" using the alias mapping table.
            • ALWAYS use the official name (A) in the rewritten query.
            • DO NOT use the alias (B) in the rewritten query.

            Here is the actual alias to segment mappings:

            **Official Segment Name Mappings (Official Name -> Alias):**
            {segment_synonyms}

            For example, if the historical conversation mentions "B", and the original question also mentions "B", you must rewrite the question to use "A" instead of "B".

            Look, if a mapping in the instruction is like this:
            students -> young kids

            Though the historical conversation and the original question may mention "students", you must rewrite the question to use "young kids" instead of "students".
            <----------- END OF PROVIDED SEGMENT ALIAS ------------>

            <----------- PROVIDED Brand Information ------------>
            This is the Brand information for the organization that the user belongs to.
            When relevant, incorporate Brand information to tailor responses, ensuring that answers are highly relevant to the user's company, goals, and operational environment.        
            Here is the Brand information:
            {brand_info}
            <----------- END OF PROVIDED Brand Information ------------>

            <----------- PROVIDED INDUSTRY DEFINITION ------------>
            This is the industry definition for the organization. This helps to understand the context of the organization and tailor responses accordingly
            Here is the industry definition:
            {industry_info}
            <----------- END OF PROVIDED INDUSTRY DEFINITION ------------>
            """
        return context

    def format_conversation_history(
        self, history: List[Dict[str, Any]], max_messages: int = 6
    ) -> str:
        """
        Format conversation history for LLM consumption.

        Cleans and formats history by:
        - Truncating to most recent messages
        - Removing markdown images
        - Formatting as "Human:" and "AI Message:" exchanges

        Args:
            history: List of conversation messages (dicts with 'role' and 'content')
            max_messages: Maximum number of messages to include (default: 6)

        Returns:
            Formatted conversation history string
        """
        logger.debug(
            f"[ContextBuilder] Formatting conversation history ({len(history)} messages)"
        )

        if not history:
            logger.debug("[ContextBuilder] No history to format")
            return ""

        # Truncate to most recent messages
        if len(history) > max_messages:
            truncated_history = history[-max_messages:]
            logger.debug(f"[ContextBuilder] Truncated to last {max_messages} messages")
        else:
            truncated_history = history

        formatted_messages = []
        for message in truncated_history:
            if not isinstance(message, dict):
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if not content:
                continue

            # Determine display role
            if role == "user":
                display_role = "Human"
            elif role == "assistant":
                display_role = "AI Message"
            else:
                continue

            # Clean markdown images from content
            cleaned_content = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", "", content)

            formatted_messages.append(f"{display_role}: {cleaned_content}")

        result = "\n\n".join(formatted_messages)
        logger.debug(f"[ContextBuilder] Formatted {len(formatted_messages)} messages")

        return result

    def extract_context_from_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[List[Any], List[str], List[Dict[str, str]]]:
        """
        Extract context from LangChain messages (used with bind_tools approach).

        When using bind_tools, tool calls and results are embedded in message objects.
        This method extracts the actual tool results from AIMessage tool_calls and
        ToolMessage objects.

        Args:
            messages: List of LangChain messages from tool execution

        Returns:
            Tuple of (context_docs, blob_urls, uploaded_file_refs)
        """
        logger.debug(
            f"[ContextBuilder] Extracting context from {len(messages)} messages"
        )

        context_docs = []
        blob_urls = []
        uploaded_file_refs = []

        # Find tool-related messages
        for msg in messages:
            # Check for ToolMessage (contains tool results)
            if hasattr(msg, "content") and hasattr(msg, "name"):
                tool_name = getattr(msg, "name", "")
                content = msg.content

                # Parse JSON content if it's a string
                result = content
                if isinstance(content, str):
                    try:
                        result = json.loads(content)
                    except Exception:
                        pass

                logger.debug(
                    f"[ContextBuilder] Processing tool message from {tool_name}"
                )

                # Extract based on tool type
                if tool_name == "agentic_search" and isinstance(result, dict):
                    search_results = result.get("results", result)

                    filtered_docs = []

                    if isinstance(search_results, dict):
                        for subquery_key, subquery_data in search_results.items():
                            if (
                                isinstance(subquery_data, dict)
                                and "documents" in subquery_data
                            ):
                                documents = subquery_data.get("documents", [])
                                if isinstance(documents, list):
                                    for doc in documents:
                                        if isinstance(doc, dict):
                                            filtered_docs.append(
                                                {
                                                    "content": doc.get("content"),
                                                    "source": doc.get("source"),
                                                }
                                            )

                    if filtered_docs:
                        context_docs.append(filtered_docs)
                        logger.debug(
                            f"[ContextBuilder] Added {len(filtered_docs)} filtered documents from agentic_search"
                        )
                    else:
                        # If not a list, append as-is (fallback)
                        context_docs.append(search_results)
                        logger.debug(
                            "[ContextBuilder] Added agentic_search results (non-list format)"
                        )

                elif tool_name == "data_analyst" and isinstance(result, dict):
                    last_message = result.get("last_agent_message", result)
                    context_docs.append(last_message)

                    # Extract blob URLs
                    result_blob_urls = result.get("blob_urls", [])
                    if isinstance(result_blob_urls, list) and result_blob_urls:
                        for blob_item in result_blob_urls:
                            if isinstance(blob_item, dict):
                                blob_path = blob_item.get("blob_path")
                                if blob_path:
                                    blob_urls.append(blob_path)
                                    context_docs.append(
                                        f"""
                                        Below is the graph/visualization link - This link is used to render a UI image. Do not change any character in this link, or the image rendering will break.
                                        In this scenario, if the link is a power point presentation (.pptx), cite it the same way as an image citation. This is a special case.
                                        <link>
                                        {blob_path}
                                        </link>
                                        """
                                    )
                                    logger.debug(
                                        f"[ContextBuilder] Added blob URL from message: {blob_path}"
                                    )

                elif tool_name == "web_fetch" and isinstance(result, dict):
                    web_content = result.get("content")
                    if web_content:
                        context_docs.append(web_content)
                    else:
                        context_docs.append(result)
                    logger.debug(
                        "[ContextBuilder] Added web_fetch content from message"
                    )

                elif tool_name == "document_chat" and isinstance(result, dict):
                    answer = result.get("answer", result)
                    context_docs.append(answer)

                    # Extract file references - caching uploaded files (openai expired in 1 hour)
                    files = result.get("files", [])
                    if files and isinstance(files, list):
                        uploaded_file_refs = files
                        logger.debug(
                            f"[ContextBuilder] Extracted {len(files)} file references from message"
                        )

        logger.info(
            f"[ContextBuilder] Extracted from messages: {len(context_docs)} context docs, "
            f"{len(blob_urls)} blob URLs, {len(uploaded_file_refs)} file refs"
        )

        return context_docs, blob_urls, uploaded_file_refs
