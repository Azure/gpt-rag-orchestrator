import os
import logging
import base64
import uuid
import time
import re
import json
import asyncio
import concurrent.futures
from langchain_community.callbacks import get_openai_callback
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
# from orc.graphs.main import create_conversation_graph
from orc.graphs.main_2 import create_conversation_graph
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from dataclasses import dataclass, field
from typing import List, Dict, Any
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage as LangchainSystemMessage,
)
from urllib.parse import unquote

from shared.util import get_setting

from langchain.schema import Document
from shared.prompts import (
    MARKETING_ANSWER_PROMPT,
    CREATIVE_BRIEF_PROMPT,
    MARKETING_PLAN_PROMPT,
    BRAND_POSITION_STATEMENT_PROMPT,
    CREATIVE_COPYWRITER_PROMPT,
)
from shared.util import get_organization
from dotenv import load_dotenv

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()
# Configure logging
logging.getLogger("azure").setLevel(logging.INFO)
logging.getLogger("azure.cosmos").setLevel(logging.INFO)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())


@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_retrieval: Flag indicating if retrieval is needed
        rewritten_query: Rewritten query for better search
        query_category: Category of the query
        augmented_query: Augmented version of the query
        mcp_tool_used: List of MCP tools that were used
        tool_results: Results from tool execution
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_retrieval: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  # rewritten query for better search
    query_category: str = field(default_factory=str)
    augmented_query: str = field(default_factory=str)
    mcp_tool_used: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)


# Prompt for Tool Calling
CATEGORY_PROMPT = {
    "Creative Brief": CREATIVE_BRIEF_PROMPT,
    "Marketing Plan": MARKETING_PLAN_PROMPT,
    "Brand Positioning Statement": BRAND_POSITION_STATEMENT_PROMPT,
    "Creative Copywriter": CREATIVE_COPYWRITER_PROMPT,
    "General": "",
}


class ConversationOrchestrator:
    """Manages conversation flow and state between user and AI agent."""

    def __init__(self, organization_id: str = None):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self.organization_id = organization_id

    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        tuple_data = memory.get_tuple(config)
        if tuple_data:
            serialized = memory.serde.dumps(tuple_data)
            return base64.b64encode(serialized).decode("utf-8")
        return ""

    def _sanitize_response(self, text: str) -> str:
        """Remove sensitive storage URLs from response text."""
        if self.storage_url in text:
            regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({self.storage_url})?(\/?documents\/?)?"
            return re.sub(regex, "", text)
        return text

    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Create a fresh memory saver for this conversation.
        
        In modern LangGraph, we use a fresh checkpointer for each conversation
        and let the graph handle memory persistence automatically.
        The conversation history is maintained through the database instead.
        """
        return MemorySaver()

    def _clean_chat_history(self, chat_history: List[dict]) -> str:
        """
        Clean the chat history and format it as a string for LLM consumption.

        Args:
            chat_history (list): List of chat message dictionaries

        Returns:
            str: Formatted chat history string in the format:
                 Human: {message}
                 AI: {message}
        """
        formatted_history = []

        for message in chat_history:
            if not message.get("content"):
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role and content:
                display_role = "Human" if role == "user" else "AI Message"
                formatted_history.append(f"{display_role}: {content}")

        return "\n\n".join(formatted_history)

    def _format_source_path(self, source_path: str) -> str:
        """
        Formats a source path by extracting relevant parts and URL decoding.

        Args:
            source_path: The raw source path from document metadata

        Returns:
            str: A clean, readable file path
        """
        if not source_path:
            return ""

        try:
            # Split the path and take elements from index 3 onwards
            path_parts = source_path.split("/")[3:]

            # URL decode each part to convert %20 to spaces, etc.
            decoded_parts = [unquote(part) for part in path_parts]

            # Join with forward slashes to create a clean path
            clean_path = "/".join(decoded_parts)

            return clean_path

        except (IndexError, Exception) as e:
            # Fallback to original path if processing fails
            logging.warning(f"Failed to format source path '{source_path}': {e}")
            return source_path

    def _format_context(
        self, context_docs: List, display_source: bool = True
    ) -> str:
        """Formats retrieved documents into a string for LLM consumption."""
        if not context_docs:
            return ""
        
        formatted_docs = []
        for doc in context_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                source = doc.metadata.get("source", "") if hasattr(doc, 'metadata') else ""
            elif isinstance(doc, dict):
                content = doc.get('Content', doc.get('content', doc.get('text', str(doc))))
                source = doc.get('Source', doc.get('source', doc.get('Title', '')))
            else:
                # Fallback to string representation
                content = str(doc)
                source = ""
            
            if display_source and source:
                formatted_source = self._format_source_path(source) if 'documents/' in source else source
                formatted_docs.append(f"\nContent: \n\n{content}\n\nSource: {formatted_source}")
            else:
                formatted_docs.append(f"\nContent: \n\n{content}")
        
        return "\n\n==============================================\n\n".join(formatted_docs)

    def process_conversation(
        self, conversation_id: str, question: str, user_info: dict
    ) -> dict:
        """
        Process a conversation turn with the AI agent.

        Args:
            conversation_id: Unique identifier for conversation
            question: User's input question
            user_info: Dictionary containing user metadata

        Returns:
            dict: Response containing conversation_id, answer and thoughts
        """
        start_time = time.time()
        logging.info(
            f"[orchestrator-process_conversation] Gathering resources for: {question}"
        )
        conversation_id = conversation_id or str(uuid.uuid4())

        try:
            # Load conversation state
            logging.info(
                f"[orchestrator-process_conversation] Loading conversation data for ID: {conversation_id}"
            )
            conversation_data = get_conversation_data(conversation_id)
            logging.info(
                "[orchestrator-process_conversation] Successfully loaded conversation data"
            )
            logging.info("[orchestrator-process_conversation] Loading memory")
            memory = self._load_memory(conversation_data.get("memory_data", ""))
            logging.info("[orchestrator-process_conversation] Memory loaded")
            # Process through agent

            # insert conversation to the memory object
            logging.info(
                "[orchestrator-process_conversation] Creating conversation graph"
            )
            agent = create_conversation_graph(
                memory=memory,
                organization_id=self.organization_id,
                conversation_id=conversation_id,
            )
            logging.info(
                "[orchestrator-process_conversation] Successfully created agent"
            )
            config = {"configurable": {"thread_id": conversation_id}}

            with get_openai_callback() as cb:
                # Get agent response
                logging.info(
                    "[orchestrator-process_conversation] Invoking agent async"
                )

                def run_async_agent():
                    return asyncio.run(agent.ainvoke({"question": question}, config))

                # Run the async function in a separate thread to avoid event loop conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_agent)
                    response = future.result()

                return {
                    "conversation_id": conversation_id,
                    "state": ConversationState(
                        question=question,
                        messages=response["messages"],
                        context_docs=response["context_docs"],
                        requires_retrieval=response.get("requires_retrieval", False),
                        rewritten_query=response["rewritten_query"],
                        query_category=response["query_category"],
                        augmented_query=response.get("augmented_query", ""),
                        mcp_tool_used=response.get("mcp_tool_used", []),
                        tool_results=response.get("tool_results", []),
                    ),
                    "conversation_data": conversation_data,
                    "memory_data": self._serialize_memory(memory, config),
                    "start_time": start_time,
                    "consumed_tokens": cb,
                }

        except Exception as e:
            logging.error(
                f"[orchestrator-process_conversation] Error retrieving resources: {str(e)}"
            )
            store_agent_error(user_info["id"], str(e), question)

    @traceable(run_type="llm")
    def generate_response(
        self,
        conversation_id: str,
        state: ConversationState,
        conversation_data: dict,
        user_info: dict,
        memory_data: str,
        start_time: float,
        user_settings: dict = None,
    ):
        """Generate final response using context and query."""
        logging.info(
            f"[orchestrator-generate_response] Generating response for: {state.question}"
        )
        blob_urls = []
        for tool_result in state.tool_results:
            results_json = json.loads(tool_result)
            blob_urls.extend(results_json.get("blob_urls", []))
        
        data = {
            "conversation_id": conversation_id,
            "thoughts": [
                f"""
                Model Used: {user_settings['model']} / Tool Selected: {state.query_category} / Original Query : {state.question} / Rewritten Query: {state.rewritten_query} / Required Retrieval: {state.requires_retrieval} / Number of documents retrieved: {len(state.context_docs) if state.context_docs else 0} / MCP Tools Used: {len(state.mcp_tool_used)} / Context Retrieved using the rewritten query: / {self._format_context(state.context_docs, display_source=True)}"""
            ],
            "images_blob_urls": blob_urls,
        }
        yield json.dumps(data)
        context = ""
        if state.context_docs:
            context = self._format_context(state.context_docs)

        logging.info(
            "[orchestrator-generate_response] Retrieving conversation history"
        )
        history = conversation_data.get("history", [])

        # Retrieve organization data once for efficiency
        logging.info("[orchestrator-generate_response] Retrieving organization data")
        organization_data = get_organization(self.organization_id)

        system_prompt = MARKETING_ANSWER_PROMPT

        # add context to the system prompt

        additional_context = f"""

        Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)

        <----------- PROVIDED SEGMENT ALIAS (VERY CRITICAL, MUST FOLLOW) ------------>
        Here is the segment alias:
        {organization_data.get('segmentSynonyms','')}
        <----------- END OF PROVIDED SEGMENT ALIAS ------------>
        
        <----------- PROVIDED CONTEXT ------------>
        {context}
        <----------- END OF PROVIDED CONTEXT ------------>

        Chat History (IMPORTANT, USED AS A CONTENXT FOR ANSWER WHENEVER APPLICABLE):

        <----------- PROVIDED CHAT HISTORY ------------>
        {self._clean_chat_history(history)}
        <----------- END OF PROVIDED CHAT HISTORY ------------>

        Query Category:

        <----------- PROVIDED QUERY CATEGORY ------------>
        {state.query_category}
        <----------- END OF PROVIDED QUERY CATEGORY ------------>

        Brand Information:

        <----------- PROVIDED Brand Information ------------>
        This is the Brand information for the organization that the user belongs to.
        Whenever possible, incorporate Brand information to tailor responses, ensuring that answers are highly relevant to the user's company, goals, and operational environment.        
        Here is the Brand information:

        {organization_data.get('brandInformation','')}
        <----------- END OF PROVIDED Brand Information ------------>

        <----------- PROVIDED INDUSTRY DEFINITION ------------>
        This is the industry definition for the organization. This helps to understand the context of the organization and tailor responses accordingly
        Here is the industry definition:

        {organization_data.get('industryInformation','')}

        <----------- END OF PROVIDED INDUSTRY DEFINITION ------------>

        System prompt for tool calling (if applicable):

        NOTE: When using the tool calling prompt, you should try to incorporate all the provided information from the Chat History and Brand information to tailor the response.
        You should also ask you to provide more information if needed.

        <----------- Important User Instructions ------------>
        This is the important user instructions for the response.
        You should follow these instructions strictly as it sets the tone of the response user is expecting.
        Here are the instructions:

        {organization_data.get('additionalInstructions','')}
        <----------- END OF Important User Instructions ------------>

        <----------- SYSTEM PROMPT FOR TOOL CALLING ------------>
        """

        # add additional context to the system prompt
        system_prompt += additional_context

        if state.query_category in CATEGORY_PROMPT:
            system_prompt += CATEGORY_PROMPT[state.query_category]

        prompt = f"""
        
        You're provided user's question and the augmented version of the question to help you understand the user's intent better. 
        If the original question and augmented question are conflicting, always use the original question. 
        Provide a detailed answer that is highly relevant to the user's question and provided context.
        
        <----------- USER QUESTION & AUGMENTED VERSION ------------>

        ORIGINAL QUESTION: {state.question}

        
        AUGMENTED VERSION OF THE QUESTION: {state.augmented_query}
        <----------- END OF USER QUESTION & AUGMENTED VERSION ------------>
        """

        logging.info(f"[orchestrator-generate_response] Prompt: {prompt}")
        # Generate response and update message history
        complete_response = ""

        try:
            if user_settings["model"] == "gpt-4.1":
                logging.info(
                    "[orchestrator-generate_response] Streaming response from Azure Chat OpenAI"
                )
                response_llm = AzureChatOpenAI(
                    temperature=user_settings["temperature"],
                    openai_api_version="2025-01-01-preview",
                    azure_deployment=user_settings["model"],
                    streaming=True,
                    timeout=30,
                    max_retries=3,
                    azure_endpoint=os.getenv("O1_ENDPOINT"),
                    api_key=os.getenv("O1_KEY"),
                )
                tokens = response_llm.stream(
                    [
                        LangchainSystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                )
                while True:
                    try:
                        token = next(tokens)
                        if token:
                            chunk = token.content
                            complete_response += chunk
                            yield chunk
                    except StopIteration:
                        break
            elif user_settings["model"] == "Claude-4-Sonnet":
                logging.info(
                    "[orchestrator-generate_response] Streaming response from Claude 4 Sonnet"
                )
                response_llm = ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    temperature=0,
                    streaming=True,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    max_tokens=5000,
                )
                tokens = response_llm.stream(
                    [
                        LangchainSystemMessage(content=system_prompt),
                        HumanMessage(content=prompt),
                    ]
                )
                while True:
                    try:
                        token = next(tokens)
                        if token:
                            chunk = token.content
                            complete_response += chunk
                            yield chunk
                    except StopIteration:
                        break
            elif user_settings["model"] == "DeepSeek-V3-0324":
                logging.info(
                    "[orchestrator-generate_response] Streaming response from DeepSeek V3"
                )
                endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
                key = os.getenv("AZURE_INFERENCE_SDK_KEY")
                client = ChatCompletionsClient(
                    endpoint=endpoint, credential=AzureKeyCredential(key)
                )

                response = client.complete(
                    messages=[
                        SystemMessage(content=system_prompt),
                        UserMessage(content=prompt),
                    ],
                    model=user_settings["model"],
                    max_tokens=10000,
                    temperature=user_settings["temperature"],
                    stream=True,
                )

                for update in response:
                    if update.choices and update.choices[0].delta:
                        chunk = update.choices[0].delta.content or ""
                        complete_response += chunk
                        yield chunk
        except Exception as e:
            logging.error(
                f"[orchestrator-generate_response] Error generating response: {str(e)}"
            )
            store_agent_error(user_info["id"], str(e), state.question)
            error_message = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            complete_response = error_message
            yield error_message

        logging.info(
            f"[orchestrator-generate_response] Response generated: {complete_response[:100] + '...' if len(complete_response) > 100 else complete_response}"
        )

        answer = self._sanitize_response(complete_response)

        # Update conversation history
        history.extend(
            [
                {"role": "user", "content": state.question},
                {
                    "role": "assistant",
                    "content": answer,
                    "thoughts": [
                        f"""Model Used: {user_settings['model']} / Tool Selected: {state.query_category} / Original Query : {state.question} / Rewritten Query: {state.rewritten_query} / Required Retrieval: {state.requires_retrieval} / Number of documents retrieved: {len(state.context_docs) if state.context_docs else 0} / MCP Tools Used: {len(state.mcp_tool_used)} / Context Retrieved using the rewritten query: / {self._format_context(state.context_docs, display_source=True)}"""
                    ],
                    "images_blob_urls": blob_urls,
                },
            ]
        )
        # Save updated state
        conversation_data.update(
            {
                "history": history,
                "memory_data": memory_data,
                "interaction": {
                    "user_id": user_info["id"],
                    "user_name": user_info["name"],
                    "response_time": round(time.time() - start_time, 2),
                    "organization_id": self.organization_id,
                },
            }
        )

        update_conversation_data(conversation_id, conversation_data)
        # TODO: ENABLE CONSUME TOKENS FOR RESPONSE GENERATION
        # store_user_consumed_tokens(user_info["id"], cb)


def get_settings(client_principal):
    # use cosmos to get settings from the logged user
    data = get_setting(client_principal)
    temperature = None if "temperature" not in data else data["temperature"]
    model = None if "model" not in data else data["model"]
    settings = {
        "temperature": temperature,
        "model": model,
    }
    logging.info(f"[orchestrator] settings: {settings}")
    return settings


async def stream_run(
    conversation_id: str,
    ask: str,
    url: str,
    client_principal: dict,
    organization_id: str = None,
):
    orchestrator = ConversationOrchestrator(organization_id=organization_id)
    resources = await asyncio.to_thread(
        orchestrator.process_conversation, conversation_id, ask, client_principal
    )
    return await asyncio.to_thread(
        orchestrator.generate_response,
        resources["conversation_id"],
        resources["state"],
        resources["conversation_data"],
        client_principal,
        resources["memory_data"],
        resources["start_time"],
        organization_id,
    )
