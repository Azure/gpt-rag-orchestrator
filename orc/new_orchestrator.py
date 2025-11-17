import os
import logging
import base64
import uuid
import time
import json
import asyncio
import concurrent.futures
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from orc.graphs.main_2 import create_conversation_graph
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import List, Optional
from langchain_core.messages import (
    HumanMessage,
    SystemMessage as LangchainSystemMessage,
)

from shared.util import get_setting, get_verbosity_instruction
from shared.progress_streamer import ProgressStreamer, ProgressSteps, STEP_MESSAGES

from shared.prompts import (
    MARKETING_ANSWER_PROMPT,
    CREATIVE_BRIEF_PROMPT,
    MARKETING_PLAN_PROMPT,
    BRAND_POSITION_STATEMENT_PROMPT,
    CREATIVE_COPYWRITER_PROMPT,
)
from shared.util import get_organization
from dotenv import load_dotenv

from orc.graphs.models import ConversationState
from orc.graphs.utils import clean_chat_history_for_llm
from orc.graphs.constants import TOOL_DISPLAY_NAME, ENV_O1_ENDPOINT, ENV_O1_KEY
from orc.orchestrator.formatters import (
    sanitize_storage_urls,
    format_context,
    extract_blob_urls,
)

load_dotenv()
# Configure logging
logging.getLogger("azure").setLevel(logging.INFO)
logging.getLogger("azure.cosmos").setLevel(logging.INFO)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

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
        return sanitize_storage_urls(text, self.storage_url)

    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Create a fresh memory saver for this conversation.

        In modern LangGraph, we use a fresh checkpointer for each conversation
        and let the graph handle memory persistence automatically.
        The conversation history is maintained through the database instead.
        """
        return MemorySaver()

    def _clean_chat_history(self, chat_history: List[dict]) -> str:
        """Use shared formatter for chat history."""
        return clean_chat_history_for_llm(chat_history)

    def _format_context(self, context_docs: List, display_source: bool = True) -> str:
        return format_context(context_docs, display_source=display_source)

    def _get_tool_name(self, state: ConversationState) -> str:
        """Get the name of the tool used from the state."""
        if state.mcp_tool_used:
            tool_name = state.mcp_tool_used[-1]["name"]
            return TOOL_DISPLAY_NAME.get(tool_name, tool_name) or ""
        return ""

    @traceable(run_type="llm")
    def generate_response_with_progress(
        self,
        conversation_id: str,
        question: str,
        user_info: dict,
        user_settings: Optional[dict] = None,
        user_timezone: str | None = None,
        blob_names: Optional[List[str]] = None,
    ):
        """Generate response with real-time progress streaming."""
        start_time = time.time()
        conversation_id = conversation_id or str(uuid.uuid4())
        blob_names = blob_names or []

        logging.info(
            f"[orchestrator-generate_response] Starting streaming response for: {question[:50]}..."
        )

        progress_data = {
            "type": "progress",
            "step": ProgressSteps.INITIALIZATION,
            "message": STEP_MESSAGES[ProgressSteps.INITIALIZATION],
            "progress": 5,
            "timestamp": time.time(),
        }
        yield f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"

        try:
            # Load conversation state
            logging.info(
                f"[orchestrator] Loading conversation data for ID: {conversation_id}"
            )
            conversation_data = get_conversation_data(
                conversation_id, user_info["id"], user_timezone=user_timezone
            )
            memory = self._load_memory(conversation_data.get("memory_data", ""))

            # Progress for graph setup
            progress_data = {
                "type": "progress",
                "step": "graph_setup",
                "message": "Setting up conversation...",
                "progress": 10,
                "timestamp": time.time(),
            }
            yield f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n"

            config = {"configurable": {"thread_id": conversation_id}}

            # Create a list to collect progress updates from the async agent
            progress_updates = []

            def stream_progress(data):
                progress_updates.append(data)

            agent = create_conversation_graph(
                memory=memory,
                organization_id=self.organization_id,
                conversation_id=conversation_id,
                user_id=user_info["id"],
                progress_streamer=ProgressStreamer(stream_progress),
            )

            def run_async_agent():
                return asyncio.run(
                    agent.ainvoke(
                        {"question": question, "blob_names": blob_names}, config
                    )
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_agent)

                while not future.done():
                    while progress_updates:
                        progress_update = progress_updates.pop(0)
                        yield progress_update
                    time.sleep(0.1)  # Small delay to prevent busy waiting

                while progress_updates:
                    progress_update = progress_updates.pop(0)
                    yield progress_update

                response = future.result()

            # Create conversation state from response
            state = ConversationState(
                question=question,
                messages=response["messages"],
                context_docs=response["context_docs"],
                requires_retrieval=response.get("requires_retrieval", False),
                rewritten_query=response["rewritten_query"],
                query_category=response["query_category"],
                augmented_query=response.get("augmented_query", ""),
                mcp_tool_used=response.get("mcp_tool_used", []),
                tool_results=response.get("tool_results", []),
                code_thread_id=response.get("code_thread_id", None),
                last_mcp_tool_used=response.get("last_mcp_tool_used", ""),
                blob_names=blob_names,
                uploaded_file_refs=response.get("uploaded_file_refs", []),
            )

            # Get blob URLs for images for data analyst tool
            blob_urls = extract_blob_urls(state.tool_results)

            # Emit thoughts data first
            tool_name = self._get_tool_name(state)
            thoughts_data = {
                "conversation_id": conversation_id,
                "thoughts": [
                    f"""Model Used: {user_settings.get('model')} / Tool Selected: {state.query_category} / Original Query : {state.question} / Rewritten Query: {state.rewritten_query} / Required Retrieval: {state.requires_retrieval} / Number of documents retrieved: {len(state.context_docs) if state.context_docs else 0} / MCP Tool Used: {tool_name} / Context Retrieved using the rewritten query: / {self._format_context(state.context_docs, display_source=True)}"""
                ],
                "images_blob_urls": blob_urls,
            }
            yield f"__METADATA__{json.dumps(thoughts_data)}__METADATA__\n"

            # Progress for response generation start
            response_start_data = {
                "type": "progress",
                "step": ProgressSteps.RESPONSE_GENERATION,
                "message": STEP_MESSAGES[ProgressSteps.RESPONSE_GENERATION],
                "progress": 60,
                "timestamp": time.time(),
            }
            yield f"__PROGRESS__{json.dumps(response_start_data)}__PROGRESS__\n"

            # Now start response generation
            yield from self._generate_final_response(
                state,
                conversation_data,
                user_info,
                user_settings,
                start_time,
                conversation_id,
            )

        except Exception as e:
            logging.error(f"[orchestrator] Error in response generation: {str(e)}")
            store_agent_error(user_info["id"], str(e), question)
            error_data = {
                "type": "error",
                "message": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "timestamp": time.time(),
            }
            yield f"__PROGRESS__{json.dumps(error_data)}__PROGRESS__\n"

    def _generate_final_response(
        self,
        state,
        conversation_data,
        user_info,
        user_settings,
        start_time,
        conversation_id,
    ):
        """Generate the final streaming response using the processed state."""

        context = ""
        if state.context_docs:
            context = self._format_context(state.context_docs)

        logging.info("[orchestrator] Retrieving conversation history")
        history = conversation_data.get("history", [])

        # Retrieve organization data once for efficiency
        logging.info("[orchestrator] Retrieving organization data")
        organization_data = get_organization(self.organization_id)

        system_prompt = MARKETING_ANSWER_PROMPT

        verbosity_instruction = get_verbosity_instruction(user_settings)

        # Add context to the system prompt
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
        - Never create a separate "Sources"/"References"/"Data Sources" section at the end in your answer. The citation system will break if you do this.
        {organization_data.get('additionalInstructions','')}

        {verbosity_instruction}

        <----------- END OF Important User Instructions ------------>

        <----------- SYSTEM PROMPT FOR TOOL CALLING ------------>
        """

        # Add additional context to the system prompt
        system_prompt += additional_context

        if state.query_category in CATEGORY_PROMPT:
            system_prompt += CATEGORY_PROMPT[state.query_category]

        # Exclude augmented query for balanced or brief verbosity settings
        detail_level = user_settings.get("detail_level", "balanced")
        augmented_query_text = (
            "" if detail_level in ["balanced", "brief"] else state.augmented_query
        )

        prompt = f"""
        
        You're provided user's question and the augmented version of the question to help you understand the user's intent better. 
        If the original question and augmented question are conflicting, always use the original question. 
        Provide a detailed answer that is highly relevant to the user's question and provided context.
        
        <----------- USER QUESTION & AUGMENTED VERSION ------------>

        ORIGINAL QUESTION: {state.question}


        AUGMENTED VERSION OF THE QUESTION: {augmented_query_text}
        <----------- END OF USER QUESTION & AUGMENTED VERSION ------------>

        <----------- USER INSTRUCTIONS ------------>
        Do not mention the user's instruction in the response:
        - Ensure that you follow the citation format for text and image. Again, if an image link is present in the context, you must include it in the response. DO not forget the `!` for image citation.
        - If there are absolutely no references or links in the context at all, you can omit the citation.
        - If the context contains citations for excel or csv files, you must refer to the excel/csv citation format as instructed above when generating the response. The format for excel/csv citation is: [[number]](file_name.extension). Please refer to examples of csv/excel citation format above for more examples
        - IMPORTANT: Never create a separate "Sources"/"References"/"Data Sources" section at the end in your answer. The citation system will break if you do this.
        - Never cite the visual more than once. If the visual is already cited, do not cite it again.
        """

        logging.info("[orchestrator] Starting final response generation")
        complete_response = ""

        try:
            model = user_settings.get("model", "gpt-4.1")
            if model == "gpt-4.1":
                logging.info("[orchestrator] Streaming response from Azure Chat OpenAI")
                response_llm = AzureChatOpenAI(
                    temperature=user_settings.get("temperature", 0.3),
                    openai_api_version="2025-01-01-preview",
                    azure_deployment=model,
                    streaming=True,
                    timeout=30,
                    max_retries=3,
                    azure_endpoint=os.getenv(ENV_O1_ENDPOINT),
                    api_key=os.getenv(ENV_O1_KEY),
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

            elif model == "Claude-4.5-Sonnet":
                logging.info("[orchestrator] Streaming response from Claude 4 Sonnet")
                response_llm = ChatAnthropic(
                    model="claude-sonnet-4-5-20250929",
                    temperature=user_settings.get("temperature", 0.3),
                    streaming=True,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    max_tokens=64000,
                    max_retries=3,
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

        except Exception as e:
            logging.error(f"[orchestrator] Error generating response: {str(e)}")
            store_agent_error(user_info["id"], str(e), state.question)
            error_message = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            complete_response = error_message
            yield error_message

        # Save conversation history
        answer = self._sanitize_response(complete_response)
        tool_name = self._get_tool_name(state)

        # Get blob URLs
        # blob_urls = extract_blob_urls(state.tool_results)

        # Update conversation history
        history.extend(
            [
                {"role": "user", "content": state.question},
                {
                    "role": "assistant",
                    "content": answer,
                    "thoughts": [
                        f"""Model Used: {user_settings.get('model', 'gpt-4.1')} / Tool Selected: {state.query_category} / Original Query : {state.question} / Rewritten Query: {state.rewritten_query} / Required Retrieval: {state.requires_retrieval} / Number of documents retrieved: {len(state.context_docs) if state.context_docs else 0} / MCP Tool Used: {tool_name} / Context Retrieved using the rewritten query: / {self._format_context(state.context_docs, display_source=True)}"""
                    ],
                    "code_thread_id": state.code_thread_id,
                    "last_mcp_tool_used": state.last_mcp_tool_used,
                    "uploaded_file_refs": state.uploaded_file_refs,
                },
            ]
        )

        # Save updated state
        conversation_data.update(
            {
                "history": history,
                "memory_data": self._serialize_memory(
                    MemorySaver(), {"configurable": {"thread_id": conversation_id}}
                ),
                "interaction": {
                    "user_id": user_info["id"],
                    "user_name": user_info["name"],
                    "response_time": round(time.time() - start_time, 2),
                    "organization_id": self.organization_id,
                },
            }
        )

        update_conversation_data(conversation_id, user_info["id"], conversation_data)


def get_settings(client_principal):
    # use cosmos to get settings from the logged user
    data = get_setting(client_principal)
    temperature = None if "temperature" not in data else data["temperature"]
    model = None if "model" not in data else data["model"]
    detail_level = None if "detail_level" not in data else data["detail_level"]
    settings = {
        "temperature": temperature,
        "model": model,
        "detail_level": detail_level,
    }
    logging.info(f"[orchestrator] settings: {settings}")
    return settings
