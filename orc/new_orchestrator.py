import os
import logging
import base64
import uuid
import time
import re
import json
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.memory import MemorySaver
from orc.graphs.main import create_conversation_graph
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
    store_user_consumed_tokens,
)
from langchain_openai import AzureChatOpenAI
from dataclasses import dataclass, field
from typing import List
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage as LangchainSystemMessage,
    RemoveMessage,
)
from langchain.schema import Document
from shared.prompts import (
    MARKETING_ORC_PROMPT,
    MARKETING_ANSWER_PROMPT,
    QUERY_REWRITING_PROMPT,
)
from shared.tools import num_tokens_from_string, messages_to_string
from dotenv import load_dotenv

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()
# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG").upper())


@dataclass
class ConversationState:
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_web_search: Flag indicating if web search is needed
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(
        default_factory=list
    )  # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = field(default=False)
    rewritten_query: str = field(
        default_factory=str
    )  # rewritten query for better search
    chat_summary: str = field(default_factory=str)
    token_count: int = field(default_factory=int)


class ConversationOrchestrator:
    """Manages conversation flow and state between user and AI agent."""

    def __init__(self, organization_id: str = None):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self.organization_id = organization_id

    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        serialized = memory.serde.dumps(memory.get_tuple(config))
        return base64.b64encode(serialized).decode("utf-8")

    def _sanitize_response(self, text: str) -> str:
        """Remove sensitive storage URLs from response text."""
        if self.storage_url in text:
            regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({self.storage_url})?(\/?documents\/?)?"
            return re.sub(regex, "", text)
        return text

    def _summarize_chat(
        token_count: int,
        messages: List[AIMessage | HumanMessage],
        chat_summary: str,
        max_tokens: int,
        llm: AzureChatOpenAI,
    ) -> dict:
        """Summarize chat history if it exceeds token limit.


        Args:
            state: Current conversation state

        Returns:
            dict: Contains updated chat_summary and token_count
        """

        if token_count > max_tokens:
            try:
                if chat_summary:
                    messages = [
                        LangchainSystemMessage(
                            content="You are a helpful assistant that summarizes conversations."
                        ),
                        HumanMessage(
                            content=f"Previous summary:\n{chat_summary}\n\nNew messages to incorporate:\n{messages}\n\nPlease extend the summary. Return only the summary text."
                        ),
                    ]

                else:
                    messages = [
                        LangchainSystemMessage(
                            content="You are a helpful assistant that summarizes conversations."
                        ),
                        HumanMessage(
                            content=f"Summarize this conversation history. Return only the summary text:\n{messages}"
                        ),
                    ]

                new_summary = llm.invoke(messages)

                return new_summary.content

            except Exception as e:
                # Log the error but continue with empty summary
                print(f"Error summarizing chat: {str(e)}")
                return chat_summary or ""
        else:
            return chat_summary or ""

    def _load_memory(self, memory_data: str) -> MemorySaver:
        """Decode and load conversation memory from base64 string."""
        memory = MemorySaver()
        if memory_data != "":
            decoded_data = base64.b64decode(memory_data)
            json_data = memory.serde.loads(decoded_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )
        return memory

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
        logging.info(f"[orchestrator] Gathering resources for: {question}")
        conversation_id = conversation_id or str(uuid.uuid4())

        try:
            # Load conversation state
            logging.info(f"[orchestrator] Loading conversation data")
            conversation_data = get_conversation_data(conversation_id)
            logging.info(f"[orchestrator] Loading memory")
            memory = self._load_memory(conversation_data.get("memory_data", ""))
            logging.info(f"[orchestrator] Memory loaded")
            # Process through agent

            # insert conversation to the memory object
            agent = create_conversation_graph(
                memory=memory,
                organization_id=self.organization_id,
                conversation_id=conversation_id,
            )
            logging.info(f"[orchestrator] Agent created")
            config = {"configurable": {"thread_id": conversation_id}}

            with get_openai_callback() as cb:
                # Get agent response
                logging.info(f"[orchestrator] Invoking agent")
                response = agent.invoke({"question": question}, config)
                logging.info(f"[orchestrator] Agent response")
                return {
                    "conversation_id": conversation_id,
                    "state": ConversationState(
                        question,
                        response["messages"],
                        response["context_docs"],
                        response["requires_web_search"],
                        response["rewritten_query"],
                        response["chat_summary"],
                        response["token_count"],
                    ),
                    "conversation_data": conversation_data,
                    "memory_data": self._serialize_memory(memory, config),
                    "start_time": start_time,
                    "consumed_tokens": cb,
                }

        except Exception as e:
            logging.error(f"[orchestrator] Error retrieving resources: {str(e)}")
            store_agent_error(user_info["id"], str(e), question)

    def _get_model_response(self, model_name: str, system_prompt: str, prompt: str):
        """Helper function to get model response based on model deployment service.
            Azure OpenAI models are handled by the AzureChatOpenAI class while others are handled by the AzureInferenceSDK class.
            Thus, there's a need to handle streaming differently for the two cases.
        
        Args:
            model_name: Deployment name of the model to use (may not be the actual model's name)
            system_prompt: System prompt for the model
            prompt: Prompt for the model

        Yields:
            tuple[str, str]: A tuple containing (chunk, accumulated_response) where:
                - chunk is the current piece of the response
                - accumulated_response is the complete response up to this point
        """
        complete_response = ""
        
        if model_name == "gpt-4o-orchestrator":
            response_llm = AzureChatOpenAI(
                temperature=0,
                openai_api_version="2024-05-01-preview",
                azure_deployment=model_name,
                streaming=True,
                timeout=30,
                max_retries=3,
            )
            tokens = response_llm.stream(
                [LangchainSystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            )
            try:
                while True:
                    try:
                        token = next(tokens)
                        if token:
                            chunk = token.content
                            complete_response += chunk
                            yield chunk, complete_response
                    except StopIteration:
                        break
            except Exception as e:
                logging.error(f"[orchestrator] Error generating response: {str(e)}")
                error_message = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
                yield error_message, error_message
        else:
            endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
            key = os.getenv("AZURE_INFERENCE_SDK_KEY")
            client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
            
            response = client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=prompt)
                ],
                model=model_name,
                max_tokens=10000,
                stream=True
            )
            
            try:
                for update in response:
                    if update.choices and update.choices[0].delta:
                        chunk = update.choices[0].delta.content or ""
                        complete_response += chunk
                        yield chunk, complete_response
            except Exception as e:
                logging.error(f"[orchestrator] Error generating response: {str(e)}")
                error_message = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
                yield error_message, error_message
        
        return complete_response

    def generate_response(
        self,
        conversation_id: str,
        state: ConversationState,
        conversation_data: dict,
        user_info: dict,
        memory_data: str,
        start_time: float,
        model_name: str = "DeepSeek-V3",
    ):
        """Generate final response using context and query."""
        logging.info(f"[orchestrator] Generating response for: {state.question}")
        data = {
            "conversation_id": conversation_id,
            "thoughts": [
                f"Tool name: agent_memory > Query sent: {state.rewritten_query}"
            ],
        }
        yield json.dumps(data)
        context = ""
        max_tokens = 2000
        if state.context_docs:
            context = "\n\n==============================================\n\n".join(
                [
                    f"\nContent: \n\n{doc.page_content}"
                    + (
                        f"\n\nSource: {doc.metadata['source']}"
                        if doc.metadata.get("source")
                        else ""
                    )
                    for doc in state.context_docs
                ]
            )

        history = conversation_data.get("history", [])

        system_prompt = MARKETING_ANSWER_PROMPT

        # add context to the system prompt

        additional_context = f"""

        Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)
        
        <----------- PROVIDED CONTEXT ------------>
        {context}
        <----------- END OF PROVIDED CONTEXT ------------>

        Chat History:

        <----------- PROVIDED CHAT HISTORY ------------>
        {self._clean_chat_history(history)}
        <----------- END OF PROVIDED CHAT HISTORY ------------>

        Chat Summary:

        <----------- PROVIDED CHAT SUMMARY ------------>
        {state.chat_summary}
        <----------- END OF PROVIDED CHAT SUMMARY ------------>
        """

        # add additional context to the system prompt
        system_prompt += additional_context

        prompt = f"""
        
        Question: 
        
        <----------- USER QUESTION ------------>

        ORIGINAL QUESTION: {state.question}

        <----------- END OF USER QUESTION ------------>

        Provide a detailed answer.
        """

        logging.info(f"Prompt: {prompt}")
        # Generate response and update message history
        complete_response = ""

        try:
            for chunk, accumulated_response in self._get_model_response(model_name, system_prompt, prompt):
                complete_response = accumulated_response  # Get the complete response at each step
                yield chunk  # Still stream the chunks to the client
        except Exception as e:
            logging.error(f"[orchestrator] Error generating response: {str(e)}")
            error_message = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            complete_response = error_message
            yield error_message
        
        logging.info(f"[orchestrator] Response generated: {complete_response}")

        #####################################################################################
        # Summary and chat history work
        #####################################################################################
        
        # Use complete_response instead of response["content"]
        current_messages = state.messages if state.messages is not None else []

        llm = AzureChatOpenAI(
            temperature=0,
            openai_api_version="2024-05-01-preview",
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
            timeout=30,
            max_retries=3,
        )

        try:
            # Try to count tokens, fallback to conservative estimate if it fails
            pre_token_count = (
                num_tokens_from_string(messages_to_string(current_messages))
                if current_messages
                else 0
            )
            print(f"Pre token count: {pre_token_count}")

            # Summarize chat history if it exceeds token limit
            # TODO: THIS CHAT SUMMARY IS NOT BEING USED CHECK FOR WHAT IS WHAT USED ON THE GRAPH BEFORE
            if pre_token_count > max_tokens:
                chat_summary = self._summarize_chat(
                    pre_token_count,
                    current_messages,
                    state.chat_summary,
                    max_tokens,
                    llm,
                )
            else:
                chat_summary = state.chat_summary

            # Prepare new messages
            if model_name == "gpt-4o-orchestrator":
                new_messages = [
                    HumanMessage(content=state.rewritten_query),
                    AIMessage(content=complete_response),
                ]
            else:
                new_messages = [
                    HumanMessage(content=state.rewritten_query),
                    AIMessage(content=complete_response),
                ]
            total_messages = (
                (current_messages + new_messages)
                if pre_token_count <= max_tokens
                else new_messages
            )
            post_token_count = num_tokens_from_string(
                messages_to_string(total_messages)
            )
            print(f"Post token count: {post_token_count}")

        except Exception as e:
            print(f"Warning: Token counting failed: {str(e)}")
            # Fallback to simple length-based estimate
            pre_token_count = sum(len(str(m.content)) // 4 for m in current_messages)
            if model_name == "gpt-4o-orchestrator":
                total_messages = current_messages + [
                    HumanMessage(content=state.rewritten_query),
                    AIMessage(content=complete_response),
                ]
            else:
                total_messages = current_messages + [
                    HumanMessage(content=state.rewritten_query),
                    AIMessage(content=complete_response),
                ]

            post_token_count = sum(len(str(m.content)) // 4 for m in total_messages)

        
        answer = self._sanitize_response(complete_response)

        # Update conversation history
        history.extend(
            [
                {"role": "user", "content": state.question},
                {
                    "role": "assistant",
                    "content": answer,
                    "thoughts": [
                        f"Tool name: agent_memory > Query sent: {state.rewritten_query}"
                    ],
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
                },
            }
        )

        update_conversation_data(conversation_id, conversation_data)
        # TODO: ENABLE CONSUME TOKENS FOR RESPONSE GENERATION
        # store_user_consumed_tokens(user_info["id"], cb)


async def stream_run(
    conversation_id: str,
    ask: str,
    url: str,
    client_principal: dict,
    organization_id: str = None,
):
    orchestrator = ConversationOrchestrator(organization_id=organization_id)
    resources = await orchestrator.process_conversation(
        conversation_id, ask, client_principal
    )
    return orchestrator.generate_response(
        resources["conversation_id"],
        resources["state"],
        resources["conversation_data"],
        client_principal,
        resources["memory_data"],
        resources["start_time"],
    )
