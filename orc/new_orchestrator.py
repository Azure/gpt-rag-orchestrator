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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain.schema import Document
from shared.prompts import MARKETING_ORC_PROMPT, MARKETING_ANSWER_PROMPT, QUERY_REWRITING_PROMPT
from shared.tools import num_tokens_from_string, messages_to_string

# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG").upper())

@dataclass
class ConversationState():
    """State container for conversation flow management.

    Attributes:
        question: Current user query
        messages: Conversation history as a list of messages
        context_docs: Retrieved documents from various sources
        requires_web_search: Flag indicating if web search is needed
    """

    question: str
    messages: List[AIMessage | HumanMessage] = field(default_factory=list) # track all messages in the conversation
    context_docs: List[Document] = field(default_factory=list)
    requires_web_search: bool = field(default=False)
    rewritten_query: str = field(default_factory=str) # rewritten query for better search 
    chat_summary: str = field(default_factory=str)
    token_count: int = field(default_factory=int)
class ConversationOrchestrator:
    """Manages conversation flow and state between user and AI agent."""
    def __init__(self):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        
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
    
    def _summarize_chat(token_count: int, 
                        messages: List[AIMessage | HumanMessage], 
                        chat_summary: str, 
                        max_tokens: int, 
                        llm: AzureChatOpenAI) -> dict:
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
                        SystemMessage(content="You are a helpful assistant that summarizes conversations."),
                        HumanMessage(content=f"Previous summary:\n{chat_summary}\n\nNew messages to incorporate:\n{messages}\n\nPlease extend the summary. Return only the summary text.")
                    ]

                else:
                    messages = [
                        SystemMessage(content="You are a helpful assistant that summarizes conversations."),
                        HumanMessage(content=f"Summarize this conversation history. Return only the summary text:\n{messages}")
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
        if memory_data:
            decoded_data = base64.b64decode(memory_data)
            json_data = memory.serde.loads(decoded_data)
            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )
        return memory

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
            conversation_data = get_conversation_data(conversation_id)
            memory = self._load_memory(conversation_data.get("memory_data", ""))

            # Process through agent
            agent = create_conversation_graph(memory = memory)
            config = {"configurable": {"thread_id": conversation_id}}

            with get_openai_callback() as cb:
                # Get agent response
                response = agent.invoke({"question": question}, config)
                return {
                    "conversation_id": conversation_id,
                    "state": ConversationState(question, response["messages"], response["context_docs"], response["requires_web_search"], response["rewritten_query"], response["chat_summary"], response["token_count"]),
                    "conversation_data": conversation_data,
                    "memory_data": self._serialize_memory(memory, config),
                    "start_time": start_time,
                    "consumed_tokens": cb
                }
            
        except Exception as e:
            logging.error(f"[orchestrator] Error retrieving resources: {str(e)}")
            store_agent_error(user_info["id"], str(e), question)
            
    def generate_response(self,conversation_id: str, state: ConversationState, conversation_data: dict, user_info: dict, memory_data: str,start_time: float):
        """Generate final response using context and query."""
        logging.info(f"[orchestrator] Generating response for: {state.question}")
        data = {
            "conversation_id": conversation_id,
            "thoughts": [f"Tool name: agent_memory > Query sent: {state.rewritten_query}"]
        }
        yield json.dumps(data)
        context = ""
        max_tokens = 2000
        if state.context_docs:
            context = "\n\n==============================================\n\n".join([
                f"\nContent: \n\n{doc.page_content}" + 
                (f"\n\nSource: {doc.metadata['source']}" if doc.metadata.get("source") else "")
                for doc in state.context_docs
            ])

        system_prompt = MARKETING_ANSWER_PROMPT
        prompt = f"""
        
        Question: 
        
        <----------- USER QUESTION ------------>
        REWRITTEN QUESTION: {state.rewritten_query}

        ORIGINAL QUESTION: {state.question}
        <----------- END OF USER QUESTION ------------>
        
        
        Context: (MUST PROVIDE CITATIONS FOR ALL SOURCES USED IN THE ANSWER)
        
        <----------- PROVIDED CONTEXT ------------>
        {context}
        <----------- END OF PROVIDED CONTEXT ------------>

        Chat History:

        <----------- PROVIDED CHAT HISTORY ------------>
        {state.messages}
        <----------- END OF PROVIDED CHAT HISTORY ------------>

        Chat Summary:

        <----------- PROVIDED CHAT SUMMARY ------------>
        {state.chat_summary}
        <----------- END OF PROVIDED CHAT SUMMARY ------------>

        Provide a detailed answer.
        """

        # Generate response and update message history
        response = {
            "content": "",
        }
        
        response_llm = AzureChatOpenAI(
            temperature=0,
            openai_api_version="2024-05-01-preview",
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
            timeout=30,
            max_retries=3)
        
        tokens = response_llm.stream([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        try:
            while True:
                try:
                    token = next(tokens)
                    if token:
                        response["content"] += f"{token.content}"
                        yield f"{token.content}"
                except StopIteration:
                    break
        except Exception as e:
            logging.error(f"[orchestrator] Error generating response: {str(e)}")
            response["content"] = "I'm sorry, I'm having trouble generating a response right now. Please try again later."
            yield response["content"]
                        
        logging.info(f"[orchestrator] Response generated: {response['content']}")
        
        #####################################################################################
        # Summary and chat history work
        #####################################################################################   
        current_messages = state.messages if state.messages is not None else []
        
        llm = AzureChatOpenAI(
            temperature=0,
            openai_api_version= "2024-05-01-preview",
            azure_deployment="gpt-4o-orchestrator",
            streaming=True,
            timeout=30,
            max_retries=3
        )

        try:
            # Try to count tokens, fallback to conservative estimate if it fails
            pre_token_count = num_tokens_from_string(messages_to_string(current_messages)) if current_messages else 0
            print(f"Pre token count: {pre_token_count}")

            # Summarize chat history if it exceeds token limit
            #TODO: THIS CHAT SUMMARY IS NOT BEING USED CHECK FOR WHAT IS WHAT USED ON THE GRAPH BEFORE
            if pre_token_count > max_tokens:
                chat_summary = self._summarize_chat(pre_token_count, current_messages, state.chat_summary, max_tokens, llm)
            else:
                chat_summary = state.chat_summary

            # Prepare new messages
            new_messages = [HumanMessage(content=state.rewritten_query), AIMessage(content=response["content"])]
            total_messages = (current_messages + new_messages) if pre_token_count <= max_tokens else new_messages
            post_token_count = num_tokens_from_string(messages_to_string(total_messages))
            print(f"Post token count: {post_token_count}")
        
        except Exception as e:
            print(f"Warning: Token counting failed: {str(e)}")
            # Fallback to simple length-based estimate
            pre_token_count = sum(len(str(m.content)) // 4 for m in current_messages)

            total_messages = current_messages + [HumanMessage(content=state.rewritten_query), 
                                                AIMessage(content=response["content"])]

            post_token_count = sum(len(str(m.content)) // 4 for m in total_messages)
        
        answer = self._sanitize_response(response["content"])
        # Update conversation history
        history = conversation_data.get("history", [])
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
        #TODO: ENABLE CONSUME TOKENS FOR RESPONSE GENERATION
        #store_user_consumed_tokens(user_info["id"], cb)


    
async def run(conversation_id: str, ask: str, url: str, client_principal: dict) -> dict:
    """
    Main entry point for processing conversations.

    Args:
        conversation_id: Unique identifier for conversation
        ask: User's question
        url: Base URL for the service
        client_principal: User information dictionary

    Returns:
        dict: Processed response from the orchestrator
    """
    orchestrator = ConversationOrchestrator()
    return await orchestrator.process_conversation(
        conversation_id, ask, client_principal
    )


async def stream_run(conversation_id: str, ask: str, url: str, client_principal: dict):
    orchestrator = ConversationOrchestrator()
    resources =  await orchestrator.process_conversation(
        conversation_id, ask, client_principal
    )
    return orchestrator.generate_response(resources["conversation_id"],resources["state"], resources["conversation_data"], client_principal, resources["memory_data"], resources["start_time"])
