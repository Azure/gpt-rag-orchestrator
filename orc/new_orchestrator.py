import os
import logging
import base64
import uuid
import time
import re
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.memory import MemorySaver
from orc.graphs.main import create_conversation_graph
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
    store_user_consumed_tokens,
)

# Configure logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG").upper())


class ConversationOrchestrator:
    """Manages conversation flow and state between user and AI agent."""

    def __init__(self):
        """Initialize orchestrator with storage URL."""
        self.storage_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")

    def _sanitize_response(self, text: str) -> str:
        """Remove sensitive storage URLs from response text."""
        if self.storage_url in text:
            regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({self.storage_url})?(\/?documents\/?)?"
            return re.sub(regex, "", text)
        return text

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

    def _serialize_memory(self, memory: MemorySaver, config: dict) -> str:
        """Convert memory state to base64 encoded string for storage."""
        serialized = memory.serde.dumps(memory.get_tuple(config))
        return base64.b64encode(serialized).decode("utf-8")

    async def process_conversation(
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
        conversation_id = conversation_id or str(uuid.uuid4())

        try:
            # Load conversation state
            conversation_data = get_conversation_data(conversation_id)
            memory = self._load_memory(conversation_data.get("memory_data", ""))

            # Process through agent
            agent = create_conversation_graph()
            config = {"configurable": {"thread_id": conversation_id}}

            with get_openai_callback() as cb:
                # Get agent response
                response = agent.invoke({"question": question}, config)
                messages = response.get("messages", [])

                answer = (
                    messages[-1].content
                    if messages
                    else "No response generated. Please try again."
                )
                answer = self._sanitize_response(answer)

            # Update conversation history
            history = conversation_data.get("history", [])
            history.extend(
                [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": answer,
                        "thoughts": [
                            f"Tool name: agent_memory > Query sent: {question}"
                        ],
                    },
                ]
            )

            # Save updated state
            conversation_data.update(
                {
                    "history": history,
                    "memory_data": self._serialize_memory(memory, config),
                    "interaction": {
                        "user_id": user_info["id"],
                        "user_name": user_info["name"],
                        "response_time": round(time.time() - start_time, 2),
                    },
                }
            )

            update_conversation_data(conversation_id, conversation_data)
            store_user_consumed_tokens(user_info["id"], cb)

            return {
                "conversation_id": conversation_id,
                "answer": answer,
                "thoughts": [f"Tool name: agent_memory > Query sent: {question}"],
            }

        except Exception as e:
            logging.error(f"[orchestrator] Error: {str(e)}")
            store_agent_error(user_info["id"], str(e), question)
            return {
                "conversation_id": conversation_id,
                "answer": "Service is currently unavailable, please retry later.",
                "thoughts": [],
            }


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
