import os
import logging
import base64
import uuid
import time
import re
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from orc.graphs.main import create_main_agent
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)
from shared.cosmos_db import store_user_consumed_tokens
from langchain_community.callbacks import get_openai_callback

# logging level
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)
AZURE_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")


async def run(conversation_id, ask, url, client_principal):
    try:
        start_time = time.time()
        logging.info(
            "[orchestrator] Starting conversation flow for user: %s",
            client_principal.get("id"),
        )

        # Create conversation_id if not provided
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())
            logging.info(
                "[orchestrator] Generated new conversation_id: %s as no ID was provided.",
                conversation_id,
            )
        else:
            logging.info(
                "[orchestrator] Using provided conversation_id: %s", conversation_id
            )

        # Get conversation data from CosmosDB
        logging.info(
            f"[orchestrator] Fetching conversation data for ID: {conversation_id}"
        )
        conversation_data = get_conversation_data(conversation_id)

        # Load memory data and deserialize
        memory_data_string = conversation_data.get("memory_data", "")
        memory = MemorySaver()

        if memory_data_string:
            try:
                logging.info(
                    f"[orchestrator] Decoding and loading memory data for conversation_id: {conversation_id}"
                )
                decoded_data = base64.b64decode(memory_data_string)
                json_data = memory.serde.loads(decoded_data)
                if json_data:
                    logging.info(
                        f"[orchestrator] Memory data loaded successfully for conversation_id: {conversation_id}"
                    )
                    memory.put(
                        config=json_data[0],
                        checkpoint=json_data[1],
                        metadata=json_data[2],
                    )
            except Exception as e:
                logging.error(
                    f"[orchestrator] Failed to load memory data for conversation_id: {conversation_id}. Error: {str(e)}"
                )
                raise

        # Create agent
        logging.info(
            f"[orchestrator] Creating main agent for conversation_id: {conversation_id}"
        )
        agent_executor = create_main_agent(checkpointer=memory, verbose=True)

        # Config
        config = {"configurable": {"thread_id": conversation_id}}
        logging.info(
            f"[orchestrator] Configuration set for conversation_id: {conversation_id}"
        )

        # Agent response
        try:
            with get_openai_callback() as cb:
                logging.info(
                    f"[orchestrator] Invoking agent for conversation_id: {conversation_id} with question: {ask}"
                )
                response = agent_executor.invoke(
                    {"question": ask},
                    config,
                )
                text = (
                    response["combined_messages"][-1].content
                    if response["combined_messages"]
                    else ""
                )
                logging.info(
                    f"[orchestrator] Raw agent response: {text[:50]} (truncated)"
                )

                # Clean up any sensitive URLs in the response
                if AZURE_STORAGE_ACCOUNT_URL in text:
                    regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({AZURE_STORAGE_ACCOUNT_URL})?(\/?documents\/?)?"
                    text = re.sub(regex, "", text)
                    response["combined_messages"][-1].content = text
                    logging.info(
                        f"[orchestrator] Sanitized response for conversation_id: {conversation_id}"
                    )
        except Exception as e:
            logging.error(
                f"[orchestrator] Error invoking agent for conversation_id: {conversation_id}. Error: {e.__class__.__name__}, {str(e)}"
            )
            store_agent_error(client_principal.get("id"), str(e), ask)
            response = {
                "conversation_id": conversation_id,
                "answer": "Service is currently unavailable, please retry later.",
                "data_points": "",
                "thoughts": ask,
            }
            return response

        # Update conversation history
        history = conversation_data.get("history", [])
        logging.info(
            f"[orchestrator] Updating conversation history for conversation_id: {conversation_id}"
        )
        history.append({"role": "user", "content": ask})

        thoughts = []
        if not thoughts:
            thoughts.append(f"Tool name: agent_memory > Query sent: {ask}")

        history.append(
            {
                "role": "assistant",
                "content": response["combined_messages"][-1].content,
                "thoughts": thoughts,
            }
        )

        # Serialize memory
        logging.info(
            f"[orchestrator] Serializing memory data for conversation_id: {conversation_id}"
        )
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        byte_string = base64.b64encode(serialized_data)
        b64_tosave = byte_string.decode("utf-8")

        # Update conversation data with new history and memory
        logging.info(
            f"[orchestrator] Updating CosmosDB for conversation_id: {conversation_id}"
        )
        conversation_data["history"] = history
        conversation_data["memory_data"] = b64_tosave

        # Add interaction details
        response_time = round(time.time() - start_time, 2)
        interaction = {
            "user_id": client_principal["id"],
            "user_name": client_principal["name"],
            "response_time": response_time,
        }
        conversation_data["interaction"] = interaction

        # Store updated conversation data
        update_conversation_data(conversation_id, conversation_data)

        # Store user consumed tokens
        logging.info(
            f"[orchestrator] Storing consumed tokens for user: {client_principal.get('id')}"
        )
        store_user_consumed_tokens(client_principal.get("id"), cb)

        # Final response
        response = {
            "conversation_id": conversation_id,
            "answer": response["combined_messages"][-1].content,
            "thoughts": thoughts,
        }

        logging.info(
            f"[orchestrator] Finished conversation flow for conversation_id: {conversation_id}. Total time: {response_time} seconds."
        )
        return response

    except Exception as e:
        logging.error(
            f"[orchestrator] Unexpected error for conversation_id: {conversation_id}. Error: {str(e)}"
        )
        store_agent_error(client_principal.get("id"), str(e), ask)
        response = {
            "conversation_id": conversation_id,
            "answer": f"There was an error processing your request. Error: {str(e)}",
            "data_points": "",
            "question": ask,
            "thoughts": [],
        }
        return response
