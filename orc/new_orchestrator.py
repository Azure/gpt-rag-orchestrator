import os
import logging
import base64
import uuid
import time
import re
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from orc.agent import create_agent
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

        # create conversation_id if not provided
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())
            logging.info(
                f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id."
            )

        model = AzureChatOpenAI(
            temperature=0.3, 
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_deployment="Agent"
        )
        mini_model = AzureChatOpenAI(
            temperature=0, 
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_deployment="gpt-4o-mini"
        )

        # get conversation data from CosmosDB
        conversation_data = get_conversation_data(conversation_id)
        # load memory data and deserialize
        memory_data_string = conversation_data["memory_data"]

        # memory
        memory = MemorySaver()
        if memory_data_string != "":
            logging.info(f"[orchestrator] {conversation_id} loading memory data.")
            decoded_data = base64.b64decode(memory_data_string)
            json_data = memory.serde.loads(decoded_data)

            if json_data:
                memory.put(
                    config=json_data[0], checkpoint=json_data[1], metadata=json_data[2]
                )

        # create agent
        agent_executor = create_agent(
            model, 
            mini_model, 
            checkpointer=memory,
            verbose=True
        )

        # config
        config = {"configurable": {"thread_id": conversation_id}}

        # agent response
        try:
            with get_openai_callback() as cb:
                response = agent_executor.invoke(
                    {"question": ask},
                    config,
                )
                if AZURE_STORAGE_ACCOUNT_URL in response["generation"]:
                    regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({AZURE_STORAGE_ACCOUNT_URL})?(\/?documents\/?)?"
                    response["generation"] = re.sub(regex, "", response["generation"])
                logging.info(f"[orchestrator] {conversation_id} agent response: {response['generation'][:50]}")
        except Exception as e:
            logging.error(f"[orchestrator] error: {e.__class__.__name__}")
            logging.error(f"[orchestrator] {conversation_id} error: {str(e)}")
            store_agent_error(client_principal["id"], str(e), ask)
            response = {
                "conversation_id": conversation_id,
                "answer": f"Service is currently unavailable, please retry later",
                "data_points": "",
                "thoughts": ask,
            }
            return response

        # history
        history = conversation_data["history"]
        history.append({"role": "user", "content": ask})
        thoughts = []

        if len(thoughts) == 0:
            thoughts.append(f"Tool name: agent_memory > Query sent: {ask}")

        history.append(
            {
                "role": "assistant",
                "content": response["generation"],
                "thoughts": thoughts,
            }
        )

        # memory serialization
        _tuple = memory.get_tuple(config)
        serialized_data = memory.serde.dumps(_tuple)
        byte_string = base64.b64encode(serialized_data)
        b64_tosave = byte_string.decode("utf-8")

        # set values on cosmos object
        conversation_data["history"] = history
        conversation_data["memory_data"] = b64_tosave

        # conversation data
        response_time = round(time.time() - start_time, 2)
        interaction = {
            "user_id": client_principal["id"],
            "user_name": client_principal["name"],
            "response_time": response_time,
        }
        conversation_data["interaction"] = interaction

        # store updated conversation data
        update_conversation_data(conversation_id, conversation_data)

        # 3) store user consumed tokens
        store_user_consumed_tokens(client_principal["id"], cb)

        # 4) return answer
        response = {
            "conversation_id": conversation_id,
            "answer": response["generation"],
            "thoughts": thoughts,
        }

        logging.info(
            f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds."
        )

        return response
    except Exception as e:
        logging.error(f"[orchestrator] {conversation_id} error: {str(e)}")
        store_agent_error(client_principal["id"], str(e), ask)
        response = {
            "conversation_id": conversation_id,
            "answer": f"There was an error processing your request. Error: {str(e)}",
            "data_points": "",
            "question": ask,
            "thoughts": [],
        }
        return response
