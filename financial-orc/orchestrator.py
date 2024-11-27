import os
import uuid
import logging
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from shared.cosmos_db import store_agent_error
from .graphs.main import create_main_agent

# Set up logging
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def run(conversation_id, question, documentName, client_principal):
    try:
        if conversation_id is None or conversation_id == "":
            conversation_id = str(uuid.uuid4())
            logging.info(
                f"[financial-orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id."
            )

        # memory
        memory = MemorySaver()

        agent_executor = create_main_agent(checkpointer=memory, verbose=True)

        config = {"configurable": {"thread_id": conversation_id}}
        inputs = {"messages": [HumanMessage(content=question)]}

        response = agent_executor.invoke(inputs, stream_mode="values", config=config)
        answer = response["messages"][-1].content

        logging.info(f"[financial-orchestrator] {conversation_id} response: {answer}")

        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
    except Exception as e:
        logging.error(f"[financial-orchestrator] {conversation_id} error: {str(e)}")
        store_agent_error(client_principal["id"], str(e), question)
        response = {
            "conversation_id": conversation_id,
            "answer": f"There was an error processing your request in financial orchestrator. Error: {str(e)}",
            "data_points": "",
            "question": question,
            "documentName": documentName,
            "thoughts": [],
        }
        return response
