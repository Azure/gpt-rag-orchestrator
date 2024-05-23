import re
import logging
import os
import time
import uuid
from shared.util import get_setting
from shared.cosmos_db import store_user_consumed_tokens, store_prompt_information
from azure.identity.aio import DefaultAzureCredential
import orc.code_orchestration as code_orchestration
from shared.cosmos_db import (
    get_conversation_data,
    update_conversation_data,
    store_agent_error,
)

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_openai import AzureChatOpenAI

from langchain.chains import LLMMathChain
from langchain.chains import LLMChain

# from langchain.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_openai_functions_agent
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities import BingSearchAPIWrapper

# from langchain_community.retrievers import AzureAISearchRetriever
from shared.tools import AzureAISearchRetriever
from langchain.tools.retriever import create_retriever_tool
from datetime import date

# logging level
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)

# Constants set from environment variables (external services credentials and configuration)

# Cosmos DB
AZURE_DB_ID = os.environ.get("AZURE_DB_ID")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_URI = f"https://{AZURE_DB_ID}.documents.azure.com:443/"

# AOAI
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM") or "false"
AZURE_OPENAI_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Langchain
CONVERSATION_MAX_HISTORY = os.environ.get("CONVERSATION_MAX_HISTORY") or "12"
CONVERSATION_MAX_HISTORY = int(CONVERSATION_MAX_HISTORY)

# BING

BING_SEARCH_API_KEY = os.environ.get("BING_SEARCH_API_KEY")
BING_SEARCH_URL = os.environ.get("BING_SEARCH_URL")

ANSWER_FORMAT = "html"  # html, markdown, none


def get_credentials():
    is_local_env = os.getenv("LOCAL_ENV") == "true"
    # return DefaultAzureCredential(exclude_managed_identity_credential=is_local_env, exclude_environment_credential=is_local_env)
    return DefaultAzureCredential()


def get_settings(client_principal):
    # use cosmos to get settings from the logged user
    data = get_setting(client_principal)
    temperature = 0.0 if "temperature" not in data else data["temperature"]
    frequency_penalty = (
        0.0 if "frequencyPenalty" not in data else data["frequencyPenalty"]
    )
    presence_penalty = 0.0 if "presencePenalty" not in data else data["presencePenalty"]
    settings = {
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    logging.info(f"[orchestrator] settings: {settings}")
    return settings


def instanciate_messages(messages_data):
    messages = []
    try:
        for message_data in messages_data:
            if message_data["type"] == "human":
                message = HumanMessage(**message_data)
            elif message_data["type"] == "system":
                message = SystemMessage(**message_data)
            elif message_data["type"] == "ai":
                message = AIMessage(**message_data)
            else:
                Exception(f"Message type {message_data['type']} not recognized.")
                message.from_dict(message_data)
            messages.append(message)
        return messages
    except Exception as e:
        logging.error(f"[orchestrator] error instanciating messages: {e}")
        return []


def replace_numbers_with_paths(text, paths):
    citations = re.findall(r"\[([0-9]+(?:,[0-9]+)*)\]", text)
    for citation in citations:
        citation = citation.split(",")
        for c in citation:
            c = int(c)
            text = text.replace(f"[{c}]", "[" + paths[c - 1] + "]")
    logging.info(f"[orchestrator] response with citations {text}")
    return text


def sort_string(string):
    return " ".join(sorted(string))


@tool
def current_time():
    """Returns the current date."""
    return f"Today's date: {date.today()}"


async def run(conversation_id, ask, client_principal):

    start_time = time.time()

    # settings
    settings = get_settings(client_principal)

    # Get conversation stored in CosmosDB

    # create conversation_id if not provided
    if conversation_id is None or conversation_id == "":
        conversation_id = str(uuid.uuid4())
        logging.info(
            f"[orchestrator] {conversation_id} conversation_id is Empty, creating new conversation_id."
        )

    logging.info(f"[orchestrator] {conversation_id} starting conversation flow.")

    # get conversation data from CosmosDB
    conversation_data = get_conversation_data(conversation_id)

    # load messages data and instanciate them

    messages_data = conversation_data["messages_data"]
    messages = instanciate_messages(messages_data)

    # initialize other settings
    model_kwargs = dict(
        frequency_penalty=settings["frequency_penalty"],
        presence_penalty=settings["presence_penalty"],
    )
    # Initialize model
    model = AzureChatOpenAI(
        temperature=settings["temperature"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        model_kwargs=model_kwargs,
    )

    # Initialize memory
    memory = ConversationBufferWindowMemory(
        k=CONVERSATION_MAX_HISTORY, memory_key="chat_history", return_messages=True
    )

    input, output = {}, {}
    for message in messages:
        if message.type == "human":
            input["input"] = message.content
        if message.type == "ai":
            output["output"] = message.content
        if "input" in input and "output" in output:
            memory.save_context(input, output)
            input, output = {}, {}

    # Define built-in tools

    llm_math = LLMMathChain(llm=model)

    @tool
    def math_tool(query: str) -> str:
        """Useful for when you need to answer questions about math."""
        return llm_math.invoke(query)

    # bing_search = BingSearchAPIWrapper(k=3)
    documents = []

    # arguments for code orchestration
    args = {
        "model": model,
        "question": ask,
        "messages": messages,
        "documents": documents,
    }
    retriever = AzureAISearchRetriever(
        content_key="chunk", top_k=3, api_version="2024-03-01-preview"
    )
    # Create agent tools
    retriever = create_retriever_tool(
        retriever,
        "Retrieval",
        "Useful for when you need to answer questions about consumer behavior, consumer pulse, segments and segmentation.",
    )
    tools = [retriever, math_tool, current_time]
    # tools = [
    #     Tool(
    #         name="Calculator",
    #         func=llm_math.run,
    #         description="Useful for when you need to answer questions about math.",
    #     ),
    #     # Tool(
    #     #   name="Bing_Search",
    #     #   description="A tool to search the web. Use it when you need to find current information that is not available in the library tools.",
    #     #   func=bing_search.run
    #     # ),
    #     Tool(
    #         name="Current_Time",
    #         description="Returns current time.",
    #         func=lambda _: current_time(),
    #     ),
    #     # Tool(
    #     #     name="Sort_String",
    #     #     func=lambda string: sort_string(string),
    #     #     description="Useful for when you need to sort a string",
    #     #     verbose=True,
    #     # ),
    # ]

    # Define agent prompt template
    system = """Your name is FredAid.
    You are a Marketing Expert designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
    YOU MUST FOLLOW THESE INSTRUCTIONS:
    1. Include a citation next to every fact with the file path within brackets. For example: [http://home/file.txt].
    2. Do not call any of the retriever tools more than once with the same query.

    Your primary goal is to be helpful, and accurate. If you need to use any tool to enhance your response, do so effectively."""

    human = """

    {input}

    {agent_scratchpad}

    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )

    # Create agent
    agent = create_openai_functions_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        return_intermediate_steps=True,
    )
    chat_history = memory.buffer_as_messages

    # 1) get answer from agent
    try:
        with get_openai_callback() as cb:
            response = agent_executor.invoke(
                {
                    "input": ask,
                    "chat_history": chat_history,
                }
            )
        logging.info(
            f"[orchestrator] {conversation_id} agent response: {response['output'][:50]}"
        )
    except Exception as e:
        logging.error(f"[orchestrator] {conversation_id} error: {e.message}")
        store_agent_error(client_principal["id"], e.message)
        response = {
            "conversation_id": conversation_id,
            "answer": f"There was an error processing your request. Error: {e}",
            "data_points": "",
            "thoughts": ask,
        }
        return response

    # agent_dict = agent_executor.dict()

    # for value in agent_dict:
    #     logging.error(f"{value}: {agent_dict[value]}")

    # 2) update and save conversation (containing history and conversation data)

    message_list = memory.buffer_as_messages

    # messages data

    # user message
    messages_data.append(response["input"])
    # ai message
    messages_data.append(response["output"])

    # history
    history = conversation_data["history"]
    history.append({"role": "user", "content": ask})
    history.append({"role": "assistant", "content": response["output"]})

    conversation_data["history"] = history
    conversation_data["messages_data"] = messages_data

    # conversation data
    response_time = round(time.time() - start_time, 2)
    interaction = {
        "user_id": client_principal["id"],
        "user_name": client_principal["name"],
        "response_time": response_time,
    }
    conversation_data["interaction"] = interaction

    if len(documents) > 0:
        interaction["sources"] = documents

    # Clear documents to prevent memory garbage
    documents.clear()

    # store updated conversation data
    update_conversation_data(conversation_id, conversation_data)

    # 3) store user consumed tokens
    store_user_consumed_tokens(client_principal["id"], cb)

    # 4) store prompt information in CosmosDB

    # TODO: store prompt information

    # 5) return answer
    response = {
        "conversation_id": conversation_id,
        "answer": response["output"],
        "data_points": interaction["sources"] if "sources" in interaction else "",
        "thoughts": response["input"],
    }

    logging.info(
        f"[orchestrator] {conversation_id} finished conversation flow. {response_time} seconds."
    )

    return response
