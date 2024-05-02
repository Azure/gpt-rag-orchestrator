# imports
import json
import logging
import os
import semantic_kernel as sk
import orc.semantic_skill as ss

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.retrievers import (
    AzureAISearchRetriever,
)

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'debug').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables

BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK") or "true"
BLOCKED_LIST_CHECK = True if BLOCKED_LIST_CHECK.lower() == "true" else False
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK") or "true"
GROUNDEDNESS_CHECK = True if GROUNDEDNESS_CHECK.lower() == "true" else False
RESPONSIBLE_AI_CHECK = os.environ.get("RESPONSIBLE_AI_CHECK") or "true"
RESPONSIBLE_AI_CHECK = True if RESPONSIBLE_AI_CHECK.lower() == "true" else False
CONVERSATION_METADATA = os.environ.get("CONVERSATION_METADATA") or "true"
CONVERSATION_METADATA = True if CONVERSATION_METADATA.lower() == "true" else False

AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.0"
AZURE_OPENAI_TEMPERATURE = float(AZURE_OPENAI_TEMPERATURE)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_TOP_P = float(AZURE_OPENAI_TOP_P)
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"
AZURE_OPENAI_RESP_MAX_TOKENS = int(AZURE_OPENAI_RESP_MAX_TOKENS)
CONVERSATION_MAX_HISTORY = os.environ.get("CONVERSATION_MAX_HISTORY") or "3"
CONVERSATION_MAX_HISTORY = int(CONVERSATION_MAX_HISTORY)

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = "2024-03-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "chat"

ORCHESTRATOR_FOLDER = "orc"
PLUGINS_FOLDER = f"{ORCHESTRATOR_FOLDER}/plugins"
BOT_DESCRIPTION_FILE = f"{ORCHESTRATOR_FOLDER}/bot_description.prompt"

def augment_prompt(query: str):
  # get top 3 results from knowledge base
  retriever = AzureAISearchRetriever(content_key="chunk", top_k=3)
  results = retriever.invoke(query)
  # get the text from the results
  source_knowledge = "\n".join([x.page_content for x in results])
  # feed into an augmented prompt
  augmented_prompt = f"""Using the contexts below, answer the query.
  Contexts:
  {source_knowledge}
  Query: {query}"""
  return augmented_prompt


async def get_answer(question, messages, settings):
    answer_dict = {}
    total_tokens = 0
    
    model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    )
    
    if(len(messages) == 0):
        messages = [
        SystemMessage(content="You are an AI assistant that helps people find information."),
        ]
    prompt = HumanMessage(
    content=augment_prompt(question) # modify this with search query
    )
    messages.append(prompt)
    res = model.invoke(messages)
    messages.append(res)
    
    answer_dict["answer"] = res.content
    answer_dict["ai_message"] = res
    answer_dict["human_message"] = messages
    answer_dict["total_tokens"] = total_tokens
    
    
    return answer_dict