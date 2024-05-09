# imports
import logging
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import HumanMessagePromptTemplate
from langchain.chains import LLMChain
from shared.tools import LineListOutputParser, retrieval_transform
from langchain_community.retrievers import AzureAISearchRetriever

# logging level

logging.getLogger("azure").setLevel(logging.WARNING)
LOGLEVEL = os.environ.get("LOGLEVEL", "debug").upper()
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

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-03-01-preview"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = "chat"

ORCHESTRATOR_FOLDER = "orc"
PLUGINS_FOLDER = f"{ORCHESTRATOR_FOLDER}/plugins"
BOT_DESCRIPTION_FILE = f"{ORCHESTRATOR_FOLDER}/bot_description.prompt"


# replaced by format_messages in the orchestrator
# >:(
# def augment_prompt(query: str, docs: list):
#     # get the text from the results
#     source_knowledge = retrieval_transform(docs)
#     # feed into an augmented prompt
#     augmented_prompt = f"""Using the contexts below, answer the query.
#   Contexts:
#   {source_knowledge}
#   Query: {query}"""
#     return augmented_prompt

def replace_numbers_with_paths(text, paths):
    citations = re.findall(r"\[([0-9]+(?:,[0-9]+)*)\]", text)
    for citation in citations:
        citation = citation.split(',')
        for c in citation:
            c = int(c)
            text = text.replace(f"[{c}]", "["+paths[c-1]+"]")
    logging.info(f"[orchestrator] response with citations {text}")
    return text

def get_document_retriever(model):
    template = """You are an AI language model assistant. You have the capability to perform advanced vector-based queries.
Your task is to construct one search query using only nouns, to retrieve relevant documents from a vector database.
Identify key concepts from the question. Combine these concepts into a relevant noun phrase.
Do not use the 'Search query' at the beginning of the query.
Original question: {question}"""

    sq_prompt = ChatPromptTemplate.from_template(template)

    llm_chain = LLMChain(
        llm=model, prompt=sq_prompt, output_parser=LineListOutputParser()
    )

    retriever = MultiQueryRetriever(
        retriever=AzureAISearchRetriever(content_key="chunk", top_k=3),
        llm_chain=llm_chain,
    )

    return retriever


async def get_answer(question, messages, settings):
    answer_dict = {}
    total_tokens = 0
    try:
        model = AzureChatOpenAI(
            temperature=settings['temperature'],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )
        
        if len(messages) == 0:
            messages = [
                SystemMessage(
                    content="You are FreddAid, a friendly marketing assistant dedicated to uncovering insights and developing effective strategies."
                ),
            ]

        # get document retriever and create retrieval chain
        retriever = get_document_retriever(model)
        retrieval_chain = retriever | retrieval_transform

        # get source knowledge from retrieval chain documents
        source_knowledge, sources = retrieval_chain.invoke(question)
        humanMessage = HumanMessagePromptTemplate.from_template(
            """Answer the following question based on this context:\n{context}\nQuestion: {question}\n Make sure you cite the source number as [x]. Do not add the word Source before the number."""
        )

        # format the message into a human message and append to messages
        humanMessage = humanMessage.format_messages(
            context=source_knowledge, question=question
        )
        messages.append(humanMessage[0])

        # get the response from the model
        res = model.invoke(messages)
        messages.append(res)

        answer_dict["answer"] = replace_numbers_with_paths(res.content, sources)
        answer_dict["ai_message"] = res
        answer_dict["human_message"] = humanMessage[0]
        answer_dict["total_tokens"] = total_tokens
        answer_dict["sources"] = sources

    except Exception as e:
        logging.error(f"[code_orchest] exception when executing RAG flow. {e}")
        answer_dict["answer"] = f"RAG flow: exception: {e}"

    answer_dict["total_tokens"] = total_tokens
    answer_dict["user_ask"] = question

    return answer_dict
