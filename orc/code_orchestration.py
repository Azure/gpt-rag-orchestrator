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
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
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
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        )

        if len(messages) == 0:
            messages = [
                SystemMessage(
                    content="You are an AI assistant that helps people find information."
                ),
            ]

        # get document retriever and create retrieval chain
        retriever = get_document_retriever(model)

        # do not parse into a string yet ↓
        # retrieval_chain = retriever | retrieval_transform
        retrieval_chain = retriever

        # get source knowledge from retrieval chain documents
        docs = retrieval_chain.invoke(question)

        # document summarization happens here
        map_prompt_template = """
        Write a summary of this chunk of text that includes the main points and any important details.
        {text}
        """

        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt_template = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
        """

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        map_reduce_chain = load_summarize_chain(
            llm=model,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            return_intermediate_steps=True,
        )

        map_reduce_outputs = map_reduce_chain.invoke(
            {"input_documents": docs}
        )

        # parsing the output
        final_mp_data = []
        for doc, out in zip(
            map_reduce_outputs["input_documents"],
            map_reduce_outputs["intermediate_steps"],
        ):
            output = {}
            output["filepath"] = doc.metadata["filepath"]
            # output["file_type"] = doc.metadata["filepath"]
            # output["page_number"] = doc.metadata["page"]
            output["content"] = doc.page_content
            output["summary"] = out
            final_mp_data.append(output)

        summaries = [x['summary'] for x in final_mp_data]

        # We can use the non-summarized text here to get a better answer
        # but we have to keep in mind, the text we need to save is the summarized one
        # to prevent the model to consume more tokens than needed, but having the key concepts
        # in the summarized text
        source_knowledge = "\n---\n".join(summaries) 
        # end of document summarization

        # source_knowledge = retrieval_transform(docs)

        humanMessage = HumanMessagePromptTemplate.from_template(
            """Answer the following question based on this context:\n{context}\nQuestion: {question}"""
        )

        # format the message into a human message and append to messages
        humanMessage = humanMessage.format_messages(
            context=source_knowledge, question=question
        )
        messages.append(humanMessage[0])

        # get the response from the model
        res = model.invoke(messages)
        messages.append(res)

        answer_dict["answer"] = res.content
        answer_dict["ai_message"] = res
        answer_dict["human_message"] = humanMessage[0]
        answer_dict["total_tokens"] = total_tokens

    except Exception as e:
        logging.error(f"[code_orchest] exception when executing RAG flow. {e}")
        answer_dict["answer"] = f"RAG flow: exception: {e}"

    answer_dict["total_tokens"] = total_tokens
    answer_dict["user_ask"] = question

    return answer_dict
