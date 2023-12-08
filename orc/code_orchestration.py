import json
import logging
import os
import re
import time
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message
from shared.util import truncate_to_max_tokens, number_of_tokens, get_aoai_config, get_blocked_list
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills import ConversationSummarySkill
from semantic_kernel.connectors.ai.open_ai.semantic_functions.open_ai_chat_prompt_template import (
    OpenAIChatPromptTemplate,
)
from semantic_kernel.connectors.ai.open_ai.utils import (
    chat_completion_with_function_call,
    get_function_calling_object,
)

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables

BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK") or "true"
BLOCKED_LIST_CHECK = True if BLOCKED_LIST_CHECK.lower() == "true" else False
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK") or "true"
GROUNDEDNESS_CHECK = True if GROUNDEDNESS_CHECK.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TEMPERATURE = float(AZURE_OPENAI_TEMPERATURE)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_TOP_P = float(AZURE_OPENAI_TOP_P)
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"
AZURE_OPENAI_RESP_MAX_TOKENS = int(AZURE_OPENAI_RESP_MAX_TOKENS)

SYSTEM_MESSAGE_PATH = f"orc/prompts/system_message.prompt"

def initialize_kernel():
    kernel = sk.Kernel(log=myLogger)
    chatgpt_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
    kernel.add_chat_service(
        "chat-gpt",
        sk_oai.AzureChatCompletion(chatgpt_config['deployment'], 
                                    chatgpt_config['endpoint'], 
                                    chatgpt_config['api_key'], 
                                    api_version=chatgpt_config['api_version'],
                                    ad_auth=True), 
    )
    return kernel

def import_custom_plugins(kernel, plugins_directory, rag_plugin_name):
    rag_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, rag_plugin_name)
    native_functions = kernel.import_native_skill_from_directory(plugins_directory, rag_plugin_name)
    rag_plugin.update(native_functions)
    return rag_plugin

def create_context(kernel, system_message, ask, messages):
    context = kernel.create_new_context()
    context.variables["bot_description"] = system_message
    context.variables["ask"] = ask
    context.variables["history"] = json.dumps(messages[-5:-1], ensure_ascii=False) # just last two interactions
    return context

def triage_ask(rag_plugin, context):
    """
    This function is used to triage the user ask and determine the intent of the request. 
    If the ask is a Q&A question, a search query is generated to search for sources.
    If it is not a Q&A question, there's no need to retrieve sources and the answer is generated.
   
    Returns:
    dict: A dictionary containing the triage response. The response includes the intent, answer, search query, and a bypass flag.
        'intents' (str): A list of intents of the request. Defaults to ['none'] if not found.
        'answer' (str): The answer to the request. Defaults to an empty string if not found.
        'search_query' (str): The search query for the request. Defaults to an empty string if not found.
        'bypass' (bool): A flag indicating whether to bypass the the reminder flow steps (in case of an error has occurred).
    """    
    triage_response= {"intents":  ["none"], "answer": "", "search_query": "", "bypass": False}
    output_context = call_semantic_function(rag_plugin["Triage"], context)
    if context.error_occurred:
        logging.error(f"[code_orchestration] error when executing RAG flow (Triage). SK error: {context.last_error_description}")
        raise Exception(f"Triage was not successful due to an error when calling semantic function: {context.last_error_descirption}")
    try:
        response = output_context.result.strip("`json\n`")
        response_json = json.loads(response)
    except json.JSONDecodeError:
        logging.error(f"[code_orchestration] error when executing RAG flow (Triage). Invalid json: {output_context.result}")
        raise Exception(f"Triage was not successful due to a JSON error. Invalid json: {output_context.result}")

    triage_response["intents"] = response_json.get('intents', ['none'])
    triage_response["answer"] = response_json.get('answer', '')
    triage_response["search_query"] = response_json.get('query_string', '')   
    return triage_response


async def get_answer(history):

    #############################
    # INITIALIZATION
    #############################

    #initialize variables    

    answer_dict = {}
    answer = ""
    intents = "none"
    system_message = prompt = open(SYSTEM_MESSAGE_PATH, "r").read()
    search_query = ""
    sources = ""
    bypass_nxt_steps = False  # flag to bypass unnecessary steps
    blocked_list = []

    # get user question

    messages = get_chat_history_as_messages(history, include_last_turn=True)
    ask = messages[-1]['content']

    logging.info(f"[code_orchestration] starting RAG flow. {ask[:50]}")
    init_time = time.time()

    #############################
    # GUARDRAILS (QUESTION)
    #############################
    if BLOCKED_LIST_CHECK:
        logging.info(f"[code_orchestration] guardrails - blocked list check.")
        try:
            blocked_list = get_blocked_list()
            for blocked_word in blocked_list:
                if blocked_word in ask.lower().split():
                    logging.info(f"[code_orchestration] blocked word found in question: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    bypass_nxt_steps = True
                    break
        except Exception as e:
            logging.error(f"[code_orchestration] could not get blocked list. {e}") 

    #############################
    # RAG-FLOW
    #############################

    if not bypass_nxt_steps:

        try:
            
            # initialize semantic kernel
            kernel = initialize_kernel()
            rag_plugin = import_custom_plugins(kernel, "orc/plugins", "RAG")
            context = create_context(kernel, system_message, ask, messages)
            conversation_summary_plugin = kernel.import_skill(ConversationSummarySkill(kernel=kernel), skill_name="ConversationSummaryPlugin")
            
            # triage (find intent and generate answer and search query when applicable)
            logging.info(f"[code_orchestration] checking intent. ask: {ask}")
            start_time = time.time()                        
            triage_response = triage_ask(rag_plugin, context)
            response_time =  round(time.time() - start_time,2)
            intents = triage_response['intents']
            logging.info(f"[code_orchestration] checked intents: {intents}. {response_time} seconds.")

            # Handle general intents
            if set(intents).intersection({"about_bot", "off_topic"}):
                answer = triage_response['answer']
                logging.info(f"[code_orchestration] triage answer: {answer}")

            # Handle question answering intent
            elif set(intents).intersection({"follow_up", "question_answering"}):         
    
                search_query = triage_response['search_query'] if triage_response['search_query'] != '' else ask

                output_context = await kernel.run_async(
                    rag_plugin["Retrieval"],
                    input_str=search_query
                )
                sources = output_context.result
                context.variables["sources"] = sources
                logging.info(f"[code_orchestration] generating bot answer. sources: {sources[:200]}")

                # Handle errors
                if context.error_occurred:
                    logging.error(f"[code_orchestration]  error when executing RAG flow (Retrieval). SK error: {context.last_error_description}")
                    answer = f"{get_message('ERROR_ANSWER')} (Retrieval) RAG flow: {context.last_error_description}"
                    bypass_nxt_steps = True

                else:
                    # Generate the answer for the user
                    logging.info(f"[code_orchestration] generating bot answer. ask: {ask}")
                    start_time = time.time()                                                          
                    context.variables["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
                    output_context = call_semantic_function(rag_plugin["Answer"], context)
                    answer = output_context.result
                    if context.error_occurred:
                        logging.error(f"[code_orchestration] error when executing RAG flow (get the answer). {context.last_error_description}")
                        answer = f"{get_message('ERROR_ANSWER')} (get the answer) RAG flow: {context.last_error_description}"
                        bypass_nxt_steps = True
                    response_time =  round(time.time() - start_time,2)              
                    logging.info(f"[code_orchestration] generated bot answer. {answer[:100]}. {response_time} seconds.")

            elif "greeting" in intents:
                answer = triage_response['answer']
                logging.info(f"[code_orchestration] triage answer: {answer}")

            else:
                logging.info(f"[code_orchestration] SK did not executed, no intent found, review Triage function.")
                answer = get_message('NO_INTENT_ANSWER')
                bypass_nxt_steps = True

        except Exception as e:
            logging.error(f"[code_orchestration] exception when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: exception: {e}"
            bypass_nxt_steps = True

    #############################
    # GUARDRAILS (ANSWER)
    #############################

    if GROUNDEDNESS_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
        try:
            logging.info(f"[code_orchestration] checking if it is grounded. answer: {answer[:50]}")  
            logging.info(f"[code_orchestration] checking if it is grounded. sources: {sources[:100]}")
            start_time = time.time()            
            context.variables["answer"] = answer                      
            output_context = call_semantic_function(rag_plugin["IsGrounded"], context)
            grounded = output_context.result
            logging.info(f"[code_orchestration] is it grounded? {grounded}.")  
            if grounded.lower() == 'no':
                logging.info(f"[code_orchestration] ungrounded answer: {answer}")
                output_context = call_semantic_function(rag_plugin["NotInSourcesAnswer"], context)
                answer = output_context.result
                answer_dict['gpt_groundedness'] = 1
                bypass_nxt_steps = True
            else:
                answer_dict['gpt_groundedness'] = 5
            response_time =  round(time.time() - start_time,2)
            logging.info(f"[code_orchestration] checked if it is grounded. {response_time} seconds.")
        except Exception as e:
            logging.error(f"[code_orchestration] could not check answer is grounded. {e}")

    if BLOCKED_LIST_CHECK and not bypass_nxt_steps:
        try:
            for blocked_word in blocked_list:
                if blocked_word in answer.lower().split():
                    logging.info(f"[code_orchestration] blocked word found in answer: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    break
        except Exception as e:
            logging.error(f"[code_orchestration] could not get blocked list. {e}")
            
    answer_dict["prompt"] = prompt
    answer_dict["answer"] = answer
    answer_dict["sources"] = sources.replace('[', '{').replace(']', '}')
    answer_dict["search_query"] = search_query
    answer_dict["model"] = AZURE_OPENAI_CHATGPT_MODEL    
    # answer_dict["prompt_tokens"] = prompt_tokens
    # answer_dict["completion_tokens"] = completion_tokens
        
    response_time =  round(time.time() - init_time,2)
    logging.info(f"[code_orchestration] Finished RAG Flow. {response_time} seconds.")

    return answer_dict
