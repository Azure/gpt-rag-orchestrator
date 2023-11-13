import json
import logging
import os
import re
import time
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message
from shared.util import truncate_to_max_tokens, number_of_tokens, get_aoai_config, get_blocked_list

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
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

SYSTEM_MESSAGE = f"orc/prompts/system_message.prompt"

async def get_answer(history):

    #############################
    # INITIALIZATION
    #############################

    #initialize variables    

    answer_dict = {}
    answer = ""
    intent = "none"
    prompt = ""
    search_query = ""
    sources = ""
    bypass_flow = False  
    blocked_list = []

    # get user question

    messages = get_chat_history_as_messages(history, include_last_turn=True)
    ask = messages[-1]['content']

    #############################
    # GUARDRAILS (QUESTION)
    #############################
    if BLOCKED_LIST_CHECK:
        logging.info(f"[code_orchestration] guardrails - blocked list check.")
        try:
            blocked_list = get_blocked_list()
            for blocked_word in blocked_list:
                if blocked_word in ask.lower():
                    logging.info(f"[code_orchestration] blocked word found in question: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    bypass_flow = True
                    break
        except Exception as e:
            logging.error(f"[code_orchestration] could not get blocked list. {e}") 

    #############################
    # RAG-FLOW
    #############################

    if not bypass_flow:

        try:
            
            # initialize semantic kernel

            logging.info(f"[code_orchestration] starting RAG flow. {ask[:50]}")

            start_time = time.time()

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

            rag_plugin_name = "RAG"
            plugins_directory = "orc/plugins"
            rag_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, rag_plugin_name)
            native_functions = kernel.import_native_skill_from_directory(plugins_directory, rag_plugin_name)
            rag_plugin.update(native_functions)

            system_message = open(SYSTEM_MESSAGE, "r").read()

            context = kernel.create_new_context()
            context.variables["bot_description"] = system_message
            context.variables["ask"] = ask

            # triage

            logging.info(f"[code_orchestration] checking intent. ask: {ask}")            
            sk_response = call_semantic_function(rag_plugin["Triage"], context)
            sk_response_json = json.loads(sk_response.result)
            if 'intent' in sk_response_json:
                intent = sk_response_json['intent']            
                logging.info(f"[code_orchestration] triaging with SK intent: {intent}.")  
            response_time =  round(time.time() - start_time,2)              
            logging.info(f"[code_orchestration] triaging with SK. {response_time} seconds.")

            # greetings

            if intent == "greeting" or intent == "about_bot":
                if 'answer' in sk_response_json:
                    answer = sk_response_json['answer']
                    logging.info(f"[code_orchestration] triaging with SK answer: {answer}.")

            # qna

            elif intent == "question_answering":
                # get sources

                sk_response = await kernel.run_async(
                    rag_plugin["Retrieval"],
                    input_str=ask,
                )
                search_query = ask
                sources = sk_response.result
                context.variables["sources"] = sources

                # get the answer

                logging.info(f"[code_orchestration] generating bot answer. ask: {ask}")
                context.variables["history"] = json.dumps(messages[:-1], ensure_ascii=False)
                sk_response = call_semantic_function(rag_plugin["Answer"], context)
                answer = sk_response.result
                logging.info(f"[code_orchestration] generating bot answer. answer: {answer}")  
                response_time =  round(time.time() - start_time,2)              
                logging.info(f"[code_orchestration] triaging with SK. {response_time} seconds.")

                if context.error_occurred:
                    logging.error(f"[code_orchestration] error when executing RAG flow. {context.last_error_description}")
                    answer = f"{get_message('ERROR_ANSWER')} RAG flow: {context.last_error_description}"
                    bypass_flow = True

                response_time =  round(time.time() - start_time,2)              
                logging.info(f"[code_orchestration] executed RAG flow with SK. {response_time} seconds.")
            else:
                logging.info(f"[code_orchestration] SK did not executed, no intent found.")

        except Exception as e:
            logging.error(f"[code_orchestration] error when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: {e}"
            bypass_flow = True


    #############################
    # GUARDRAILS (ANSWER)
    #############################

    if GROUNDEDNESS_CHECK and intent == 'question_answering' and not bypass_flow:
        try:
            logging.info(f"[code_orchestration] checking if it is grounded. answer: {answer[:50]}")  
            logging.info(f"[code_orchestration] checking if it is grounded. sources: {sources[:100]}")
            context.variables["answer"] = answer                      
            sk_response = call_semantic_function(rag_plugin["Grounded"], context)
            grounded = sk_response.result
            logging.info(f"[code_orchestration] is it grounded? {grounded}.")  
            if grounded.lower() == 'no':
                logging.info(f"[code_orchestration] ungrounded answer: {answer}.")
                answer = get_message('UNGROUNDED_ANSWER')
                answer_dict['gpt_groundedness'] = 1
                bypass_flow = True
            else:
                answer_dict['gpt_groundedness'] = 5
            response_time =  round(time.time() - start_time,2)
            logging.info(f"[code_orchestration] checking if it is grounded with SK. {response_time} seconds.")
        except Exception as e:
            logging.error(f"[code_orchestration] could not check answer is grounded. {e}")

    if BLOCKED_LIST_CHECK and not bypass_flow:
        try:
            for blocked_word in blocked_list:
                if blocked_word in answer.lower():
                    logging.info(f"[code_orchestration] blocked word found in answer: {blocked_word}.")
                    answer = get_message('BLOCKED_ANSWER')
                    break
        except Exception as e:
            logging.error(f"[code_orchestration] could not get blocked list. {e}")
            
    answer_dict["prompt"] = prompt
    answer_dict["answer"] = answer
    answer_dict["sources"] = sources
    answer_dict["search_query"] = search_query
    answer_dict["model"] = AZURE_OPENAI_CHATGPT_MODEL    
    # answer_dict["prompt_tokens"] = prompt_tokens
    # answer_dict["completion_tokens"] = completion_tokens    
    
    return answer_dict
