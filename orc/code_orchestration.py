import json
import logging
import os
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

            # load plugins that will be used

            rag_plugin_name = "RAG"
            rag_plugin = kernel.import_semantic_skill_from_directory("orc/plugins", rag_plugin_name)
            native_functions = kernel.import_native_skill_from_directory("orc/plugins", rag_plugin_name)
            rag_plugin.update(native_functions)

            # preparare Chat semantic function

            system_message  = open(SYSTEM_MESSAGE, "r").read()
            prompt = system_message

            prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
                max_tokens=AZURE_OPENAI_RESP_MAX_TOKENS,
                temperature=AZURE_OPENAI_TEMPERATURE,
                top_p=AZURE_OPENAI_TOP_P,
                function_call="auto",
                chat_system_prompt=system_message,
            )

            prompt_template = OpenAIChatPromptTemplate(
                "{{$user_input}}", kernel.prompt_template_engine, prompt_config
            )

            # add message history to prompt template
            for message in messages[:-1]:
                prompt_template.add_message(message['role'], message['content'])

            # register chat function

            function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
            chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)

            # oai function definitions

            filter = {"exclude_skill": ["ChatBot"]}
            functions = get_function_calling_object(kernel, filter)
            
            # create context and execute chat function

            context = kernel.create_new_context()
            context.variables["user_input"] = ask


            logging.info(f"[code_orchestration] calling Chat function.")
            context = await chat_completion_with_function_call(
                    kernel,
                    chat_skill_name="ChatBot",
                    chat_function_name="Chat",
                    context=context,
                    functions=functions,
            )


            # error handling

            if context.error_occurred:
                logging.error(f"[code_orchestration] error when executing RAG flow. {context.last_error_description}")
                answer = f"{get_message('ERROR_ANSWER')} RAG flow: {context.last_error_description}"
                bypass_flow = True
            else:
                answer = context.result

            # when RAG-Retrieval is called we get the search query and sources  
            next_to_last_message = prompt_template.messages[-2]
            if next_to_last_message['role'] == "function" and next_to_last_message['name'] == "RAG-Retrieval":
                # sources
                function_input = json.loads(next_to_last_message['content'])
                if 'sources' in function_input:
                    sources = function_input['sources'] 
                # search query
                function_call_message = prompt_template.messages[-3]
                if 'arguments' in function_call_message['function_call']:
                    arguments = json.loads(function_call_message['function_call']['arguments'])
                    search_query = arguments['input']

                response_time =  round(time.time() - start_time,2)              
                logging.info(f"[code_orchestration] executed RAG flow with SK. {response_time} seconds")

        except Exception as e:
            logging.error(f"[code_orchestration] error when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: {e}"
            bypass_flow = True


    #############################
    # GUARDRAILS (ANSWER)
    #############################

    if GROUNDEDNESS_CHECK and not bypass_flow:
        if sources != "":
            try:
                start_time = time.time()
                groundedness_threshold = 3
                context = kernel.create_new_context()
                context.variables["answer"] = answer
                FUNCTION_PROMPT_SIZE = 1500
                extra_tokens = FUNCTION_PROMPT_SIZE + number_of_tokens(answer, AZURE_OPENAI_CHATGPT_MODEL)  # prompt + answer
                sources = truncate_to_max_tokens(sources, extra_tokens, AZURE_OPENAI_CHATGPT_MODEL)        
                context.variables["sources"] = sources
                logging.info(f"[code_orchestration] checking groundedness. answer: {answer}.")
                logging.info(f"[code_orchestration] checking groundedness. sources: {sources}.")                
                semantic_response = call_semantic_function(rag_plugin["Groundedness"], context)
                response_time =  round(time.time() - start_time,2)              
                if not semantic_response.error_occurred:
                    if semantic_response.result.isdigit():
                        gpt_groundedness = int(semantic_response.result)  
                        logging.info(f"[code_orchestration] checked groundedness: {gpt_groundedness}. {response_time} seconds")
                        if gpt_groundedness < groundedness_threshold: 
                            logging.info(f"[code_orchestration] ungrounded answer: {answer[:50]}.")
                            answer = get_message('UNGROUNDED_ANSWER')
                        answer_dict['gpt_groundedness'] = gpt_groundedness
                    else:
                        logging.error(f"[code_orchestration] could not calculate groundedness.")
                else:
                    logging.error(f"[code_orchestration] could not calculate groundedness. {semantic_response.last_error_description}")
            except Exception as e:
                logging.error(f"[code_orchestration] could not calculate groundedness. {e}")
        else:
            logging.info(f"[code_orchestration] no sources found. Skipping groundedness check and setting groundedness to 0.")
            answer_dict['gpt_groundedness'] = 0

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
