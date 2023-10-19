import json
import logging
import os
import re
import semantic_kernel as sk
from shared.util import call_semantic_function, chat_complete, get_chat_history_as_messages, get_message
from shared.util import truncate_to_max_tokens, number_of_tokens, load_sk_plugin, get_aoai_config
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# Configurations

FUNCTIONS_CONFIGURATION = f"orc/plugins/functions.json"
QUESTION_ANSWERING_PROMPT_FILE = f"orc/prompts/question_answering.code.prompt"
QUALITY_CONTROL_STEP = os.environ.get("QUALITY_CONTROL_STEP") or "true"
QUALITY_CONTROL_STEP = True if QUALITY_CONTROL_STEP.lower() == "true" else False
AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")


def get_answer(history):

    #############################
    # INITIALIZATION
    #############################

    #initialize variables
    
    prompt = ""
    answer = ""
    prompt_tokens = 0
    completion_tokens = 0
    answer_dict = {}
    answer_dict["search_query"] = ''
    answer_dict["sources"] = ''
    rag_processing_error = False

    # azure openai function preparation

    # load function descriptions
    with open(FUNCTIONS_CONFIGURATION, 'r') as f:
        functions_definitions = json.load(f)
    
    # define functions
    def get_sources(): 
        chatgpt_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
        rag_plugin = load_sk_plugin('RAG', chatgpt_config)
        return rag_plugin["Retrieval"]

    # map functions to code
    function_dict = {
        "get_sources": get_sources,
    }

    # prepare messages history

    messages = []
    chat_history_messages = get_chat_history_as_messages(history, include_last_turn=True)
    prompt  = open(QUESTION_ANSWERING_PROMPT_FILE, "r").read()             
    messages = [{"role": "system", "content": prompt}] + chat_history_messages

    #############################
    # RAG FLOW
    #############################

    try:

        # first call to the model to see if a function call is needed
        response = chat_complete(messages, functions=functions_definitions, function_call="auto")

        if 'error' in response:
            e = response['error']
            logging.error(f"[code_orchestration] error when executing RAG flow. {e}")
            answer = f"{get_message('ERROR_ANSWER')} RAG flow: {e}"
            rag_processing_error = True
        
        else:
            response_message = response['choices'][0]['message']
            prompt_tokens += response['usage']['prompt_tokens']
            completion_tokens += response['usage']['completion_tokens']

            if 'function_call' in response_message:
                function_name = response_message["function_call"]["name"]
                function_args = json.loads(response_message["function_call"]["arguments"])
                function_to_call = function_dict[function_name] 
                context_variables = sk.ContextVariables()
                context_variables.update(function_args)
                function_response = function_to_call(context_variables.variables)

                # add the function results to the messages history giving context to the model
                messages = messages + [
                    {
                        "role": response_message["role"],
                        "function_call": {
                            "name": function_name,
                            "arguments": response_message["function_call"]["arguments"],
                        },
                        "content": None
                    },
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response.result,
                    }
                ]                    

                # generates the answer after adding the function results to the context
                response = chat_complete(messages, functions=functions_definitions, function_call="none")

                if 'error' in response:
                    e = response['error']
                    logging.error(f"[code_orchestration] error when executing RAG flow. {e}")
                    answer = f"{get_message('ERROR_ANSWER')} RAG flow: {e}"
                    rag_processing_error = True
                else:
                    response_message = response['choices'][0]['message']
                    prompt_tokens += response['usage']['prompt_tokens']
                    completion_tokens += response['usage']['completion_tokens']
                    
                    # add the assistant answer
                    messages.append( # adding assistant response to messages
                        {
                            "role": response_message["role"],
                            "content": response_message["content"]
                        }
                    )

                    # answer generated by the model for the user question
                    answer = messages[-1]['content']

                    # store answer metadata when calling get_sources function
                    if function_name == "get_sources":
                        answer_dict["search_query"] = function_args['question']     
                        answer_dict["sources"] = json.loads(function_response.result)['sources'] 

            else:
                
                # add the assistant answer
                messages.append( # adding assistant response to messages
                    {
                        "role": response_message["role"],
                        "content": response_message["content"]
                    }
                )  
                # answer generated by the model for the user question
                answer = messages[-1]['content']

    except Exception as e:
        logging.error(f"[code_orchestration] error when executing RAG flow. {e}")
        answer = f"{get_message('ERROR_ANSWER')} RAG flow: {e}"
        rag_processing_error = True

 
    #############################
    # QUALITY CONTROL STEP
    #############################
 
    if QUALITY_CONTROL_STEP and not rag_processing_error:
        # groundedness needs to be equal or greater than 3 to be considered a good answer
        next_to_last = messages[-2]
        if next_to_last['role'] == 'function' and next_to_last['name'] == 'get_sources':
            try:
                # call semantic function to calculate groundedness
                kernel = sk.Kernel()
                chatgpt_config = get_aoai_config(AZURE_OPENAI_CHATGPT_MODEL)
                kernel.add_chat_service("chat_completion", AzureChatCompletion(chatgpt_config['deployment'], chatgpt_config['endpoint'], chatgpt_config['api_key']))
                rag_plugin = load_sk_plugin('RAG', chatgpt_config)
                context = kernel.create_new_context()
                context['answer'] = re.sub(r'\[.*?\]', '', answer)

                # truncate sources to not hit model max token
                sources = json.loads(next_to_last['content'])['sources']
                extra_tokens = 1500 + number_of_tokens(answer, AZURE_OPENAI_CHATGPT_MODEL)  # prompt + answer
                sources = truncate_to_max_tokens(sources, extra_tokens, AZURE_OPENAI_CHATGPT_MODEL) 
                context['sources'] = sources
                semantic_response = call_semantic_function(rag_plugin["Groundedness"], context)
                if not semantic_response.error_occurred:
                    if semantic_response.result.isdigit():
                        gpt_groundedness = int(semantic_response.result)  
                        logging.info(f"[code_orchestration] groundedness: {gpt_groundedness}.")
                        if gpt_groundedness < 3: 
                            logging.info(f"[code_orchestration] ungrounded answer: {answer}.")
                            answer = get_message('UNGROUNDED_ANSWER')
                        answer_dict['gpt_groundedness'] = gpt_groundedness
                    else:
                        logging.error(f"[code_orchestration] could not calculate groundedness.")
                else:
                    logging.error(f"[code_orchestration] could not calculate groundedness. {semantic_response.last_error_description}")
            except Exception as e:
                logging.error(f"[code_orchestration] could not calculate groundedness. {e}")

    answer_dict["prompt"] = prompt
    answer_dict["answer"] = answer
    answer_dict["model"] = answer    
    answer_dict["prompt_tokens"] = prompt_tokens
    answer_dict["completion_tokens"] = completion_tokens

    return answer_dict