# imports
import json
import logging
import os
import semantic_kernel as sk
import time
from orc.plugins.Conversation.BingSearch import BingConnector
from orc.plugins.Conversation.Triage.wrapper import triage
from orc.plugins.ResponsibleAI.Fairness.wrapper import fairness
from semantic_kernel.functions.kernel_arguments import KernelArguments
from shared.util import call_semantic_function, get_chat_history_as_messages, get_message, get_last_messages
from shared.util import get_blocked_list, create_kernel, get_usage_tokens, escape_xml_characters

# logging level

logging.getLogger('azure').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'debug').upper()
logging.basicConfig(level=LOGLEVEL)
myLogger = logging.getLogger(__name__)

# Env Variables

BLOCKED_LIST_CHECK = os.environ.get("BLOCKED_LIST_CHECK") or "true"
BLOCKED_LIST_CHECK = True if BLOCKED_LIST_CHECK.lower() == "true" else False

# Cat Test - Remove intents
GROUNDEDNESS_CHECK = os.environ.get("GROUNDEDNESS_CHECK") or "true"
GROUNDEDNESS_CHECK = True if GROUNDEDNESS_CHECK.lower() == "true" else False
RESPONSIBLE_AI_CHECK = os.environ.get("RESPONSIBLE_AI_CHECK") or "true"
RESPONSIBLE_AI_CHECK = True if RESPONSIBLE_AI_CHECK.lower() == "true" else False
CONVERSATION_METADATA = os.environ.get("CONVERSATION_METADATA") or "true"
CONVERSATION_METADATA = True if CONVERSATION_METADATA.lower() == "true" else False

AZURE_OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_TEMPERATURE = float(AZURE_OPENAI_TEMPERATURE)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_TOP_P = float(AZURE_OPENAI_TOP_P)
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"

AZURE_OPENAI_RESP_MAX_TOKENS = int(AZURE_OPENAI_RESP_MAX_TOKENS)
CONVERSATION_MAX_HISTORY = os.environ.get("CONVERSATION_MAX_HISTORY") or "3"
CONVERSATION_MAX_HISTORY = int(CONVERSATION_MAX_HISTORY)

ORCHESTRATOR_FOLDER = "orc"
PLUGINS_FOLDER = f"{ORCHESTRATOR_FOLDER}/plugins"
BOT_DESCRIPTION_FILE = f"{ORCHESTRATOR_FOLDER}/bot_description.prompt"


async def get_answer(history):


    #############################
    # INITIALIZATION
    #############################

    #initialize variables    

    answer_dict = {}

    # prompt = "The prompt is only recorded for question-answering intents"

    answer = ""
    # Cat Test - Remove intents 
    # intents = []
    bot_description = open(BOT_DESCRIPTION_FILE, "r").read()
    
    # Cat Test - Remove intents
    # search_query = ""

    sources = ""
    # Cat Test - Remove Intents (Added to Test)
    prompt = ""
    
    # Cat Test - Remove intents
    # bypass_nxt_steps = False  # flag to bypass unnecessary steps
    # blocked_list = []

    # conversation metadata

    # Cat Test - Remove intents
    # conversation_plugin_answer = ""
    # conversation_history_summary = ''
    # triage_language = ''
    # answer_generated_by = "none"
    # prompt_tokens = 0
    # completion_tokens = 0
    
    # get user question

    messages = get_chat_history_as_messages(history, include_last_turn=True)
    
    # Cat Test - Remove inents to Initialize ask (user's question) after ensuring messages contain valid content
    if not messages or len(messages) == 0:
        raise ValueError("Conversation history is empty or invalid. Cannot proceed.")

    
    
    ask = messages[-1]['content']
    # Cat Test - Remove intents - Added the below if
    if not ask:
        raise ValueError("User question is empty. Cannot proceed.")

    logging.info(f"[code_orchest] starting RAG flow. {ask[:50]}")
    init_time = time.time()

    # Cat Test - Remove intents to Assign the search query from the ask
    search_query = ask.strip()
    if not search_query:
        raise ValueError("Search query is empty. Cannot proceed with embedding generation.")
    

    #############################
    # GUARDRAILS (QUESTION)
    #############################
    
    # Cat Test - Remove intents
    # if BLOCKED_LIST_CHECK:
    #     logging.debug(f"[code_orchest] blocked list check.")
    #     try:
    #         blocked_list = get_blocked_list()
    #         for blocked_word in blocked_list:
    #             if blocked_word in ask.lower().split():
    #                 logging.info(f"[code_orchest] blocked word found in question: {blocked_word}.")
    #                 answer = get_message('BLOCKED_ANSWER')
    #                 answer_generated_by = 'blocked_list_check'
    #                 bypass_nxt_steps = True
    #                 break
    #     except Exception as e:
    #         logging.error(f"[code_orchest] could not get blocked list. {e}")
    #     response_time =  round(time.time() - init_time,2)
    #     logging.info(f"[code_orchest] finished blocked list check. {response_time} seconds.")            

    #############################
    # RAG-FLOW
    #############################

# Cat Test - Remove intents
    # if not bypass_nxt_steps:

    try:
            
            # create kernel
            kernel = create_kernel()

            # create the arguments that will used by semantic functions

            # Cat Test - Remove intents
            # arguments = KernelArguments()
            # Cat Test - Remove intents(commented above and added the below)
            arguments = sk.KernelArguments()

            arguments["bot_description"] = bot_description
            arguments["ask"] = ask
            
            # arguments["history"] = json.dumps(get_last_messages(messages, CONVERSATION_MAX_HISTORY), ensure_ascii=False)
            # Cat Test - Remove intents(commented above and added the below)
            arguments["history"] = json.dumps(get_last_messages(messages, 3), ensure_ascii=False)
            
            arguments["previous_answer"] = messages[-2]['content'] if len(messages) > 1 else ""

            # import RAG plugins

            # Cat Test - Remove intents
            # conversationPlugin = kernel.import_plugin_from_prompt_directory(PLUGINS_FOLDER, "Conversation")
            retrievalPlugin = kernel.import_native_plugin_from_directory(PLUGINS_FOLDER, "Retrieval")

            # Cat Test - Remove intents - Added these to test
            arguments["language"] = arguments.get("language", "en")  # Default language is English
            arguments["conversation_summary"] = arguments.get("conversation_summary", "")


            # conversation summary
            # Cat Test - Remove intents
            # logging.debug(f"[code_orchest] summarizing conversation")
            # start_time = time.time()
            # if arguments["history"] != '[]':
            #     function_result = await call_semantic_function(kernel, conversationPlugin["ConversationSummary"], arguments)
            #     prompt_tokens += get_usage_tokens(function_result, 'prompt')
            #     completion_tokens += get_usage_tokens(function_result, 'completion')            
            #     conversation_history_summary =  str(function_result)
            # else:
            #     conversation_history_summary = ""
            #     logging.info(f"[code_orchest] first time talking no need to summarize.")
            # arguments["conversation_summary"] = conversation_history_summary
            # response_time = round(time.time() - start_time,2)
            # logging.info(f"[code_orchest] finished summarizing conversation: {conversation_history_summary}. {response_time} seconds.")

            # triage (find intent and generate answer and search query when applicable)
            # Cat Test - Remove intents
            # logging.debug(f"[code_orchest] checking intent. ask: {ask}")
            # start_time = time.time()
            # triage_dict = await triage(kernel, conversationPlugin, arguments)
            # intents = triage_dict['intents']
            # Cat Test - Remove intents
            # triage_language = triage_dict['language']
            # arguments["language"] = triage_language
            # prompt_tokens += triage_dict["prompt_tokens"]
            # completion_tokens += triage_dict["completion_tokens"]
            # response_time = round(time.time() - start_time,2)
            # logging.info(f"[code_orchest] finished checking intents: {intents}. {response_time} seconds.")

            # Handle general intents

            # Cat Test - Remove intents
            # if set(intents).intersection({"about_bot", "off_topic"}):
            #     answer = triage_dict['answer']
            #     answer_generated_by = "conversation_plugin_triage"
            #     logging.info(f"[code_orchest] triage answer: {answer}")

            # Handle question answering intent

            # Cat Test - Remove intents
            # elif set(intents).intersection({"follow_up", "question_answering" }):         
    
            #     search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask

                # run retrieval function
                # Cat Test - Remove intents
                # function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], sk.KernelArguments(input=search_query))
                # sources = function_result.value
                # formatted_sources = sources[:100].replace('\n', ' ')
                # escaped_sources = escape_xml_characters(sources)
                # arguments["sources"] = escaped_sources
                # logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")
            
                # Generate the answer augmented by the retrieval
                # Cat Test - Remove intents
            #     logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
            #     start_time = time.time()                                                          
            #     arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
            #     function_result = await call_semantic_function(kernel, conversationPlugin["Answer"], arguments)
            #     answer =  str(function_result)
            #     conversation_plugin_answer = answer
            #     answer_generated_by = "conversation_plugin_answer"
            #     prompt_tokens += get_usage_tokens(function_result, 'prompt')
            #     completion_tokens += get_usage_tokens(function_result, 'completion')
            #     prompt = str(function_result.metadata['messages'][0])
            #     response_time =  round(time.time() - start_time,2)              
            #     logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")

            # elif "greeting" in intents:
            #     answer = triage_dict['answer']
            #     answer_generated_by = "conversation_plugin_triage"
            #     logging.info(f"[code_orchest] triage answer: {answer}")
            
            # elif "price_range" in intents:
            #     search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask

                # run retrieval function
                # Cat Test - Remove intents
                # function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], sk.KernelArguments(input=search_query))
                # sources = function_result.value
                # formatted_sources = sources[:100].replace('\n', ' ')
                # escaped_sources = escape_xml_characters(sources)
                # arguments["sources"] = escaped_sources
                # logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")
            
                # Generate the answer augmented by the retrieval
                # Cat Test - Remove intents
            #     logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
            #     start_time = time.time()                                                          
            #     arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
            #     function_result = await call_semantic_function(kernel, conversationPlugin["PriceRange"], arguments)
            #     answer =  str(function_result)
            #     conversation_plugin_answer = answer
            #     answer_generated_by = "conversation_plugin_answer"
            #     prompt_tokens += get_usage_tokens(function_result, 'prompt')
            #     completion_tokens += get_usage_tokens(function_result, 'completion')
            #     prompt = str(function_result.metadata['messages'][0])
            #     response_time =  round(time.time() - start_time,2)              
            #     logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")

            # elif "purchase_order" in intents:
            #     search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask

                # run retrieval function
                # Cat Test - Remove intents
                # function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], sk.KernelArguments(input=search_query))
                # sources = function_result.value
                # formatted_sources = sources[:100].replace('\n', ' ')
                # escaped_sources = escape_xml_characters(sources)
                # arguments["sources"] = escaped_sources
                # logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")
            
                # Generate the answer augmented by the retrieval
                # Cat Test - Remove intents
            #     logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
            #     start_time = time.time()                                                          
            #     arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
            #     function_result = await call_semantic_function(kernel, conversationPlugin["PurchaseOrder"], arguments)
            #     answer =  str(function_result)
            #     conversation_plugin_answer = answer
            #     answer_generated_by = "conversation_plugin_answer"
            #     prompt_tokens += get_usage_tokens(function_result, 'prompt')
            #     completion_tokens += get_usage_tokens(function_result, 'completion')
            #     prompt = str(function_result.metadata['messages'][0])
            #     response_time =  round(time.time() - start_time,2)              
            #     logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")        

            # elif "specifications" in intents:
            #     search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask

                # run retrieval function
                # Cat Test - Remove intents
                # function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], sk.KernelArguments(input=search_query))
                # sources = function_result.value
                # formatted_sources = sources[:100].replace('\n', ' ')
                # escaped_sources = escape_xml_characters(sources)
                # arguments["sources"] = escaped_sources
                # logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")
            
                # Generate the answer augmented by the retrieval
                # Cat Test - Remove intents
            #     logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
            #     start_time = time.time()                                                          
            #     arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
            #     function_result = await call_semantic_function(kernel, conversationPlugin["Specifications"], arguments)
            #     answer =  str(function_result)
            #     conversation_plugin_answer = answer
            #     answer_generated_by = "conversation_plugin_answer"
            #     prompt_tokens += get_usage_tokens(function_result, 'prompt')
            #     completion_tokens += get_usage_tokens(function_result, 'completion')
            #     prompt = str(function_result.metadata['messages'][0])
            #     response_time =  round(time.time() - start_time,2)              
            #     logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")        

            # elif "ice_document_generation" in intents:
            #     search_query = triage_dict['search_query'] if triage_dict['search_query'] != '' else ask

            
            
                # run retrieval function
            function_result = await kernel.invoke(retrievalPlugin["VectorIndexRetrieval"], sk.KernelArguments(input=search_query))
            sources = function_result.value
            formatted_sources = sources[:100].replace('\n', ' ')
            escaped_sources = escape_xml_characters(sources)
            arguments["sources"] = escaped_sources
            logging.info(f"[code_orchest] generating bot answer. sources: {formatted_sources}")
            
            # Generate the answer augmented by the retrieval
            logging.info(f"[code_orchest] generating bot answer. ask: {ask}")
            start_time = time.time()                                                          
            arguments["history"] = json.dumps(messages[:-1], ensure_ascii=False) # update context with full history
                # function_result = await call_semantic_function(kernel, conversationPlugin["GenerateIceDocument"], arguments)
            # Cat Test - Remove intents (commented the above to add below) 
            function_result = await call_semantic_function(kernel, kernel.import_plugin_from_prompt_directory(PLUGINS_FOLDER, "Conversation")["Answer"], arguments)
                
            answer =  str(function_result)
                # Cat Test - Remove intents
                # conversation_plugin_answer = answer
                # answer_generated_by = "conversation_plugin_answer"
                
                # prompt_tokens += get_usage_tokens(function_result, 'prompt')
                # completion_tokens += get_usage_tokens(function_result, 'completion')
            # Cat Test - Remove intents (commented the above to add below)
            completion_tokens = function_result.metadata.get('usage', {}).get('completion_tokens', 0)
            prompt = getattr(function_result.metadata['messages'][0], 'content', 'No prompt available')

        
            # prompt = str(function_result.metadata['messages'][0])
      


            response_time =  round(time.time() - start_time,2)              
            logging.info(f"[code_orchest] finished generating bot answer. {response_time} seconds. {answer[:100]}.")

            # Cat Test - Remove intents
            # elif intents == ["none"]:
            #     logging.info(f"[code_orchest] No intent found, review Triage function.")
            #     answer = get_message('NO_INTENT_ANSWER')
            #     answer_generated_by = "no_intent_found_check"                
            #     bypass_nxt_steps = True

    except Exception as e:
        logging.error(f"[code_orchest] exception when executing RAG flow. {e}")
        answer = f"{get_message('ERROR_ANSWER')} RAG flow: exception: {e}"
            # Cat Test - Remove intents
            # answer_generated_by = "exception_rag_flow"
            # bypass_nxt_steps = True


    #############################
    # GUARDRAILS (ANSWER)
    #############################

    # if BLOCKED_LIST_CHECK and not bypass_nxt_steps:
    #     try:
    #         for blocked_word in blocked_list:
    #             if blocked_word in answer.lower().split():
    #                 logging.info(f"[code_orchest] blocked word found in answer: {blocked_word}.")
    #                 answer = get_message('BLOCKED_ANSWER')
    #                 answer_generated_by = "blocked_word_check"
    #                 break
    #     except Exception as e:
    #         logging.error(f"[code_orchest] could not get blocked list. {e}")

    # if GROUNDEDNESS_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
    #         try:
    #             logging.info(f"[code_orchest] checking if it is grounded. answer: {answer[:50]}")
    #             start_time = time.time()            
    #             arguments["answer"] = answer                      
    #             function_result = await call_semantic_function(kernel, conversationPlugin["IsGrounded"], arguments)
    #             grounded =  str(function_result)
    #             prompt_tokens += get_usage_tokens(function_result, 'prompt')
    #             completion_tokens += get_usage_tokens(function_result, 'completion')            
    #             logging.info(f"[code_orchest] is it grounded? {grounded}.")  
    #             if grounded.lower() == 'no':
    #                 logging.info(f"[code_orchest] ungrounded answer: {answer}")
    #                 function_result = await call_semantic_function(kernel, conversationPlugin["NotInSourcesAnswer"], arguments)
    #                 prompt_tokens += get_usage_tokens(function_result, 'prompt')
    #                 completion_tokens += get_usage_tokens(function_result, 'completion')            
    #                 answer =  str(function_result)
    #                 answer_dict['gpt_groundedness'] = 1
    #                 answer_generated_by = "gpt_groundedness_check"
    #                 bypass_nxt_steps = True
    #             else:
    #                 answer_dict['gpt_groundedness'] = 5
    #             response_time =  round(time.time() - start_time,2)
    #             logging.info(f"[code_orchest] finished checking if it is grounded. {response_time} seconds.")
    #         except Exception as e:
    #             logging.error(f"[code_orchest] could not check answer is grounded. {e}")            

    # if RESPONSIBLE_AI_CHECK and set(intents).intersection({"follow_up", "question_answering"}) and not bypass_nxt_steps:
    #         try:
    #             logging.info(f"[code_orchest] checking responsible AI (fairness). answer: {answer[:50]}")
    #             start_time = time.time()            
    #             arguments["answer"] = answer
    #             raiPlugin = kernel.import_plugin_from_prompt_directory(PLUGINS_FOLDER, "ResponsibleAI")
    #             fairness_dict = await fairness(kernel, raiPlugin, arguments)
    #             fair = fairness_dict['fair']
    #             fairness_answer = fairness_dict['answer']
    #             prompt_tokens += fairness_dict["prompt_tokens"]
    #             completion_tokens += fairness_dict["completion_tokens"]
    #             logging.info(f"[code_orchest] responsible ai check. Is it fair? {fair}.")
    #             if not fair:
    #                 answer = fairness_answer
    #                 answer_generated_by = "rai_plugin_fairness"
    #             answer_dict['pass_rai_fairness_check'] = fair
    #             response_time =  round(time.time() - start_time,2)
    #             logging.info(f"[code_orchest] finished checking responsible AI (fairness). {response_time} seconds.")
    #         except Exception as e:
    #             logging.error(f"[code_orchest] could not check responsible AI (fairness). {e}")
# Cat Test - Remove intents - The Above GUARDRAILS (Answer) was already commented

    answer_dict["user_ask"] = ask
    answer_dict["answer"] = answer
    answer_dict["search_query"] = search_query

    # additional metadata for debugging
    # Cat Test - Remove intents
    # if CONVERSATION_METADATA:
    #     answer_dict["intents"] = intents
    #     answer_dict["triage_language"] = triage_language     
    #     answer_dict["answer_generated_by"] = answer_generated_by
    #     answer_dict["conversation_history_summary"] = conversation_history_summary
    #     answer_dict["conversation_plugin_answer"] = conversation_plugin_answer
    #     answer_dict["model"] = AZURE_OPENAI_CHATGPT_MODEL
    #     answer_dict["prompt_tokens"] = prompt_tokens
    #     answer_dict["completion_tokens"] = completion_tokens


    # answer_dict["prompt"] = prompt
    answer_dict["sources"] = sources.replace('[', '{').replace(']', '}')

    response_time =  round(time.time() - init_time,2)
    logging.info(f"[code_orchest] finished RAG Flow. {response_time} seconds.")

    return answer_dict