# imports
import json
import logging
from shared.util import call_semantic_function, get_usage_tokens

async def triage(kernel, conversation_plugin, arguments):
    """
    This function is used to triage the user's request and determine its intent. 
    Depending on the intent, it either generates a search query for sources (if it's a Q&A question) 
    or directly generates an answer (if it's not a Q&A question).

    Args:
        kernel (object): The kernel object.
        conversation_plugin (dict): The conversation plugin dictionary.
        arguments (dict): The arguments for the function.

    Returns:
        dict: A dictionary containing the triage response. The response includes:
            'intents' (list): A list of intents of the request. Defaults to ['none'] if not found.
            'answer' (str): The answer to the request. Defaults to an empty string if not found.
            'search_query' (str): The search query for the request. Defaults to an empty string if not found.
            'prompt_tokens' (list): The prompt tokens from the function result. 
            'completion_tokens' (list): The completion tokens from the function result.
            'bypass' (bool): A flag indicating whether to bypass the reminder flow steps (in case an error has occurred).
            
    Raises:
        Exception: If there's a JSON decoding error when processing the function result.
    """
    triage_dict= {"intents":  ["none"], "answer": "", "search_query": "", "bypass": False}
    function_result =  await call_semantic_function(kernel, conversation_plugin["Triage"], arguments)
    message_content = str(function_result)
    try:
        response = message_content.strip("`json\n`")
        response_json = json.loads(response)
    except json.JSONDecodeError:
        logging.error(f"[code_orchest] error when executing RAG flow (Triage). Invalid json: {function_result.result}")
        raise Exception(f"Triage was not successful due to a JSON error. Invalid json: {function_result.result}")
    intents = response_json.get('intents', ['none'])
    triage_dict["intents"] = intents if intents != [] else ['none']
    triage_dict["answer"] = response_json.get('answer', '')
    triage_dict["search_query"] = response_json.get('query_string', '') 
    triage_dict["prompt_tokens"] = get_usage_tokens(function_result, 'prompt')
    triage_dict["completion_tokens"] = get_usage_tokens(function_result, 'completion')
    return triage_dict