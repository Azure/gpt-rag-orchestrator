import logging
import os

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

def get_answer(prompt, history):
    answer = "Prompt flow orchestration is not implemented yet."
    answer_dict = {
        "prompt" : "",
        "answer" : answer,
        "search_query" : "",
        "sources": "",
        "prompt_tokens" : 0,
        "completion_tokens" : 0
    }
    return answer_dict