import json
import logging
import os
import requests
import time

# logging level
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

# TODO
def get_answer(prompt, history):
    answer = ""
    answer_dict = {
        "prompt" : "",
        "answer" : answer,
        "search_query" : "",
        "sources": "",
        "prompt_tokens" : 0,
        "completion_tokens" : 0
    }
    return answer_dict