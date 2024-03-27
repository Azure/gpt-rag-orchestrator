import logging
import azure.functions as func
import json
import os

from shared.util import get_conversations, get_conversation

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    id = req.params.get('id')
    if not id:
        conversations = get_conversations()
    else:
        conversations = get_conversation(id)
    return func.HttpResponse(json.dumps(conversations), mimetype="application/json", status_code=200)