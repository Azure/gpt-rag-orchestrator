import logging
import azure.functions as func
#from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, JSONResponse
import json
import os
from . import orchestrator

from orc.configuration import Configuration
config = Configuration()

LOGLEVEL = config.get_value('LOGLEVEL', 'DEBUG').upper()
if LOGLEVEL == 'DEBUG':
    LOGLEVEL = logging.DEBUG
elif LOGLEVEL == 'INFO':
    LOGLEVEL = logging.INFO
elif LOGLEVEL == 'WARNING':
    LOGLEVEL = logging.WARNING
elif LOGLEVEL == 'ERROR':
    LOGLEVEL = logging.ERROR

logging.basicConfig(level=LOGLEVEL)

authType = config.get_value('FUNC_AUTH_TYPE', 'function').upper()
# Create the Function App with the desired auth level.
app = func.FunctionApp(http_auth_level=authType)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    conversation_id = req_body.get('conversation_id')
    question = req_body.get('question')

    # Get client principal information
    client_principal_id = req_body.get('client_principal_id', '00000000-0000-0000-0000-000000000000')
    client_principal_name = req_body.get('client_principal_name', 'anonymous')
    client_group_names = req_body.get('client_group_names', '')
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name,
        'group_names': client_group_names        
    }

    if question:

        result = await orchestrator.run(conversation_id, question, client_principal)

        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse('{"error": "no question found in json input"}', mimetype="application/json", status_code=200)
