import logging
import azure.functions as func
import json
import os
from . import new_orchestrator
from shared.util import get_user

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    conversation_id = req_body.get('conversation_id')
    question = req_body.get('question')
    client_principal_id = req_body.get('client_principal_id')
    client_principal_name = req_body.get('client_principal_name') 
    url = req_body.get('url', '')
    if not client_principal_id or client_principal_id == '':
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'    
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }

    organization_id = None
    user = get_user(client_principal_id)
    if "data" in user and "organizationId" in user["data"]:
        organization_id = user["data"].get("organizationId")
        logging.info(f"Retrieved organizationId: {organization_id} from user data")

    if question:
        result = await new_orchestrator.stream_run(conversation_id, question, url, client_principal, organization_id)
        result = {
            'conversation_id': conversation_id,
            'question': question,
            'result': "Ok"}

        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse('{"error": "no question found in json input"}', mimetype="application/json", status_code=200)
