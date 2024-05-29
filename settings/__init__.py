import logging
import azure.functions as func
import json
import os

from shared.util import get_settings, set_settings, get_setting

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    
    if req.method == "POST":
        try:
            req_params = json.loads(req.get_body())
        except:
            return func.HttpResponse("Invalid request body", status_code=400)
    if req.method == "GET":
        req_params = req.params

    client_principal_id = req_params.get('client_principal_id')
    client_principal_name = req_params.get('client_principal_name')
    
    if not client_principal_id or client_principal_id == '':
        logging.info('client_principal_id not set, setting to anonymous')
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'    
    
    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }

    if req.method == "POST":
        try:
            req_body = json.loads(req.get_body())
        except:
            return func.HttpResponse("Invalid request body", status_code=400)
        
        logging.info('Python HTTP trigger function processed a request to set settings. request body: %s', req.get_json())
        
        set_settings(
            client_principal=client_principal,
            temperature=req_body.get('temperature', 0.0),
            frequency_penalty=req_body.get('frequency_penalty', 0.0),
            presence_penalty=req_body.get('presence_penalty', 0.0)
        )

        return func.HttpResponse(json.dumps(req_body), mimetype="application/json", status_code=200)
    else:
        logging.info('Python HTTP trigger function processed a request to get settings.')
        settings = get_setting(client_principal)
        return func.HttpResponse(json.dumps(settings), mimetype="application/json", status_code=200)
        