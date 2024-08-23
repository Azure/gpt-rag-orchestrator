import logging
import azure.functions as func
import json
import os

from shared.util import get_invitation, create_invitation

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    req_params = None
    
    if req.method == "GET":
        req_params = req.params
        user_id = req_params["user_id"]
        logging.info('Python HTTP trigger function processed a request to get settings.')
        invitation = get_invitation(user_id)
        return func.HttpResponse(json.dumps(invitation), mimetype="application/json", status_code=200)
    if req.method == "POST":
        try:
            req_body = json.loads(req.get_body())
        except:
            return func.HttpResponse("Invalid request body", status_code=400)
        
        logging.info('Python HTTP trigger function processed a request to set settings. request body: %s', req.get_json())
        invited_user_email = req_body.get("invited_user_email")
        organization_id = req_body.get("organization_id")
        create_invitation(
            invited_user_email,organization_id
        )
        return func.HttpResponse(json.dumps(req_body), mimetype="application/json", status_code=200)
    else:
        logging.error('Method not allowed')
        return func.HttpResponse("Method not allowed", status_code=405)
        
    
    
        