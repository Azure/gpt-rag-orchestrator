import logging
import azure.functions as func
import json
import os

from shared.util import get_invitation, create_invitation, get_invitations

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get invitations.')
    req_params = None
    if req.method == "GET":
        req_params = req.params
        user_id = req_params.get("user_id")
        organization_id = req_params.get("organizationId")
        if organization_id:
            invitations = get_invitations(organization_id)
            return func.HttpResponse(json.dumps(invitations), mimetype="application/json", status_code=200)
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
        role = req_body.get("role")
        create_invitation(
            invited_user_email, organization_id, role
        )
        return func.HttpResponse(json.dumps(req_body), mimetype="application/json", status_code=200)
    else:
        logging.error('Method not allowed')
        return func.HttpResponse("Method not allowed", status_code=405)
        
    
    
        