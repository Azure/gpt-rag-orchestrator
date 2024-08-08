import logging
import azure.functions as func
import json
import os

from shared.util import update_organization_subscription, get_organization

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    if req.method == "GET":
        req_params = req.params
        organizationId = req_params["organizationId"]
        organization = get_organization(organizationId)
        return func.HttpResponse(
            json.dumps(organization), mimetype="application/json", status_code=200)
        
    
    if req.method == "POST":
        req_body = req.get_json()
        user_id = req_body.get("id")
        organizationId = req_body.get("organizationId")
        suscriptionId = req_body.get("subscriptionId")
        sessionId = req_body.get("sessionId")
        paymentStatus = req_body.get("paymentStatus")
        organizationName = req_body.get("organizationName")
        expirationDate = req_body.get("expirationDate")
        user = update_organization_subscription(user_id, organizationId, suscriptionId, sessionId, paymentStatus, organizationName, expirationDate)
        return func.HttpResponse(
            json.dumps(user), mimetype="application/json", status_code=200)
        
    else:
        return func.HttpResponse("Method not allowed", status_code=405)
    
    
