import logging
import azure.functions as func
import json
import os

from shared.util import get_feedback, get_feedback_all, set_feedback

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request for feedback.')

    conversations = None
    client_principal_id = None
    client_principal_name = None
    conversation_id = None
    feedback_message = None
    question = None
    answer = None
    rating = None
    category = None

    req_params = None

    if req.method == "POST":
        try:
            req_params = json.loads(req.get_body())
        except:
            return func.HttpResponse("Invalid request body", status_code=400)
    if req.method == "GET":
        req_params = req.params
    
    client_principal_id   = req_params.get('client_principal_id')
    client_principal_name = req_params.get('client_principal_name')
    conversation_id       = req_params.get('conversation_id')
    conversation_id       = req_params.get('conversation_id')
    feedback_message      = req_params.get('feedback')
    question              = req_params.get('question')
    answer                = req_params.get('answer')
    rating                = req_params.get('rating')
    category              = req_params.get('category')

    if not client_principal_id or client_principal_id == '':
        client_principal_id = '00000000-0000-0000-0000-000000000000'
        client_principal_name = 'anonymous'  

    if req.method == "POST": 
        # required
        if not conversation_id or conversation_id == '':
            return func.HttpResponse("Invalid request body: conversation_id not set", status_code=400)
        if not question or question == '':
            return func.HttpResponse("Invalid request body: question not set", status_code=400)
        if not answer or answer == '':
            return func.HttpResponse("Invalid request body: answer not set", status_code=400)
        # optional
        if not feedback_message:
            feedback_message = ''
        if not category:
            category = 'Not set'

    client_principal = {
        'id': client_principal_id,
        'name': client_principal_name
    }

    if req.method == "GET":
        if conversation_id:
            conversations = get_feedback(conversation_id, client_principal)
        else:
            conversations = get_feedback_all(client_principal)
    elif req.method == "POST":
        conversations = set_feedback(
            client_principal=client_principal,
            conversation_id=conversation_id,
            feedback_message=feedback_message,
            question=question,
            answer=answer,
            rating=rating,
            category=category
        )

    if "error" in conversations:
        return func.HttpResponse(conversations["error"], status_code=400)

    return func.HttpResponse(json.dumps(conversations), mimetype="application/json", status_code=200)