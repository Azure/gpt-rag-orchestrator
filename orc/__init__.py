import logging
import azure.functions as func
import json
import logging
from . import orchestrator

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    req_body = req.get_json()
    conversation_id = req_body.get('conversation_id')
    
    question = req_body.get('question')

    if question:
        result = orchestrator.run(conversation_id, question)
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    else:
        return func.HttpResponse('{"error": "no question found in json input"}', mimetype="application/json", status_code=200)
