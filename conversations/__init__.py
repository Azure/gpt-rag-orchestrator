import logging
import azure.functions as func
import json
import os

from shared.util import get_conversations, get_conversation, delete_conversation

LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)

async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    id = req.params.get('id')

    if req.method == "GET":
        req_body = req.get_json()
        user_id = req_body.get('user_id')
        if not id:
            conversations = get_conversations(user_id)
        else:
            conversations = get_conversation(id, user_id)
        return func.HttpResponse(json.dumps(conversations), mimetype="application/json", status_code=200)

    elif req.method == "DELETE":
        req_body = req.get_json()
        user_id = req_body.get('user_id')
        if id:
            try:
                delete_conversation(id, user_id)
                return func.HttpResponse("Conversation deleted successfully", status_code=200)
            except Exception as e:
                logging.error(f"Error deleting conversation: {str(e)}")
                return func.HttpResponse("Error deleting conversation", status_code=500)
        else:
            return func.HttpResponse("Missing conversation ID", status_code=400)

    else:
        return func.HttpResponse("Method not allowed", status_code=405)
