import logging
import azure.functions as func
import os
import json
from . import orchestrator

LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
logging.basicConfig(level=LOGLEVEL)


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("[financial-orchestrator] Python HTTP trigger function processed a request.")

    # request body should look like this:
    # {
    #     "question": "string",
    #     "conversation_id": "string"
    #     "documentName": "string",
    #     "client_principal_id": "string",
    #     "client_principal_name": "string",
    # }

    req_body = req.get_json()
    conversation_id = req_body.get("conversation_id")
    question = req_body.get("question")

    client_principal_id = req_body.get("client_principal_id")
    client_principal_name = req_body.get("client_principal_name")

    # User is anonymous if no client_principal_id is provided
    if not client_principal_id or client_principal_id == "":
        client_principal_id = "00000000-0000-0000-0000-000000000000"
        client_principal_name = "anonymous"

    client_principal = {"id": client_principal_id, "name": client_principal_name}

    documentName = req_body.get("documentName", "")

    if not documentName or documentName == "":
        logging.error("[financial-orchestrator] no documentName found in json input")
        return func.HttpResponse(
            json.dumps({"error": "no documentName found in json input"}),
            mimetype="application/json",
            status_code=400,
        )

    # validate documentName exists in a hardcoded list
    # if not documentName in ['financial', 'feedback']:
    #     return func.HttpResponse('{"error": "invalid documentName"}', mimetype="application/json", status_code=200)

    if question:
        result = await orchestrator.run(
            conversation_id, question, documentName, client_principal
        )
        return func.HttpResponse(
            json.dumps(result), mimetype="application/json", status_code=200
        )
    else:
        logging.error("[financial-orchestrator] no question found in json input")
        return func.HttpResponse(
            json.dumps({"error": "no question found in json input"}),
            mimetype="application/json",
            status_code=400,
        )
