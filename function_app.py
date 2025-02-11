import time
import azure.functions as func
import logging
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse
from orc import new_orchestrator

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="orc", methods=[func.HttpMethod.POST])
async def stream_response(req: Request) -> StreamingResponse:
    """Endpoint to stream LLM responses to the client"""
    
    logging.info('Python HTTP trigger function processed a request.')

    req_body = await req.json()
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
    
    orchestrator = new_orchestrator.ConversationOrchestrator()
    resources =  orchestrator.process_conversation(
        conversation_id, question, client_principal
    )
    return StreamingResponse(orchestrator.generate_response(resources["conversation_id"],resources["state"], resources["conversation_data"], client_principal, resources["memory_data"], resources["start_time"]), media_type="text/event-stream")