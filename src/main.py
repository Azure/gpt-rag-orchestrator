import os
import logging
import uvicorn

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from orchestration.orchestrator import Orchestrator

from connectors.appconfig import AppConfigClient
from dependencies import get_config, validate_api_key_header
from telemetry import Telemetry
from constants import APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME
from util.tools import is_azure_environment

# app configuration
cfg : AppConfigClient = get_config()
    
@asynccontextmanager
async def lifespan(app: FastAPI):

    Telemetry.configure_monitoring(cfg, APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME)

    # Placeholder for future startup logic
    yield  # <-- application starts here
    # (optional) cleanup logic after shutdown
 
# ----------------------------------------
# Create FastAPI app with lifespan
# ----------------------------------------
app = FastAPI(
        title="GPT RAG Orchestrator",
        description="GPT RAG Orchestrator FastAPI",
        version="1.0.0",
        lifespan=lifespan
    )

# ----------------------------------------
# Streaming endpoint
# ----------------------------------------
@app.post("/orchestrator", dependencies=[Depends(validate_api_key_header)])
async def orchestrator_endpoint(request: Request):
    """
    Accepts JSON payload {"ask": "...", optional "conversation_id": "..."},
    then streams back an answer via SSE.
    """
    payload = await request.json()
    ask = payload.get("ask")
    if not ask:
        raise HTTPException(status_code=400, detail="No 'ask' field in request body")

    user_context = payload.get("user-context", {})

    orchestrator = await Orchestrator.create(conversation_id=payload.get("conversation_id"), user_context=user_context)

    async def sse_event_generator():
        try:
            # Consume each chunk from the orchestrator
            async for chunk in orchestrator.stream_response(ask):
                yield f"{chunk}"
        except Exception as e:
            logging.exception("Error in SSE generator")
            yield f"event: error\ndata: {str(e)}\n\n"

    # Return an SSE response
    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream"
    )

HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app)

# Run the app locally
if (not is_azure_environment()):
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="debug", timeout_keep_alive=60)
