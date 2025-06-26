import os
import logging

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse

from orchestration.orchestrator import Orchestrator

from connectors.appconfig import AppConfigClient
from dependencies import get_config, validate_api_key_header

# app configuration
cfg : AppConfigClient = get_config()

# ----------------------------------------
# Logging configuration
# ----------------------------------------
log_level = cfg.get("LOG_LEVEL", "INFO")
logging.basicConfig(  
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
http_logger.setLevel(logging.DEBUG) 
class DebugModeFilter(logging.Filter):
    def filter(self, record):
        return logging.getLogger().getEffectiveLevel() == logging.DEBUG
http_logger.addFilter(DebugModeFilter())
    
@asynccontextmanager
async def lifespan(app: FastAPI):
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

    orchestrator = Orchestrator(conversation_id=payload.get("conversation_id"))

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