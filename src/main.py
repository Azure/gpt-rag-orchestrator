import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from orchestration.orchestrator import Orchestrator
from connectors.appconfig import AppConfigClient
from dependencies import get_config, validate_auth
from telemetry import Telemetry
from schemas import OrchestratorRequest, ORCHESTRATOR_RESPONSES
from constants import APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME
from util.tools import is_azure_environment

# ----------------------------------------
# Initialization and logging
# - Minimal early logging so config/auth warnings are visible during startup
# - Azure SDK and HTTP pipeline logs are verbose only when LOG_LEVEL=DEBUG;
#   otherwise they’re kept at WARNING to reduce noise
# ----------------------------------------

## Early minimal logging (INFO) until config is loaded; refined by Telemetry.configure_basic
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
# Load version from VERSION file 
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
try:
    APP_VERSION = VERSION_FILE.read_text().strip()
except FileNotFoundError:
    APP_VERSION = "0.0.0"

# 2) Create configuration client (sets cfg.auth_failed=True if auth is unavailable)
cfg: AppConfigClient = get_config()

# 3) Configure logging level/format from LOG_LEVEL
Telemetry.configure_basic(cfg)
Telemetry.log_log_level_diagnostics(cfg)

# 4) If authentication failed, exit immediately
if getattr(cfg, "auth_failed", False):
    logging.warning("The orchestrator is not authenticated (run 'az login' or configure Managed Identity). Exiting...")
    logging.shutdown()
    os._exit(1)

# ----------------------------------------
# Create FastAPI app with lifespan
# ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Telemetry.configure_monitoring(cfg, APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME)
    yield  # <-- application runs here
    # cleanup logic after shutdown

app = FastAPI(
    title="GPT-RAG Orchestrator",
    description="GPT-RAG Orchestrator FastAPI",
    version=APP_VERSION,
    lifespan=lifespan
)

@app.post(
    "/orchestrator",
    dependencies=[Depends(validate_auth)], 
    summary="Ask orchestrator a question",
    response_description="Returns the orchestrator’s response in real time, streamed via SSE.",
    responses=ORCHESTRATOR_RESPONSES
)
async def orchestrator_endpoint(
    body: OrchestratorRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
):
    """
    Accepts JSON payload with ask/question, optional conversation_id and context,
    then streams back an answer via SSE.
    """

    # Prefer "ask", fallback to "question" for compatibility
    ask = body.ask or body.question
    if not ask:
        raise HTTPException(status_code=400, detail="No 'ask' or 'question' field in request body")

    user_context = body.user_context or {}

    orchestrator = await Orchestrator.create(
        conversation_id=body.conversation_id,
        user_context=user_context
    )

    async def sse_event_generator():
        try:
            async for chunk in orchestrator.stream_response(ask):
                yield f"{chunk}"
        except Exception as e:
            logging.exception("Error in SSE generator")
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream"
    )

# Instrumentation
HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app)

# Run the app locally (avoid nested event loop when started by uvicorn CLI)
if __name__ == "__main__" and not is_azure_environment():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info", timeout_keep_alive=60)
