import logging
import uvicorn

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from orchestration.orchestrator import Orchestrator

from connectors.appconfig import AppConfigClient
from dependencies import get_config, validate_dapr_token, validate_api_key_header
from telemetry import Telemetry
from constants import APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME
from util.tools import is_azure_environment

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
    
# ----------------------------------------
# Create FastAPI app with lifespan
# ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    Telemetry.configure_monitoring(cfg, APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME)

    # (optional) startup logic before application starts
    yield  # <-- application starts here
    # (optional) cleanup logic after shutdown
 

app = FastAPI(
        title="GPT RAG Orchestrator",
        description="GPT RAG Orchestrator FastAPI",
        version="1.0.0",
        lifespan=lifespan
    )

# ----------------------------------------
# Streaming endpoint
# ----------------------------------------
@app.post("/orchestrator", dependencies=[Depends(validate_dapr_token)])
async def orchestrator_endpoint(request: Request):
    """
    Accepts JSON payload {"ask": "...", optional "conversation_id": "..."},
    then streams back an answer via SSE.
    """
    payload = await request.json()
    ask = payload.get("ask")
    if not ask:
        raise HTTPException(status_code=400, detail="No 'ask' field in request body")
    
    if payload.get("type") == "feedback":
        # Handle feedback submission
        conversation_id = payload.get("conversation_id")
        if not conversation_id:
            logging.error(f"No 'conversation_id' provided in feedback payload, and payload is {payload}")
            raise HTTPException(status_code=400, detail="No 'conversation_id' field in request body")

        # Create orchestrator instance and save feedback
        orchestrator = await Orchestrator.create(conversation_id=conversation_id)
        await orchestrator.save_feedback({
            "ask": ask,
            "conversation_id": conversation_id,
            "question_id": payload.get("question_id"),
            "is_positive": payload.get("is_positive"),
            "stars_rating": payload.get("stars_rating"),
            "feedback_text": payload.get("feedback_text")
        })
        return {"status": "success", "message": "Feedback saved successfully"}
    
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