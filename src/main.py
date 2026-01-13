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
from dependencies import get_config, validate_auth, validate_access_token, get_user_groups_from_graph
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

# Suppress Azure SDK HTTP logging immediately (before any Azure imports)
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    logger = logging.getLogger(_azure_logger)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True
    logger.handlers.clear()
    
# Load version from VERSION file 
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
try:
    APP_VERSION = VERSION_FILE.read_text().strip()
except FileNotFoundError:
    APP_VERSION = "0.0.0"


def _startup_banner() -> None:
    name = "GPT-RAG Orchestrator"
    version = None
    try:
        if VERSION_FILE.exists():
            version = VERSION_FILE.read_text().strip() or None
    except Exception:
        version = None

    title = f"{name}{(' v' + version) if version else ''}"
    banner_lines = [
        "",
        "╔══════════════════════════════════════════════╗",
        f"║  {title}".ljust(47) + "║",
        "║  FastAPI Orchestrator API                    ║",
        "╚══════════════════════════════════════════════╝",
        "",
    ]
    for line in banner_lines:
        logging.info(line)

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
    _startup_banner()
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
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Accepts JSON payload with ask/question, optional conversation_id and context,
    then streams back an answer via SSE.
    """

    # Determine operation type first (defensive: body may not include type)
    op_type = getattr(body, "type", None)

    # If Authorization header is provided, validate the access token and apply authorization checks.
    # If Authorization header is missing, treat the request as anonymous.
    user_context = body.user_context or {}
    if authorization:
        logging.debug("[Orchestrator] Authorization header provided; validating access token...")

        # Extract Bearer token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            logging.warning("[Orchestrator] Invalid Authorization header format")
            raise HTTPException(status_code=401, detail="Invalid Authorization header format")

        access_token = authorization[7:]  # Remove "Bearer " prefix
        logging.debug("[Orchestrator] Access token received, length: %d chars", len(access_token))

        try:
            # Validate token and extract user info
            user_info = await validate_access_token(access_token)
            user_context["principal_id"] = user_info.get("oid")
            user_context["principal_name"] = user_info.get("preferred_username")
            user_context["user_name"] = user_info.get("name")

            logging.debug(
                "[Orchestrator] User info extracted: OID=%s, Username=%s, Name=%s",
                user_info.get("oid"),
                user_info.get("preferred_username"),
                user_info.get("name"),
            )

            # Fetch user groups from Graph API
            logging.debug("[Orchestrator] Fetching user groups from Graph API...")
            groups = await get_user_groups_from_graph(user_info.get("oid"))
            user_context["groups"] = groups

            logging.debug("[Orchestrator] User groups: %s", groups)

            # Check authorization based on groups/principals
            allowed_names = [n.strip() for n in cfg.get("ALLOWED_USER_NAMES", default="").split(",") if n.strip()]
            allowed_ids = [id.strip() for id in cfg.get("ALLOWED_USER_PRINCIPALS", default="").split(",") if id.strip()]
            allowed_groups = [g.strip() for g in cfg.get("ALLOWED_GROUP_NAMES", default="").split(",") if g.strip()]

            logging.debug(
                "[Orchestrator] Authorization checks - Allowed names: %s, IDs: %s, Groups: %s",
                allowed_names,
                allowed_ids,
                allowed_groups,
            )

            is_authorized = (
                not (allowed_names or allowed_ids or allowed_groups) or
                user_info.get("preferred_username") in allowed_names or
                user_info.get("oid") in allowed_ids or
                any(g in allowed_groups for g in groups)
            )

            if not is_authorized:
                logging.warning(
                    "[Orchestrator] ❌ Access denied for user %s (%s)",
                    user_info.get("oid"),
                    user_info.get("preferred_username"),
                )
                raise HTTPException(status_code=403, detail="You are not authorized to perform this action")

            logging.info("[Orchestrator] ✅ Authorization successful for user %s", user_info.get("preferred_username"))
        except HTTPException:
            raise
        except Exception as e:
            logging.error("[Orchestrator] Error validating user token: %s", e)
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    else:
        # No Authorization header; treat as anonymous
        user_context.setdefault("principal_id", "anonymous")
        user_context.setdefault("principal_name", "anonymous")
        user_context.setdefault("groups", [])

    # Feedback submissions: allow missing ask/question; validate only what's required
    if op_type == "feedback":
        # Handle feedback submission
        conversation_id = body.conversation_id
        if not conversation_id:
            logging.error(f"No 'conversation_id' provided in feedback body, and payload is {body}")
            raise HTTPException(status_code=400, detail="No 'conversation_id' field in request body")

        # Create orchestrator instance and save feedback
        orchestrator = await Orchestrator.create(conversation_id=conversation_id, user_context=user_context)
        # Build feedback dict defensively; optional fields may be absent
        _qid = getattr(body, "question_id", None)
        feedback = {
            "conversation_id": conversation_id,
            "question_id": _qid,
            "is_positive": getattr(body, "is_positive", None),
            "stars_rating": getattr(body, "stars_rating", None),
            # Normalize empty strings to None
            "feedback_text": (getattr(body, "feedback_text", None) or "").strip() or None,
        }
        await orchestrator.save_feedback(feedback)
        return {"status": "success", "message": "Feedback saved successfully"}    

    # For non-feedback operations, require an ask/question
    ask = (getattr(body, "ask", None) or getattr(body, "question", None))
    if not ask:
        raise HTTPException(status_code=400, detail="No 'ask' or 'question' field in request body")

    orchestrator = await Orchestrator.create(
        conversation_id=body.conversation_id,
        user_context=user_context
    )

    async def sse_event_generator():
        try:
            _qid = getattr(body, "question_id", None) 
            async for chunk in orchestrator.stream_response(ask, _qid):
                yield f"{chunk}"
        except Exception as e:
            logging.exception("Error in SSE generator")
            yield "event: error\ndata: An internal server error occurred.\n\n"

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

