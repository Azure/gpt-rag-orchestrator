import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from collections.abc import Mapping

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from orchestration.orchestrator import Orchestrator
from connectors.appconfig import AppConfigClient
from dependencies import get_config, validate_auth, validate_access_token
from telemetry import Telemetry
from schemas import OrchestratorRequest, ORCHESTRATOR_RESPONSES
from constants import APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME
from util.tools import is_azure_environment
from util.jwt_utils import extract_bearer_token

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


def _format_request_debug(
    request: Request,
    conversation_id: Optional[str],
    question_id: Optional[str],
    op_type: Optional[str],
) -> str:
    # NOTE: keep this safe. Do not log secrets/tokens.
    sensitive_markers = (
        "authorization",
        "cookie",
        "set-cookie",
        "token",
        "apikey",
        "api-key",
        "secret",
        "password",
    )
    # A small allowlist of headers we often want to see fully.
    safe_value_allowlist = {
        "host",
        "user-agent",
        "content-type",
        "content-length",
        "accept",
        "accept-encoding",
        "accept-language",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-host",
        "x-request-id",
        "traceparent",
        "tracestate",
        "x-correlation-id",
    }

    def _is_sensitive(name: str) -> bool:
        n = name.lower()
        return any(m in n for m in sensitive_markers)

    def _redact(name: str, value: str) -> str:
        # Show presence + length only.
        return f"<redacted len={len(value)}>"

    # Pull a few request metadata fields
    method = request.method
    path = request.url.path
    client_host = getattr(getattr(request, "client", None), "host", None)

    # Normalize headers into a simple dict
    hdrs: Mapping[str, str] = request.headers
    header_lines: list[str] = []
    for k in sorted(hdrs.keys(), key=lambda s: s.lower()):
        v = hdrs.get(k, "")
        if not v:
            continue
        kl = k.lower()
        if _is_sensitive(kl):
            header_lines.append(f"  - {k}: {_redact(kl, v)}")
        elif kl in safe_value_allowlist:
            header_lines.append(f"  - {k}: {v}")
        else:
            # For non-sensitive, non-allowlisted headers, log a short preview.
            preview = (v[:120] + "…") if len(v) > 120 else v
            header_lines.append(f"  - {k}: {preview}")

    req_id = hdrs.get("x-request-id") or hdrs.get("x-correlation-id") or None
    traceparent = hdrs.get("traceparent") or None

    lines = [
        "[Orchestrator] ── Request Debug ─────────────────────────────",
        f"method: {method}",
        f"path:   {path}",
        f"client: {client_host or 'unknown'}",
        f"type:   {op_type or 'ask'}",
        f"conversation_id: {conversation_id or '∅'}",
        f"question_id:     {question_id or '∅'}",
    ]
    if req_id:
        lines.append(f"request_id: {req_id}")
    if traceparent:
        lines.append(f"traceparent: {traceparent}")

    lines.append("headers:")
    lines.extend(header_lines if header_lines else ["  (none)"])
    lines.append("[Orchestrator] ─────────────────────────────────────────────")
    return "\n".join(lines)

# 2) Create configuration client (sets cfg.auth_failed=True if auth is unavailable)
cfg: AppConfigClient = get_config()

# 3) Configure logging level/format from LOG_LEVEL
Telemetry.configure_basic(cfg)
Telemetry.log_log_level_diagnostics(cfg)

# Reduce noise from low-level HTTP libraries unless explicitly requested.
# When LOG_LEVEL=DEBUG we still want app debug, but httpcore/httpx/urllib3 can overwhelm logs.
if os.getenv("HTTP_CLIENT_DEBUG", "false").lower() not in ("1", "true", "yes"):
    for _noisy in [
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
    ]:
        logging.getLogger(_noisy).setLevel(logging.WARNING)

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
    request: Request,
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

    # Anonymous-mode toggle (mirrors frontend behavior):
    # - If ALLOW_ANONYMOUS=true, requests without Authorization can proceed as anonymous.
    # - If ALLOW_ANONYMOUS=false, Authorization is required (401 when missing).
    # - If Entra auth isn't configured (tenant/client id missing), ALLOW_ANONYMOUS controls whether to proceed.
    try:
        _tenant_id = (cfg.get("OAUTH_AZURE_AD_TENANT_ID", default="") or "").strip()
        _client_id = (
            (cfg.get("OAUTH_AZURE_AD_CLIENT_ID", default="") or "").strip()
            or (cfg.get("CLIENT_ID", default="") or "").strip()
        )
        auth_configured = bool(_tenant_id and _client_id)
    except Exception:
        auth_configured = False

    # Default to allowing anonymous access unless explicitly disabled.
    allow_anonymous = cfg.get("ALLOW_ANONYMOUS", default=True, type=bool)

    # For troubleshooting, track the auth decision taken for this request.
    auth_header_present = bool(authorization)
    auth_decision = "unknown"

    # Pretty request/header logging (DEBUG only)
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        try:
            logging.debug(
                _format_request_debug(
                    request=request,
                    conversation_id=getattr(body, "conversation_id", None),
                    question_id=getattr(body, "question_id", None),
                    op_type=op_type,
                )
            )
        except Exception:
            # Never fail the request due to logging
            logging.debug("[Orchestrator] Failed to render request debug info", exc_info=True)

    # If Authorization header is provided, validate the access token and apply authorization checks.
    # If Authorization header is missing, treat the request as anonymous only when ALLOW_ANONYMOUS=true.
    user_context = body.user_context or {}
    access_token: Optional[str] = None
    if authorization:
        # If auth isn't configured, decide whether to proceed based on ALLOW_ANONYMOUS.
        if not auth_configured:
            if allow_anonymous:
                auth_decision = "allow_anonymous_auth_not_configured"
                logging.warning(
                    "[Orchestrator] Authorization header provided but Entra auth is not configured; proceeding as anonymous (ALLOW_ANONYMOUS=true)"
                )
                user_context.setdefault("principal_id", "anonymous")
                user_context.setdefault("principal_name", "anonymous")
            else:
                auth_decision = "reject_auth_not_configured"
                logging.warning(
                    "[Orchestrator] Authentication required (ALLOW_ANONYMOUS=false) but Entra auth is not configured (missing OAUTH_AZURE_AD_TENANT_ID/OAUTH_AZURE_AD_CLIENT_ID)"
                )
                raise HTTPException(status_code=401, detail="Authentication is not configured")
        else:
            auth_decision = "validate_bearer_token"
            logging.debug("[Orchestrator] Authorization header provided; validating access token...")

            # Extract Bearer token safely (handles extra whitespace, casing, etc.)
            access_token = extract_bearer_token(authorization)
            if not access_token:
                logging.warning("[Orchestrator] Invalid Authorization header format (expected 'Bearer <token>')")
                raise HTTPException(status_code=401, detail="Invalid Authorization header format")

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

                # Check authorization based on user principals/names
                allowed_names = [n.strip() for n in cfg.get("ALLOWED_USER_NAMES", default="").split(",") if n.strip()]
                allowed_ids = [id.strip() for id in cfg.get("ALLOWED_USER_PRINCIPALS", default="").split(",") if id.strip()]

                logging.debug(
                    "[Orchestrator] Authorization policy - allowed_names=%d allowed_ids=%d",
                    len(allowed_names),
                    len(allowed_ids),
                )

                is_authorized = (
                    not (allowed_names or allowed_ids) or
                    user_info.get("preferred_username") in allowed_names or
                    user_info.get("oid") in allowed_ids
                )

                if not is_authorized:
                    # High-signal deny reason (no secrets)
                    deny_reasons = []
                    if allowed_names and user_info.get("preferred_username") not in allowed_names:
                        deny_reasons.append("preferred_username_not_allowed")
                    if allowed_ids and user_info.get("oid") not in allowed_ids:
                        deny_reasons.append("oid_not_allowed")
                    deny_reason = ",".join(deny_reasons) if deny_reasons else "policy_requires_no_restrictions"
                    logging.warning(
                        "[Orchestrator] ❌ Access denied: user_oid=%s user=%s reason=%s",
                        user_info.get("oid"),
                        user_info.get("preferred_username"),
                        deny_reason,
                    )
                    raise HTTPException(status_code=403, detail="You are not authorized to perform this action")

                logging.info(
                    "[Orchestrator] ✅ Authenticated request: conversation_id=%s question_id=%s user=%s oid=%s",
                    getattr(body, "conversation_id", None) or "∅",
                    getattr(body, "question_id", None) or "∅",
                    user_info.get("preferred_username") or user_info.get("oid") or "<unknown>",
                    user_info.get("oid") or "<unknown>",
                )

                auth_decision = "authenticated"
            except HTTPException as e:
                # Always log the rejection reason (safe: no tokens). This makes 401/403 troubleshooting easier.
                logging.warning(
                    "[Orchestrator] Request rejected: status=%d detail=%s",
                    e.status_code,
                    getattr(e, "detail", None),
                )
                raise
            except Exception as e:
                logging.error(
                    "[Orchestrator] Error validating user token: %s: %s",
                    type(e).__name__,
                    str(e),
                )
                raise HTTPException(status_code=401, detail="Invalid or expired token")
    else:
        # No Authorization header: allow anonymous only when explicitly enabled.
        if allow_anonymous:
            auth_decision = "allow_anonymous_missing_auth_header"
            logging.debug("[Orchestrator] No Authorization header; treating as anonymous (ALLOW_ANONYMOUS=true)")
            user_context.setdefault("principal_id", "anonymous")
            user_context.setdefault("principal_name", "anonymous")
        else:
            auth_decision = "reject_missing_auth_header"
            # Mirror token-invalid behavior: 401 when auth is required.
            logging.warning(
                "[Orchestrator] Missing Authorization header and ALLOW_ANONYMOUS=false; rejecting request"
            )
            raise HTTPException(status_code=401, detail="Missing Authorization header")

    # One INFO line per request: quickly answers "was this anonymous? who was it?" without DEBUG.
    try:
        principal_name = (user_context.get("principal_name") or "").strip()
        principal_id = (user_context.get("principal_id") or "").strip()
        principal = principal_name or principal_id or "anonymous"
        auth_mode = "authenticated" if (authorization and principal != "anonymous") else "anonymous"
        logging.info(
            "[Orchestrator] Request context: type=%s conversation_id=%s question_id=%s auth=%s principal=%s allow_anonymous=%s auth_configured=%s auth_header=%s decision=%s",
            op_type or "ask",
            getattr(body, "conversation_id", None) or "∅",
            getattr(body, "question_id", None) or "∅",
            auth_mode,
            principal,
            allow_anonymous,
            auth_configured,
            auth_header_present,
            auth_decision,
        )
    except Exception:
        # Never fail due to logging
        pass

    logging.debug(
        "[Orchestrator] Request identity resolved: principal_name=%s principal_id=%s",
        user_context.get("principal_name"),
        user_context.get("principal_id"),
    )

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
        user_context=user_context,
        request_access_token=access_token if authorization else None,
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

