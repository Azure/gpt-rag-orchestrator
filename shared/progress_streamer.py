import json
import time
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class ProgressStreamer:
    """Handles progress streaming for conversation processing pipeline."""

    def __init__(self, yield_func: Optional[Callable[[str], None]] = None):
        """Initialize progress streamer with optional yield function."""
        self.yield_func = yield_func

    def emit_progress(
        self,
        step: str,
        message: str,
        progress: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Emit a progress update."""
        progress_data = {
            "type": "progress",
            "step": step,
            "message": message,
            "progress": progress,
            "timestamp": time.time(),
            "data": data or {},
        }

        if self.yield_func:
            try:
                self.yield_func(f"__PROGRESS__{json.dumps(progress_data)}__PROGRESS__\n")
            except Exception as e:
                logger.warning(f"Failed to yield progress: {e}")

    def emit_error(self, step: str, message: str, error_details: Optional[str] = None):
        """Emit an error progress update."""
        error_data = {
            "type": "error",
            "step": step,
            "message": message,
            "timestamp": time.time(),
            "error_details": error_details,
        }

        if self.yield_func:
            try:
                self.yield_func(f"__PROGRESS__{json.dumps(error_data)}__PROGRESS__\n")
            except Exception as e:
                logger.warning(f"Failed to yield error: {e}")


class ProgressSteps:
    """Constants for standardized progress step names."""

    INITIALIZATION = "initialization"
    QUERY_REWRITE = "query_rewrite"
    AGENTIC_SEARCH = "agentic_search"
    DATA_ANALYSIS = "data_analysis"
    RESPONSE_GENERATION = "response_generation"


STEP_MESSAGES = {
    ProgressSteps.INITIALIZATION: "Firing up the FreddAid engine...",
    ProgressSteps.QUERY_REWRITE: "FreddAid’s tweezers out - refining your question...",
    ProgressSteps.AGENTIC_SEARCH: "FreddAid is on the case - searching for answers...",
    ProgressSteps.DATA_ANALYSIS: "FreddAid’s analysis gears are grinding...",
    ProgressSteps.RESPONSE_GENERATION: "Composing your response...",
}
