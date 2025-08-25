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
                self.yield_func(json.dumps(progress_data) + "\n" + " " * 8192 + "\n")
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
                self.yield_func(json.dumps(error_data) + "\n" + " " * 8192 + "\n")
            except Exception as e:
                logger.warning(f"Failed to yield error: {e}")


class ProgressSteps:
    """Constants for standardized progress step names."""

    INITIALIZATION = "initialization"
    QUERY_REWRITE = "query_rewrite"
    QUERY_CATEGORIZATION = "query_categorization"
    ROUTING = "routing"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    AGENTIC_SEARCH = "agentic_search"
    DATA_ANALYSIS = "data_analysis"
    RESPONSE_GENERATION = "response_generation"


STEP_MESSAGES = {
    ProgressSteps.INITIALIZATION: "Starting conversation...",
    ProgressSteps.QUERY_REWRITE: "Analyzing & rewriting your question...",
    ProgressSteps.QUERY_CATEGORIZATION: "Categorizing your request...",
    ProgressSteps.ROUTING: "Determining information sources needed...",
    ProgressSteps.TOOL_SELECTION: "Selecting appropriate tools...",
    ProgressSteps.TOOL_EXECUTION: "Executing tools...",
    ProgressSteps.AGENTIC_SEARCH: "Executing Agentic Search...",
    ProgressSteps.DATA_ANALYSIS: "Executing Data Analysis Tool...",
    ProgressSteps.RESPONSE_GENERATION: "Generating your response...",
}
