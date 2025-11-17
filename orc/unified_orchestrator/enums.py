"""
Enums for the unified orchestrator.

This module contains enum definitions used throughout the orchestrator workflow.
"""

from enum import Enum
from shared.prompts import (
    VERBOSITY_MODE_BRIEF,
    VERBOSITY_MODE_BALANCED,
    VERBOSITY_MODE_DETAILED,
)


class VerbosityLevel(str, Enum):
    """Verbosity levels for response generation."""
    BRIEF = "brief"
    BALANCED = "balanced"
    DETAILED = "detailed"


# Mapping from verbosity level to prompt text
VERBOSITY_PROMPTS = {
    VerbosityLevel.BRIEF: VERBOSITY_MODE_BRIEF,
    VerbosityLevel.BALANCED: VERBOSITY_MODE_BALANCED,
    VerbosityLevel.DETAILED: VERBOSITY_MODE_DETAILED,
}
