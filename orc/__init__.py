"""
Orchestrator module exports.

This module exports the unified conversation orchestrator and utility functions
for use by Azure Function entry points.
"""

from orc.unified_orchestrator import ConversationOrchestrator
from shared.util import get_setting

import logging

logger = logging.getLogger(__name__)


def get_settings(client_principal):
    """
    Retrieve user settings from Cosmos DB.
    
    This function maintains backward compatibility with the existing API
    by wrapping the shared.util.get_setting function.
    
    Args:
        client_principal: Client principal object with user information
        
    Returns:
        Dictionary with user settings (temperature, model, detail_level)
    """
    data = get_setting(client_principal)
    temperature = None if "temperature" not in data else data["temperature"]
    model = None if "model" not in data else data["model"]
    detail_level = None if "detail_level" not in data else data["detail_level"]
    
    settings = {
        "temperature": temperature,
        "model": model,
        "detail_level": detail_level,
    }
    
    logger.info(f"[orc] Retrieved settings: {settings}")
    return settings


__all__ = [
    "ConversationOrchestrator",
    "get_settings",
]
