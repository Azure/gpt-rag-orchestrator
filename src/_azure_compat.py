"""
Runtime compatibility shim for agent-framework-azure-ai 1.0.0b260130.

That package imports class names from azure.ai.projects.models that were
renamed in azure-ai-projects 2.0.1.  Instead of patching files on disk,
we inject the old names as aliases at import time.

This module must be imported before any agent_framework code runs
(see main.py).
"""

from azure.ai.projects import models as _models

_ALIASES = {
    "CodeInterpreterToolAuto": "AutoCodeInterpreterToolParam",
    "ResponseTextFormatConfigurationJsonObject": "TextResponseFormatJsonObject",
    "ResponseTextFormatConfigurationJsonSchema": "TextResponseFormatJsonSchema",
    "ResponseTextFormatConfigurationText": "TextResponseFormatText",
    "PromptAgentDefinitionText": "PromptAgentDefinitionTextOptions",
    "AgentReference": "AgentDetails",
}

for _old, _new in _ALIASES.items():
    if not hasattr(_models, _old) and hasattr(_models, _new):
        setattr(_models, _old, getattr(_models, _new))
