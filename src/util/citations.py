"""
Citation Processing Utilities.

Shared helpers for processing Bing Grounding citations from
Azure AI Agent Service message deltas. Extracted from the legacy
SingleAgentRAGStrategyV1 so that any strategy can reuse them.
"""

import re

from azure.ai.agents.models import MessageDeltaChunk

# Pre-compiled regex for citation placeholders (e.g., 【7:0†source】)
# Compiled once at module load for better performance
CITATION_PLACEHOLDER_PATTERN = re.compile(r'【(\d+):(\d+)†[^】]*】')


def truncate_title(title: str, max_length: int = 30) -> str:
    """
    Truncate title to max_length, cutting at the last space before the limit.

    Args:
        title: The title to truncate
        max_length: Maximum length (default 30)

    Returns:
        Truncated title with '...' if needed
    """
    if not title or len(title) <= max_length:
        return title

    # Find the last space before max_length
    truncated = title[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > 0:
        return truncated[:last_space] + '...'
    else:
        # No space found, just truncate at max_length
        return truncated + '...'


def process_bing_citations(delta: MessageDeltaChunk) -> str:
    """
    Process Bing Grounding citations from message delta.
    Replaces placeholders like 【3:0†source】 with proper [title](url) format.

    Note: Only works with OpenAI/Azure OpenAI models that include url_citation
    annotations. Other models (e.g., DeepSeek) don't provide annotations,
    so placeholders are simply removed.

    Args:
        delta: The message delta chunk containing text and potential annotations
    """
    text = delta.text
    if not text:
        return text

    # Collect annotation objects from the delta
    raw = getattr(delta, "delta", None)
    annotations = []

    if raw:
        raw_content = getattr(raw, "content", [])
        for piece in raw_content:
            txt = getattr(piece, "text", None)
            if txt:
                anns = getattr(txt, "annotations", None)
                if anns:
                    annotations.extend(anns)

    # Process URL citations (used by Bing Grounding with OpenAI/Azure models)
    for ann in annotations:
        placeholder = None
        url = None
        title = None

        # Convert annotation to dict (handles Pydantic models, etc.)
        ann_dict = None
        if hasattr(ann, 'model_dump'):
            ann_dict = ann.model_dump()
        elif hasattr(ann, 'dict'):
            ann_dict = ann.dict()
        elif hasattr(ann, '__dict__'):
            ann_dict = ann.__dict__
        elif isinstance(ann, dict):
            ann_dict = ann

        if ann_dict:
            # Handle nested _data structure (Azure SDK format)
            if '_data' in ann_dict:
                ann_dict = ann_dict['_data']

            ann_type = ann_dict.get('type', '')
            if ann_type == 'url_citation' or 'url_citation' in ann_dict:
                placeholder = ann_dict.get('text')
                url_citation = ann_dict.get('url_citation', {})

                if isinstance(url_citation, dict):
                    url = url_citation.get('url')
                    title = url_citation.get('title')
                else:
                    url = getattr(url_citation, 'url', None)
                    title = getattr(url_citation, 'title', None)

        if not title:
            title = url

        if url and placeholder and placeholder in text:
            display_title = truncate_title(title, 30)
            citation = f"[{display_title}]({url})"
            text = text.replace(placeholder, citation)

    # Clean up citation placeholders for models that don't include annotations (e.g., DeepSeek)
    # Uses pre-compiled CITATION_PLACEHOLDER_PATTERN for better performance
    text = CITATION_PLACEHOLDER_PATTERN.sub('', text)

    return text
