import json
import logging
from typing import Any, List, Optional
from urllib.parse import unquote
import re

logger = logging.getLogger(__name__)


def sanitize_storage_urls(text: str, storage_url: Optional[str]) -> str:
    """Remove sensitive storage URLs from response text.

    Args:
        text: Original text content
        storage_url: Storage account URL to sanitize (if present)
    """
    if not text or not storage_url:
        return text
    try:
        regex = rf"(Source:\s?\/?)?(source:)?(https:\/\/)?({storage_url})?(\/?documents\/?)?"
        return re.sub(regex, "", text)
    except Exception as e:
        logger.warning(f"Failed to sanitize storage URLs: {e}")
        return text


def format_source_path(source_path: str) -> str:
    """
    Formats a source path by extracting relevant parts and URL decoding.

    Args:
        source_path: The raw source path from document metadata

    Returns:
        A clean, readable file path
    """
    if not source_path:
        return ""

    try:
        path_parts = source_path.split("/")[3:]
        decoded_parts = [unquote(part) for part in path_parts]
        clean_path = "/".join(decoded_parts)
        return clean_path
    except Exception as e:
        logger.warning(f"Failed to format source path '{source_path}': {e}")
        return source_path


def format_context(context_docs: List[Any], display_source: bool = True) -> str:
    """Formats retrieved documents into a string for LLM consumption."""
    if not context_docs:
        return ""

    formatted_docs = []
    for doc in context_docs:
        if hasattr(doc, "page_content"):
            content = doc.page_content
            source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        elif isinstance(doc, dict):
            content = doc.get("Content", doc.get("content", doc.get("text", str(doc))))
            source = doc.get("Source", doc.get("source", doc.get("Title", "")))
        else:
            content = str(doc)
            source = ""

        if display_source and source:
            formatted_source = (
                format_source_path(source) if "documents/" in source else source
            )
            formatted_docs.append(
                f"\nContent: \n\n{content}\n\nSource: {formatted_source}"
            )
        else:
            formatted_docs.append(f"\nContent: \n\n{content}")

    return "\n\n==============================================\n\n".join(formatted_docs)


def extract_blob_urls(tool_results: List[Any]) -> List[str]:
    """Extract blob URLs from tool results if present."""
    if not tool_results:
        return []
    urls: List[str] = []
    for item in tool_results:
        try:
            if isinstance(item, str):
                parsed = json.loads(item)
                urls.extend(parsed.get("blob_urls", []))
            elif isinstance(item, dict):
                urls.extend(item.get("blob_urls", []))
        except Exception:
            # Ignore malformed entries
            continue
    return urls
