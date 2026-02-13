"""General helper utilities."""

import hashlib
import uuid
from typing import Any, List, Mapping


def generate_request_id() -> str:
    """Generate a short unique request id."""
    return hashlib.md5(uuid.uuid4().bytes).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Rough token estimate using 4 chars ~= 1 token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: List[Mapping[str, Any]]) -> int:
    """Estimate total tokens for message list content fields."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += estimate_tokens(item["text"])
    return total


def safe_int(value: Any, default: int = 0) -> int:
    """Safely parse int with fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
