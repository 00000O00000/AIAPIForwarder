"""General helper utilities."""

import hashlib
import json
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


def estimate_claude_request_tokens(body: Mapping[str, Any]) -> int:
    """估算 Claude Messages API 格式请求的 token 数。

    遍历 system、messages（含 text/tool_use/tool_result 等 content block）
    和 tools 定义的文本内容，使用 4 字符 ≈ 1 token 的粗略估算。
    """
    total = 0

    # system 字段：可能是字符串或 content block 列表
    system = body.get("system")
    if isinstance(system, str):
        total += estimate_tokens(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total += estimate_tokens(block.get("text", ""))

    # messages 字段
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block_type == "tool_use":
                    total += estimate_tokens(block.get("name", ""))
                    total += estimate_tokens(json.dumps(block.get("input", {})))
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        total += estimate_tokens(result_content)
                    elif isinstance(result_content, list):
                        for sub_block in result_content:
                            if isinstance(sub_block, dict):
                                total += estimate_tokens(sub_block.get("text", ""))
                    else:
                        total += estimate_tokens(str(result_content))

    # tools 字段
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        total += estimate_tokens(tool.get("name", ""))
        total += estimate_tokens(tool.get("description", ""))
        input_schema = tool.get("input_schema")
        if input_schema:
            total += estimate_tokens(json.dumps(input_schema))

    return total
