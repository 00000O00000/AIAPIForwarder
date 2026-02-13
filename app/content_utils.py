"""内容转换工具函数模块。

提供 OpenAI、Claude、Gemini 之间内容格式、工具定义、finish_reason 等的纯函数转换。
所有函数均为无状态纯函数，不依赖任何实例状态。
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def safe_list(value: Any) -> List[Any]:
    """安全地将值转换为列表，如果不是列表则返回空列表。"""
    return value if isinstance(value, list) else []


def passthrough_fields(source: dict, excluded: set) -> dict:
    """从字典中过滤出未被排除的字段。"""
    return {k: v for k, v in source.items() if k not in excluded}


def normalize_stop(stop_value: Any) -> Optional[Any]:
    """标准化 stop 字段值。"""
    if stop_value is None:
        return None
    if isinstance(stop_value, str):
        return stop_value
    if isinstance(stop_value, (list, tuple, set)):
        stops = [item for item in stop_value if isinstance(item, str)]
        if not stops:
            return None
        if len(stops) == 1:
            return stops[0]
        return stops
    return None


# ---------------------------------------------------------------------------
# data: URL 解析
# ---------------------------------------------------------------------------

def parse_data_url(value: Any) -> Optional[Tuple[str, str]]:
    """解析 data: URL，返回 (mime_type, base64_data)。"""
    if not isinstance(value, str):
        return None
    if not value.startswith("data:") or ";base64," not in value:
        return None
    prefix, data = value.split(";base64,", 1)
    mime_type = prefix[5:] or "application/octet-stream"
    return mime_type, data


# ---------------------------------------------------------------------------
# 内容提取与转换
# ---------------------------------------------------------------------------

def openai_content_to_text(content: Any) -> str:
    """从 OpenAI content（字符串或 parts 列表）中提取纯文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    result = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in ("text", "input_text", "output_text"):
                result.append(part.get("text", ""))
            elif "text" in part:
                result.append(part.get("text", ""))
    return "".join(result)


def openai_content_to_response_input(content: Any) -> List[dict]:
    """将 OpenAI content 转换为 Response API 的 input 格式。"""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    result = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        p_type = part.get("type")
        if p_type in ("text", "input_text", "output_text"):
            result.append({"type": "input_text", "text": part.get("text", "")})
        elif p_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            result.append({"type": "input_image", "image_url": image_url})
    return result or [{"type": "input_text", "text": ""}]


# ---------------------------------------------------------------------------
# Claude ↔ OpenAI 内容转换
# ---------------------------------------------------------------------------

def claude_content_to_openai(content: Any) -> Any:
    """将 Claude content 转换为 OpenAI content，保留 cache_control。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: List[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        b_type = block.get("type")
        if b_type == "text":
            part: dict = {"type": "text", "text": block.get("text", "")}
            if block.get("cache_control"):
                part["cache_control"] = block["cache_control"]
            parts.append(part)
        elif b_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                mime = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        elif b_type == "tool_result":
            parts.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
    if len(parts) == 1 and parts[0].get("type") == "text" and not parts[0].get("cache_control"):
        return parts[0].get("text", "")
    return parts or ""


def openai_content_to_claude_blocks(content: Any) -> List[dict]:
    """将 OpenAI content 转换为 Claude content blocks，保留 cache_control。"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks: List[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        p_type = part.get("type")
        if p_type in ("text", "input_text", "output_text"):
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif p_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            parsed = parse_data_url(image_url)
            if parsed:
                mime_type, data = parsed
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}})
            else:
                blocks.append({"type": "text", "text": str(image_url)})
    return blocks or [{"type": "text", "text": ""}]


# ---------------------------------------------------------------------------
# Gemini ↔ OpenAI 内容转换
# ---------------------------------------------------------------------------

def gemini_parts_to_openai_content(parts: List[dict]) -> Any:
    """将 Gemini parts 转换为 OpenAI content。"""
    if not isinstance(parts, list):
        return str(parts)
    result: List[dict] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            result.append({"type": "text", "text": part.get("text", "")})
        elif "inlineData" in part:
            inline_data = part.get("inlineData", {})
            mime = inline_data.get("mimeType", "application/octet-stream")
            data = inline_data.get("data", "")
            if mime.startswith("image/"):
                result.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            else:
                result.append({"type": "text", "text": f"[binary:{mime}]"})
        elif "fileData" in part:
            file_data = part.get("fileData", {})
            uri = file_data.get("fileUri", "")
            mime = file_data.get("mimeType", "")
            if mime.startswith("image/"):
                result.append({"type": "image_url", "image_url": {"url": uri}})
            else:
                result.append({"type": "text", "text": uri})
        elif "functionCall" in part or "functionResponse" in part:
            result.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
    if len(result) == 1 and result[0].get("type") == "text":
        return result[0].get("text", "")
    return result or ""


def openai_content_to_gemini_parts(content: Any) -> List[dict]:
    """将 OpenAI content 转换为 Gemini parts。"""
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]
    parts: List[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        p_type = part.get("type")
        if p_type in ("text", "input_text", "output_text"):
            parts.append({"text": part.get("text", "")})
        elif p_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            parsed = parse_data_url(image_url)
            if parsed:
                mime_type, data = parsed
                parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
            else:
                parts.append({"fileData": {"mimeType": "image/*", "fileUri": image_url}})
    return parts or [{"text": ""}]


# ---------------------------------------------------------------------------
# 工具定义转换
# ---------------------------------------------------------------------------

def openai_tools_to_claude_tools(tools: List[dict]) -> List[dict]:
    """将 OpenAI tools 转换为 Claude tools。"""
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        result.append({
            "name": fn.get("name"),
            "description": fn.get("description"),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def claude_tools_to_openai_tools(tools: List[dict]) -> List[dict]:
    """将 Claude tools 转换为 OpenAI tools。"""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def openai_tools_to_gemini_tools(tools: List[dict]) -> List[dict]:
    """将 OpenAI tools 转换为 Gemini tools。"""
    declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        declarations.append({
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return [{"functionDeclarations": declarations}] if declarations else []


def gemini_tools_to_openai_tools(tools: List[dict]) -> List[dict]:
    """将 Gemini tools 转换为 OpenAI tools。"""
    result = []
    for group in tools:
        for declaration in group.get("functionDeclarations", []):
            result.append({
                "type": "function",
                "function": {
                    "name": declaration.get("name"),
                    "description": declaration.get("description"),
                    "parameters": declaration.get("parameters", {"type": "object", "properties": {}}),
                },
            })
    return result


# ---------------------------------------------------------------------------
# tool_choice 转换
# ---------------------------------------------------------------------------

def convert_tool_choice_to_claude(tool_choice: Any) -> Any:
    """将 OpenAI tool_choice 转换为 Claude tool_choice。"""
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "any"):
            return {"type": "auto" if tool_choice == "auto" else "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function", {})
            return {"type": "tool", "name": function.get("name")}
        if tool_choice.get("type") in ("auto", "any"):
            return {"type": tool_choice.get("type")}
    return {"type": "auto"}


def convert_tool_choice_to_openai(tool_choice: Any) -> Any:
    """将 Claude tool_choice 转换为 OpenAI tool_choice。"""
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice.get("name")}}
        if tool_choice.get("type") in ("auto", "any"):
            return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    return "auto"


def convert_tool_choice_to_gemini_tool_config(tool_choice: Any) -> Optional[dict]:
    """将 OpenAI tool_choice 转换为 Gemini toolConfig。"""
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("function", {}).get("name")
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def gemini_tool_config_to_openai(tool_config: dict) -> Any:
    """将 Gemini toolConfig 转换为 OpenAI tool_choice。"""
    mode = (tool_config or {}).get("functionCallingConfig", {}).get("mode", "AUTO")
    if mode == "NONE":
        return "none"
    if mode == "ANY":
        allowed = (tool_config or {}).get("functionCallingConfig", {}).get("allowedFunctionNames", [])
        if allowed:
            return {"type": "function", "function": {"name": allowed[0]}}
        return "required"
    return "auto"


# ---------------------------------------------------------------------------
# finish_reason 转换
# ---------------------------------------------------------------------------

def normalize_finish_reason(value: Optional[str]) -> str:
    """将各 provider 的 finish_reason 标准化为 OpenAI 格式。"""
    if value is None:
        return "stop"
    normalized = str(value).lower()
    if normalized in ("end_turn", "stop", "stopped", "finish", "finished"):
        return "stop"
    if normalized in ("max_tokens", "length", "token_limit", "max_output_tokens"):
        return "length"
    if normalized in ("tool_use", "tool_calls", "function_call"):
        return "tool_calls"
    return normalized


def openai_finish_to_claude(value: Optional[str]) -> str:
    """将 OpenAI finish_reason 映射为 Claude stop_reason。"""
    normalized = normalize_finish_reason(value)
    if normalized == "length":
        return "max_tokens"
    if normalized == "tool_calls":
        return "tool_use"
    return "end_turn"


def openai_finish_to_gemini(value: Optional[str]) -> str:
    """将 OpenAI finish_reason 映射为 Gemini finishReason。"""
    normalized = normalize_finish_reason(value)
    if normalized == "length":
        return "MAX_TOKENS"
    return "STOP"
