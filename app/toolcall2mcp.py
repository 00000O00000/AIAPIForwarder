"""ToolCall2MCP 模块。

为不支持原生 tool_call 的逆向 API 提供基于提示词注入的工具调用能力。
通过自定义 <tooluse-special> XML 标签引导 AI 输出工具调用，再由网关解析转码。
"""

import json
import logging
import re
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# XML 标签名称，区别于原版 MCP 的 <tooluse>
TOOLUSE_TAG = "tooluse-special"

# 匹配 <tooluse-special>...</tooluse-special> 的正则，允许内容跨行
_TOOLUSE_PATTERN = re.compile(
    rf"<{TOOLUSE_TAG}>\s*(.*?)\s*</{TOOLUSE_TAG}>",
    re.DOTALL,
)


# ------------------------------------------------------------------
# 系统提示词构建
# ------------------------------------------------------------------

def build_toolcall_system_prompt(tools: List[dict]) -> str:
    """将 OpenAI 格式的 tools 列表编译为系统级提示词。

    生成的提示词描述每个工具的名称、描述、参数 schema，
    并严格界定 AI 应以 <tooluse-special> XML 格式输出工具调用。
    """
    if not tools:
        return ""

    tool_descriptions: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", {})
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        desc_parts = [f"### {name}"]
        if description:
            desc_parts.append(f"描述: {description}")
        if parameters:
            desc_parts.append(f"参数 JSON Schema:\n```json\n{json.dumps(parameters, ensure_ascii=False, indent=2)}\n```")
        tool_descriptions.append("\n".join(desc_parts))

    tools_text = "\n\n".join(tool_descriptions)

    return f"""## 工具调用协议 (ToolUse-Special)

你拥有以下可用工具。当你需要调用工具时，必须严格按照下面的格式输出：

<{TOOLUSE_TAG}>
[
  {{
    "name": "工具名称",
    "arguments": {{
      "参数名": "参数值"
    }}
  }}
]
</{TOOLUSE_TAG}>

### 格式规则

1. `<{TOOLUSE_TAG}>` 和 `</{TOOLUSE_TAG}>` 标签必须各占一行。
2. 标签内是一个标准 JSON 数组，每个元素代表一次工具调用。
3. 每次工具调用包含 `name`（字符串）和 `arguments`（JSON 对象）。
4. 可以在同一个 `<{TOOLUSE_TAG}>` 块中调用多个工具（并行调用）。
5. 如果你不需要调用工具，直接正常回复文本即可，不要输出此标签。
6. 工具调用标签可以出现在你回复文本的任意位置。标签之外的文本将作为你的正常回复。
7. 不要在标签内放置任何非 JSON 内容（如注释或 markdown）。

### 可用工具

{tools_text}"""


# ------------------------------------------------------------------
# 请求预处理：注入提示词 + 转换历史消息
# ------------------------------------------------------------------

def inject_toolcall_prompt(body: dict) -> dict:
    """对启用 toolcall2mcp 的请求进行预处理。

    1. 将 tools 定义编译为系统提示词并注入 messages
    2. 将历史消息中的 assistant+tool_calls 和 tool 消息转为纯文本
    3. 剥离 tools 和 tool_choice 字段

    Args:
        body: OpenAI 内部格式的请求体（深拷贝后操作，不修改原始数据）

    Returns:
        处理后的请求体
    """
    result = deepcopy(body)
    tools = result.get("tools")
    if not tools or not isinstance(tools, list):
        # 没有 tools，无需处理
        return result

    # 构建工具提示词
    tool_prompt = build_toolcall_system_prompt(tools)
    if not tool_prompt:
        return result

    # 转换消息列表
    messages = result.get("messages", [])
    converted_messages = _convert_messages_for_toolcall2mcp(messages, tool_prompt)
    result["messages"] = converted_messages

    # 剥离 tools 相关字段
    result.pop("tools", None)
    result.pop("tool_choice", None)

    logger.debug("toolcall2mcp: 已注入工具提示词，转换 %d 条消息，剥离 tools 字段", len(messages))
    return result


def _convert_messages_for_toolcall2mcp(messages: List[dict], tool_prompt: str) -> List[dict]:
    """将消息列表中的 tool_call/tool 消息转为纯文本，并注入系统提示词。

    处理规则：
    - system 消息保留，工具提示词追加到最后一条 system 消息
    - assistant + tool_calls → 拼接文本内容和工具调用的纯文本表示
    - tool 消息 → 转为 user 消息（工具返回结果的纯文本表示）
    - 其他消息原样保留
    """
    converted: List[dict] = []
    has_system = False
    tool_prompt_injected = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        if role == "system":
            has_system = True
            converted.append(msg)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            converted.append(_convert_assistant_tool_calls_to_text(msg))
            continue

        if role == "tool":
            converted.append(_convert_tool_result_to_text(msg))
            continue

        converted.append(msg)

    # 注入工具提示词到系统消息
    if has_system:
        # 追加到最后一条 system 消息
        for i in range(len(converted) - 1, -1, -1):
            if converted[i].get("role") == "system":
                original_content = _extract_text_content(converted[i].get("content", ""))
                converted[i] = {
                    "role": "system",
                    "content": f"{original_content}\n\n{tool_prompt}" if original_content else tool_prompt,
                }
                tool_prompt_injected = True
                break

    if not tool_prompt_injected:
        # 没有 system 消息，在消息列表开头插入
        converted.insert(0, {"role": "system", "content": tool_prompt})

    return converted


def _convert_assistant_tool_calls_to_text(msg: dict) -> dict:
    """将 assistant + tool_calls 消息转换为纯文本 assistant 消息。"""
    text_parts: List[str] = []

    # 保留原始文本内容
    original_text = _extract_text_content(msg.get("content", ""))
    if original_text:
        text_parts.append(original_text)

    # 将 tool_calls 转为 <tooluse-special> 格式文本
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        calls_data = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {"raw": raw_args}
            else:
                parsed_args = raw_args

            calls_data.append({
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": func.get("name", ""),
                "arguments": parsed_args,
            })

        if calls_data:
            calls_json = json.dumps(calls_data, ensure_ascii=False, indent=2)
            text_parts.append(f"<{TOOLUSE_TAG}>\n{calls_json}\n</{TOOLUSE_TAG}>")

    return {"role": "assistant", "content": "\n".join(text_parts)}


def _convert_tool_result_to_text(msg: dict) -> dict:
    """将 tool 结果消息转换为 user 消息的纯文本表示。"""
    tool_call_id = msg.get("tool_call_id", "unknown")
    content = _extract_text_content(msg.get("content", ""))

    text = f"[工具返回结果]\n工具调用ID: {tool_call_id}\n返回内容:\n{content}"
    return {"role": "user", "content": text}


def _extract_text_content(content: Any) -> str:
    """从 OpenAI 格式的 content 中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


# ------------------------------------------------------------------
# 响应后处理：解析 <tooluse-special> → tool_calls
# ------------------------------------------------------------------

def parse_tooluse_special(text: str) -> Tuple[str, List[dict]]:
    """从文本中提取 <tooluse-special> 标签并解析为 tool_calls 列表。

    Args:
        text: AI 回复的完整文本

    Returns:
        (clean_text, tool_calls)
        - clean_text: 移除 <tooluse-special> 标签后的文本
        - tool_calls: OpenAI 格式的 tool_calls 列表
    """
    if not text or TOOLUSE_TAG not in text:
        return text or "", []

    tool_calls: List[dict] = []
    matches = _TOOLUSE_PATTERN.findall(text)

    for match in matches:
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError as e:
            logger.warning("toolcall2mcp: JSON 解析失败: %s，原始内容: %.200s", e, match)
            continue

        # 支持单个对象或数组
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            logger.warning("toolcall2mcp: 解析结果不是列表或对象，跳过")
            continue

        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if not name:
                continue

            arguments = item.get("arguments", {})
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments, ensure_ascii=False)
            elif isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = json.dumps(arguments, ensure_ascii=False)

            call_id = item.get("id", f"call_{uuid.uuid4().hex[:12]}")

            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str,
                },
            })

    # 清理文本：移除完整的 <tooluse-special>...</tooluse-special> 块
    clean_text = _TOOLUSE_PATTERN.sub("", text).strip()

    if tool_calls:
        logger.debug("toolcall2mcp: 从回复中解析出 %d 个工具调用", len(tool_calls))

    return clean_text, tool_calls


def apply_toolcall_to_openai_response(openai_data: dict) -> dict:
    """对 OpenAI 格式的响应数据进行后处理：解析 <tooluse-special> 并转码为 tool_calls。

    如果响应的 assistant 消息文本中包含 <tooluse-special> 标签，
    将其解析为 tool_calls 并写入响应数据，同时更新 finish_reason。

    Args:
        openai_data: OpenAI 格式的响应数据（会被原地修改）

    Returns:
        修改后的 openai_data
    """
    choices = openai_data.get("choices")
    if not choices or not isinstance(choices, list):
        return openai_data

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            continue

        content = message.get("content", "")
        if not isinstance(content, str):
            continue

        clean_text, tool_calls = parse_tooluse_special(content)

        if tool_calls:
            message["content"] = clean_text
            # 合并工具调用（保留已有的 tool_calls）
            existing = message.get("tool_calls", [])
            if isinstance(existing, list) and existing:
                message["tool_calls"] = existing + tool_calls
            else:
                message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"

    return openai_data
