"""格式转换模块。

处理 OpenAI、Claude、Gemini 之间的请求和响应格式转换。
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from . import content_utils as cu

# 客户端格式常量
CLIENT_FORMAT_OPENAI = "openai"
CLIENT_FORMAT_CLAUDE = "claude"
CLIENT_FORMAT_GEMINI = "gemini"

# Provider 格式常量
PROVIDER_FORMAT_OPENAI = "openai"
PROVIDER_FORMAT_OPENAI_RESPONSE = "openai-response"
PROVIDER_FORMAT_CLAUDE = "claude"
PROVIDER_FORMAT_GEMINI = "gemini"


class FormatConverter:
    """请求和响应的格式转换器。"""

    # ------------------------------------------------------------------
    # 客户端格式检测
    # ------------------------------------------------------------------

    def detect_client_format(self, body: dict, endpoint: str) -> str:
        """基于端点判定客户端格式。

        Claude 和 Gemini 格式通过路由处理器的 forced_client_format 强制指定，
        此处仅处理未强制指定的情况，统一返回 OpenAI 格式。
        """
        return CLIENT_FORMAT_OPENAI

    # ------------------------------------------------------------------
    # 客户端 → OpenAI 内部格式
    # ------------------------------------------------------------------

    def convert_client_request_to_openai(
        self,
        body: dict,
        endpoint: str,
        client_format: str,
        route_model: Optional[str],
        route_stream: Optional[bool],
    ) -> dict:
        """将客户端请求转换为 OpenAI 内部格式。"""
        if endpoint != "/chat/completions":
            result = body.copy()
            if route_model and "model" not in result:
                result["model"] = route_model
            if route_stream is not None and "stream" not in result:
                result["stream"] = route_stream
            return result

        if client_format == CLIENT_FORMAT_CLAUDE:
            return self._openai_request_from_claude(body, route_model, route_stream)
        if client_format == CLIENT_FORMAT_GEMINI:
            return self._openai_request_from_gemini(body, route_model, route_stream)
        return self._openai_request_from_openai(body, route_model, route_stream)

    def _openai_request_from_openai(self, body: dict, route_model: Optional[str], route_stream: Optional[bool]) -> dict:
        result = body.copy()
        if route_model and "model" not in result:
            result["model"] = route_model
        if route_stream is not None and "stream" not in result:
            result["stream"] = route_stream
        return result

    def _openai_request_from_claude(self, body: dict, route_model: Optional[str], route_stream: Optional[bool]) -> dict:
        known_keys = {
            "model", "messages", "system", "max_tokens", "temperature", "top_p", "top_k",
            "stream", "stop_sequences", "tools", "tool_choice",
            "thinking", "metadata", "anthropic_version", "anthropic_beta",
        }
        result = cu.passthrough_fields(body, known_keys)
        result["model"] = body.get("model") or route_model
        result["messages"] = []
        if "system" in body:
            result["messages"].append({"role": "system", "content": cu.claude_content_to_openai(body.get("system"))})
        for msg in cu.safe_list(body.get("messages")):
            if not isinstance(msg, dict):
                continue
            result["messages"].extend(self.claude_message_to_openai_messages(msg))

        for key in ("max_tokens", "temperature", "top_p", "top_k"):
            if key in body:
                result[key] = body.get(key)
        if "stop_sequences" in body:
            result["stop"] = body.get("stop_sequences")
        if "tools" in body:
            result["tools"] = cu.claude_tools_to_openai_tools(body.get("tools") or [])
        if "tool_choice" in body:
            result["tool_choice"] = cu.convert_tool_choice_to_openai(body.get("tool_choice"))
        result["stream"] = bool(body.get("stream", route_stream if route_stream is not None else False))
        # thinking 字段暂存为内部标记，在转发到 Claude provider 时还原
        if "thinking" in body:
            result["_claude_thinking"] = body["thinking"]
        # metadata 作为内部标记透传
        if "metadata" in body:
            result["metadata"] = body["metadata"]
        return result

    def _openai_request_from_gemini(self, body: dict, route_model: Optional[str], route_stream: Optional[bool]) -> dict:
        known_keys = {
            "model", "contents", "systemInstruction", "generationConfig", "tools", "toolConfig", "safetySettings", "cachedContent", "stream"
        }
        result = cu.passthrough_fields(body, known_keys)
        result["model"] = body.get("model") or route_model
        result["messages"] = []

        system_instruction = body.get("systemInstruction")
        if system_instruction is not None:
            if isinstance(system_instruction, dict):
                content = cu.gemini_parts_to_openai_content(system_instruction.get("parts", []))
            else:
                content = str(system_instruction)
            result["messages"].append({"role": "system", "content": content})

        for item in cu.safe_list(body.get("contents")):
            if not isinstance(item, dict):
                continue
            result["messages"].extend(self.gemini_item_to_openai_messages(item))

        cfg = body.get("generationConfig", {}) or {}
        mapping = {
            "temperature": "temperature",
            "topP": "top_p",
            "topK": "top_k",
            "maxOutputTokens": "max_tokens",
            "candidateCount": "n",
            "stopSequences": "stop",
            "presencePenalty": "presence_penalty",
            "frequencyPenalty": "frequency_penalty",
        }
        for src, dst in mapping.items():
            if src in cfg:
                result[dst] = cfg.get(src)

        if body.get("tools"):
            result["tools"] = cu.gemini_tools_to_openai_tools(body.get("tools") or [])
        if body.get("toolConfig"):
            result["tool_choice"] = cu.gemini_tool_config_to_openai(body.get("toolConfig"))
        if "safetySettings" in body:
            result["safetySettings"] = body.get("safetySettings")
        if "cachedContent" in body:
            result["cachedContent"] = body.get("cachedContent")
        if cfg:
            result["gemini_generation_config"] = cfg
        result["stream"] = bool(body.get("stream", route_stream if route_stream is not None else False))
        return result

    # ------------------------------------------------------------------
    # OpenAI 内部格式 → Provider
    # ------------------------------------------------------------------

    def openai_to_provider_request_openai_response(self, body: dict) -> dict:
        """将 OpenAI 内部格式转换为 OpenAI Response API 格式。

        对齐 CCR openai.responses.transformer.ts 的 transformRequestIn 逻辑：
        - messages → input（包含 tool/assistant 特殊消息转换）
        - tools → Responses API 扁平结构（含 web_search / Edit 特殊处理）
        - reasoning 参数透传
        - 删除 temperature / max_tokens
        - parallel_tool_calls = false
        """
        known = {
            "model", "messages", "max_tokens", "max_completion_tokens",
            "temperature", "top_p", "stream", "tools", "tool_choice",
            "n", "stop", "_claude_thinking", "metadata", "reasoning",
        }
        result = cu.passthrough_fields(body, known)
        result["model"] = body.get("model")

        # --- reasoning 参数（CCR: {effort, summary: "detailed"}）---
        reasoning = body.get("reasoning") or body.get("_claude_thinking")
        if reasoning:
            if isinstance(reasoning, dict):
                result["reasoning"] = {
                    "effort": reasoning.get("effort", "medium"),
                    "summary": "detailed",
                }
            else:
                result["reasoning"] = {"effort": "medium", "summary": "detailed"}

        # --- 删除 temperature / max_tokens（CCR: delete） ---
        # CCR 会删除这两个字段，Responses API 不使用它们
        # temperature 不传递；max_tokens 转为 max_output_tokens
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is not None:
            result["max_output_tokens"] = max_tokens

        # --- messages → input ---
        input_items: list = []
        instructions_parts: list = []

        for msg in cu.safe_list(body.get("messages")):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # system → instructions
            if role == "system":
                if isinstance(content, list):
                    for item in content:
                        text = ""
                        if isinstance(item, str):
                            text = item
                        elif isinstance(item, dict) and "text" in item:
                            text = item.get("text", "")
                        if text:
                            input_items.append({"role": "system", "content": text})
                else:
                    instructions_parts.append(cu.openai_content_to_text(content))
                continue

            # tool → function_call_output
            if role == "tool":
                tool_output = {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": cu.openai_content_to_text(content) if not isinstance(content, str) else content,
                }
                input_items.append(tool_output)
                continue

            # assistant + tool_calls → 每个 tool_call 生成 function_call
            if role == "assistant" and msg.get("tool_calls"):
                for tc in cu.safe_list(msg.get("tool_calls")):
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "arguments": func.get("arguments", ""),
                        "name": func.get("name", ""),
                        "call_id": tc.get("id", ""),
                    })
                continue

            # user / assistant / developer 普通消息 → content type 映射
            mapped_role = role if role in ("user", "assistant", "developer") else "user"
            converted_content = self._normalize_response_api_content(content, role)
            if converted_content is not None:
                msg_item: dict = {"role": mapped_role}
                if converted_content:
                    msg_item["content"] = converted_content
                input_items.append(msg_item)

        result["input"] = input_items
        if instructions_parts:
            result["instructions"] = "\n\n".join([x for x in instructions_parts if x])

        # --- tools 转换为 Responses API 扁平结构 ---
        if "tools" in body and body["tools"]:
            result["tools"] = self._convert_tools_for_response_api(body["tools"])

        # --- 其他可透传字段 ---
        for key in ("top_p", "stream", "tool_choice", "stop"):
            if key in body:
                result[key] = body[key]

        # CCR: parallel_tool_calls = false
        result["parallel_tool_calls"] = False

        return result

    @staticmethod
    def _normalize_response_api_content(content: Any, role: str) -> Any:
        """将消息 content 转换为 Responses API 格式。

        text → input_text / output_text（取决于 role）
        image_url → input_image / output_image
        删除 cache_control
        """
        if isinstance(content, str):
            content_type = "output_text" if role == "assistant" else "input_text"
            return [{"type": content_type, "text": content}]

        if not isinstance(content, list):
            return None

        result = []
        for item in content:
            if not isinstance(item, dict):
                continue
            # 删除 cache_control
            item_type = item.get("type", "")
            if item_type in ("text", "input_text", "output_text"):
                mapped_type = "output_text" if role == "assistant" else "input_text"
                result.append({"type": mapped_type, "text": item.get("text", "")})
            elif item_type == "image_url":
                mapped_type = "output_image" if role == "assistant" else "input_image"
                image_payload: dict = {"type": mapped_type}
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_payload["image_url"] = image_url.get("url", "")
                elif isinstance(image_url, str):
                    image_payload["image_url"] = image_url
                result.append(image_payload)
            # 其他类型忽略（CCR: return null）

        return result if result else None

    @staticmethod
    def _convert_tools_for_response_api(tools: list) -> list:
        """将 OpenAI tools 列表转换为 Responses API 格式。

        对齐 CCR：
        - web_search 工具移除后添加为 {type: "web_search"}
        - WebSearch 工具删除 allowed_domains
        - Edit 工具设 strict=true 并强制 required 字段
        - 其他工具扁平化为 {type, name, description, parameters}
        """
        web_search_found = False
        converted: list = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", {})
            name = func.get("name", "")

            # 特殊处理：web_search 移出后作为内置类型追加
            if name == "web_search":
                web_search_found = True
                continue

            # 特殊处理：WebSearch 删除 allowed_domains
            params = func.get("parameters", {})
            if name == "WebSearch" and isinstance(params, dict):
                props = params.get("properties", {})
                if isinstance(props, dict):
                    props.pop("allowed_domains", None)

            # 特殊处理：Edit 工具设 strict 并强制 required
            if name == "Edit":
                converted.append({
                    "type": tool.get("type", "function"),
                    "name": name,
                    "description": func.get("description", ""),
                    "parameters": {
                        **params,
                        "required": ["file_path", "old_string", "new_string", "replace_all"],
                    },
                    "strict": True,
                })
                continue

            # 通用：扁平化
            converted.append({
                "type": tool.get("type", "function"),
                "name": name,
                "description": func.get("description", ""),
                "parameters": params,
            })

        # 追加内置 web_search 类型
        if web_search_found:
            converted.append({"type": "web_search"})

        return converted

    def openai_to_provider_request_claude(self, body: dict) -> dict:
        """将 OpenAI 内部格式转换为 Claude API 格式。"""
        known = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "stream", "stop", "tools", "tool_choice", "metadata", "system", "_claude_thinking"}
        result = cu.passthrough_fields(body, known)
        result["model"] = body.get("model")

        system_blocks, messages = self.openai_messages_to_claude(body.get("messages", []))
        result["messages"] = messages
        if system_blocks:
            if len(system_blocks) == 1 and system_blocks[0].get("type") == "text":
                result["system"] = system_blocks[0].get("text", "")
            else:
                result["system"] = system_blocks

        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        result["max_tokens"] = max_tokens if max_tokens is not None else 1024
        for key in ("temperature", "top_p", "top_k", "stream", "metadata"):
            if key in body:
                result[key] = body.get(key)
        if "stop" in body:
            stop = body.get("stop")
            result["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        if "tools" in body:
            result["tools"] = cu.openai_tools_to_claude_tools(body.get("tools") or [])
        if "tool_choice" in body:
            claude_tc = cu.convert_tool_choice_to_claude(body.get("tool_choice"))
            if isinstance(claude_tc, dict) and claude_tc.get("type") == "none":
                # Claude 不支持 none，移除 tools 和 tool_choice 以禁用工具
                result.pop("tools", None)
            else:
                result["tool_choice"] = claude_tc
        # 还原 Claude thinking 字段
        if "_claude_thinking" in body:
            result["thinking"] = body["_claude_thinking"]
        result.setdefault("anthropic_version", "2023-06-01")
        return result

    def openai_to_provider_request_gemini(self, body: dict) -> dict:
        """将 OpenAI 内部格式转换为 Gemini API 格式。"""
        known = {"model", "messages", "stream", "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens", "n", "stop", "presence_penalty", "frequency_penalty", "tools", "tool_choice", "safetySettings", "cachedContent", "gemini_generation_config", "_claude_thinking", "metadata"}
        result = cu.passthrough_fields(body, known)

        system_parts, contents = self.openai_messages_to_gemini(body.get("messages", []))
        result["contents"] = contents
        if system_parts:
            result["systemInstruction"] = {"parts": system_parts}

        cfg = dict(body.get("gemini_generation_config") or {})
        mapping = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "n": "candidateCount",
            "presence_penalty": "presencePenalty",
            "frequency_penalty": "frequencyPenalty",
        }
        for src, dst in mapping.items():
            if src in body:
                cfg[dst] = body.get(src)
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is not None:
            cfg["maxOutputTokens"] = max_tokens
        if "stop" in body:
            stop = body.get("stop")
            cfg["stopSequences"] = stop if isinstance(stop, list) else [stop]
        if cfg:
            result["generationConfig"] = cfg

        if "tools" in body:
            result["tools"] = cu.openai_tools_to_gemini_tools(body.get("tools") or [])
        if "tool_choice" in body:
            tool_config = cu.convert_tool_choice_to_gemini_tool_config(body.get("tool_choice"))
            if tool_config:
                result["toolConfig"] = tool_config
        if "safetySettings" in body:
            result["safetySettings"] = body.get("safetySettings")
        if "cachedContent" in body:
            result["cachedContent"] = body.get("cachedContent")
        return result

    # ------------------------------------------------------------------
    # 消息级转换
    # ------------------------------------------------------------------

    def openai_messages_to_claude(self, messages: Any) -> Tuple[List[dict], List[dict]]:
        """将 OpenAI 消息列表转换为 Claude 的 (system_blocks, messages)。"""
        system_blocks: List[dict] = []
        result_messages: List[dict] = []
        for msg in cu.safe_list(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_blocks.extend(cu.openai_content_to_claude_blocks(content))
                continue
            if role == "tool":
                tool_result_block: dict = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", "tool"),
                    "content": cu.openai_content_to_text(content),
                }
                if msg.get("cache_control"):
                    tool_result_block["cache_control"] = msg["cache_control"]
                result_messages.append({"role": "user", "content": [tool_result_block]})
                continue
            blocks = cu.openai_content_to_claude_blocks(content)
            if role == "assistant":
                # 还原 thinking content block（CCR: message.thinking → {type: "thinking", ...}）
                thinking = msg.get("thinking")
                if isinstance(thinking, dict) and thinking.get("signature"):
                    blocks.insert(0, {
                        "type": "thinking",
                        "thinking": thinking.get("content", ""),
                        "signature": thinking.get("signature", ""),
                    })
                for tool_call in cu.safe_list(msg.get("tool_calls")):
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {})
                    raw_arguments = function.get("arguments", {})
                    if isinstance(raw_arguments, str):
                        try:
                            parsed_arguments = json.loads(raw_arguments)
                        except Exception:
                            parsed_arguments = {"raw": raw_arguments}
                    else:
                        parsed_arguments = raw_arguments if isinstance(raw_arguments, dict) else {"raw": str(raw_arguments)}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id", f"tool_{int(time.time() * 1000)}"),
                            "name": function.get("name", "tool"),
                            "input": parsed_arguments,
                        }
                    )
            result_messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
        return system_blocks, result_messages

    def openai_messages_to_gemini(self, messages: Any) -> Tuple[List[dict], List[dict]]:
        """将 OpenAI 消息列表转换为 Gemini 的 (system_parts, contents)。"""
        system_parts: List[dict] = []
        contents: List[dict] = []
        for msg in cu.safe_list(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.extend(cu.openai_content_to_gemini_parts(content))
                continue
            if role == "tool":
                parts = [{"functionResponse": {"name": msg.get("name", "tool"), "response": {"content": cu.openai_content_to_text(content)}}}]
                contents.append({"role": "user", "parts": parts})
                continue

            parts = cu.openai_content_to_gemini_parts(content)
            if role == "assistant":
                for tool_call in cu.safe_list(msg.get("tool_calls")):
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {})
                    raw_arguments = function.get("arguments", {})
                    if isinstance(raw_arguments, str):
                        try:
                            parsed_arguments = json.loads(raw_arguments)
                        except Exception:
                            parsed_arguments = {"raw": raw_arguments}
                    else:
                        parsed_arguments = raw_arguments if isinstance(raw_arguments, dict) else {"raw": str(raw_arguments)}
                    parts.append({"functionCall": {"name": function.get("name", "tool"), "args": parsed_arguments}})
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        return system_parts, contents

    def claude_message_to_openai_messages(self, message: dict) -> List[dict]:
        """将单条 Claude 消息转换为 OpenAI 消息列表。"""
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, list):
            return [{"role": "assistant" if role == "assistant" else "user", "content": cu.claude_content_to_openai(content)}]

        text_blocks: List[dict] = []
        tool_calls: List[dict] = []
        tool_results: List[dict] = []
        thinking_data: Optional[dict] = None

        for block in content:
            if not isinstance(block, dict):
                continue
            b_type = block.get("type")
            if b_type == "text":
                text_part: dict = {"type": "text", "text": block.get("text", "")}
                if block.get("cache_control"):
                    text_part["cache_control"] = block["cache_control"]
                text_blocks.append(text_part)
            elif b_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    mime = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    text_blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            elif b_type == "tool_use":
                tool_input = block.get("input", {})
                arguments = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{int(time.time() * 1000)}"),
                        "type": "function",
                        "function": {"name": block.get("name", "tool"), "arguments": arguments},
                    }
                )
            elif b_type == "tool_result":
                tool_msg: dict = {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", "tool"),
                    "content": cu.openai_content_to_text(block.get("content", "")),
                }
                if block.get("cache_control"):
                    tool_msg["cache_control"] = block["cache_control"]
                tool_results.append(tool_msg)
            elif b_type == "thinking":
                # CCR: assistant 消息中的 thinking content block → message.thinking 对象
                if block.get("signature"):
                    thinking_data = {
                        "content": block.get("thinking", ""),
                        "signature": block.get("signature", ""),
                    }

        messages: List[dict] = []
        if role == "assistant":
            assistant_msg: dict = {"role": "assistant"}
            if len(text_blocks) == 1 and text_blocks[0].get("type") == "text":
                assistant_msg["content"] = text_blocks[0].get("text", "")
            else:
                assistant_msg["content"] = text_blocks
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if thinking_data:
                assistant_msg["thinking"] = thinking_data
            messages.append(assistant_msg)
            return messages

        if text_blocks:
            if len(text_blocks) == 1 and text_blocks[0].get("type") == "text":
                messages.append({"role": "user", "content": text_blocks[0].get("text", "")})
            else:
                messages.append({"role": "user", "content": text_blocks})
        messages.extend(tool_results)
        if not messages:
            messages.append({"role": "user", "content": ""})
        return messages

    def gemini_item_to_openai_messages(self, item: dict) -> List[dict]:
        """将单条 Gemini content item 转换为 OpenAI 消息列表。"""
        role = "assistant" if item.get("role") == "model" else "user"
        parts = cu.safe_list(item.get("parts"))
        text_parts: List[dict] = []
        tool_calls: List[dict] = []
        tool_messages: List[dict] = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text_parts.append({"type": "text", "text": part.get("text", "")})
            elif "inlineData" in part or "fileData" in part:
                converted = cu.gemini_parts_to_openai_content([part])
                if isinstance(converted, list):
                    text_parts.extend(converted)
                else:
                    text_parts.append({"type": "text", "text": str(converted)})
            elif "functionCall" in part:
                function_call = part.get("functionCall", {})
                args = function_call.get("args", {})
                arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{int(time.time() * 1000)}_{len(tool_calls)}",
                        "type": "function",
                        "function": {"name": function_call.get("name", "tool"), "arguments": arguments},
                    }
                )
            elif "functionResponse" in part:
                function_response = part.get("functionResponse", {})
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{function_response.get('name', 'tool')}",
                        "content": json.dumps(function_response.get("response", {}), ensure_ascii=False),
                    }
                )

        openai_messages: List[dict] = []
        if role == "assistant":
            assistant_msg: dict = {"role": "assistant"}
            if len(text_parts) == 1 and text_parts[0].get("type") == "text":
                assistant_msg["content"] = text_parts[0].get("text", "")
            else:
                assistant_msg["content"] = text_parts
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            openai_messages.append(assistant_msg)
            return openai_messages

        if text_parts:
            if len(text_parts) == 1 and text_parts[0].get("type") == "text":
                openai_messages.append({"role": "user", "content": text_parts[0].get("text", "")})
            else:
                openai_messages.append({"role": "user", "content": text_parts})
        openai_messages.extend(tool_messages)
        if not openai_messages:
            openai_messages.append({"role": "user", "content": ""})
        return openai_messages

    # ------------------------------------------------------------------
    # Provider 响应 → OpenAI 内部格式
    # ------------------------------------------------------------------

    def provider_response_to_openai(self, data: dict, provider_format: str, model_fallback: Optional[str]) -> dict:
        """将上游 provider 响应转换为 OpenAI 内部格式。"""
        if provider_format == PROVIDER_FORMAT_OPENAI:
            return data

        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            # 对齐 CCR openai.responses.transformer.ts 的 convertResponseToChat
            output_items = cu.safe_list(data.get("output"))

            # --- 提取文本和图片内容 ---
            text_parts: List[str] = []
            tool_calls_list: list = []
            thinking_data: Optional[dict] = None
            annotations_list: list = []

            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")

                if item_type == "message":
                    # 提取 reasoning
                    if item.get("reasoning"):
                        reasoning = item.get("reasoning", {})
                        if isinstance(reasoning, dict):
                            thinking_data = {"content": reasoning.get("content", ""), "signature": reasoning.get("signature", "")}
                        else:
                            thinking_data = {"content": str(reasoning)}
                    for content_block in cu.safe_list(item.get("content")):
                        if not isinstance(content_block, dict):
                            continue
                        block_type = content_block.get("type", "")
                        if block_type == "output_text":
                            text_parts.append(content_block.get("text", ""))
                            # 提取 annotations
                            for ann in cu.safe_list(content_block.get("annotations")):
                                if isinstance(ann, dict):
                                    annotations_list.append({
                                        "type": "url_citation",
                                        "url_citation": {
                                            "url": ann.get("url", ""),
                                            "title": ann.get("title", ""),
                                            "content": "",
                                            "start_index": ann.get("start_index", 0),
                                            "end_index": ann.get("end_index", 0),
                                        },
                                    })
                        elif block_type in ("output_image", "output_image_base64"):
                            # 图片类型暂不处理（本项目当前无需此场景）
                            pass

                elif item_type == "function_call":
                    tool_calls_list.append({
                        "id": item.get("call_id") or item.get("id", f"call_{int(time.time() * 1000)}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    })

                elif item_type == "output_text":
                    text_parts.append(item.get("text", ""))

            # --- 兜底：output_text 顶层字段 ---
            if not text_parts and not tool_calls_list:
                output_text = data.get("output_text")
                if isinstance(output_text, list):
                    text_parts = [str(x) for x in output_text]
                elif isinstance(output_text, str):
                    text_parts = [output_text]

            content_text = "".join(text_parts)

            # --- 构建 message ---
            message_payload: dict = {"role": "assistant", "content": content_text}
            if tool_calls_list:
                message_payload["tool_calls"] = tool_calls_list
            if thinking_data:
                message_payload["thinking"] = thinking_data
            if annotations_list:
                message_payload["annotations"] = annotations_list

            usage = data.get("usage", {})
            return {
                "id": data.get("id", f"chatcmpl_{int(time.time() * 1000)}"),
                "object": "chat.completion",
                "created": data.get("created_at", int(time.time())),
                "model": data.get("model", model_fallback or ""),
                "choices": [{
                    "index": 0,
                    "message": message_payload,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if tool_calls_list else "stop",
                }],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

        if provider_format == PROVIDER_FORMAT_CLAUDE:
            text = ""
            tool_calls = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_input = block.get("input", {})
                    arguments = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": block.get("id", f"call_{int(time.time() * 1000)}"),
                            "type": "function",
                            "function": {"name": block.get("name", "tool"), "arguments": arguments},
                        }
                    )
            usage = data.get("usage", {})
            message_payload: dict = {"role": "assistant", "content": text}
            if tool_calls:
                message_payload["tool_calls"] = tool_calls
            return {
                "id": data.get("id", f"chatcmpl_{int(time.time() * 1000)}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", model_fallback or ""),
                "choices": [{"index": 0, "message": message_payload, "finish_reason": cu.normalize_finish_reason(data.get("stop_reason", "end_turn"))}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
            }

        if provider_format == PROVIDER_FORMAT_GEMINI:
            candidates = data.get("candidates", [])
            candidate = candidates[0] if candidates else {}
            text = ""
            tool_calls = []
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part.get("text", "")
                elif "functionCall" in part:
                    function_call = part.get("functionCall", {})
                    arguments = function_call.get("args", {})
                    arguments = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": f"call_{int(time.time() * 1000)}_{len(tool_calls)}",
                            "type": "function",
                            "function": {"name": function_call.get("name", "tool"), "arguments": arguments},
                        }
                    )
            usage = data.get("usageMetadata", {})
            message_payload: dict = {"role": "assistant", "content": text}
            if tool_calls:
                message_payload["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl_{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("modelVersion", model_fallback or ""),
                "choices": [{"index": 0, "message": message_payload, "finish_reason": cu.normalize_finish_reason(candidate.get("finishReason", "STOP"))}],
                "usage": {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                },
            }

        return data

    # ------------------------------------------------------------------
    # OpenAI 内部格式 → 客户端响应
    # ------------------------------------------------------------------

    def convert_openai_response_to_client(self, openai_data: dict, client_format: str) -> dict:
        """将 OpenAI 内部格式响应转换为客户端格式。"""
        if client_format == CLIENT_FORMAT_CLAUDE:
            return self.openai_to_claude_response(openai_data)
        if client_format == CLIENT_FORMAT_GEMINI:
            return self.openai_to_gemini_response(openai_data)
        return openai_data

    def openai_to_claude_response(self, openai_data: dict) -> dict:
        """将 OpenAI 响应转换为 Claude 响应，包含 thinking 和 cache_read_input_tokens。"""
        choice = (openai_data.get("choices") or [{}])[0]
        usage = openai_data.get("usage", {})
        message = choice.get("message", {})
        text = cu.openai_content_to_text(message.get("content", ""))
        content_blocks: List[dict] = []
        # CCR: thinking content block 输出
        thinking = message.get("thinking")
        if isinstance(thinking, dict) and thinking.get("content"):
            content_blocks.append({
                "type": "thinking",
                "thinking": thinking.get("content", ""),
                "signature": thinking.get("signature", ""),
            })
        content_blocks.append({"type": "text", "text": text})
        tool_calls_list = cu.safe_list(message.get("tool_calls"))
        for tool_call in tool_calls_list:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except Exception:
                    parsed_arguments = {"raw": raw_arguments}
            else:
                parsed_arguments = raw_arguments if isinstance(raw_arguments, dict) else {"raw": str(raw_arguments)}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", f"tool_{int(time.time() * 1000)}"),
                    "name": function.get("name", "tool"),
                    "input": parsed_arguments,
                }
            )
        # 有 tool_calls 时 stop_reason 为 tool_use
        if tool_calls_list:
            stop_reason = "tool_use"
        else:
            stop_reason = cu.openai_finish_to_claude(choice.get("finish_reason"))
        # CCR: usage 中包含 cache_read_input_tokens
        prompt_tokens_details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
        input_tokens = max(0, (usage.get("prompt_tokens", 0) or 0) - cached_tokens)
        return {
            "id": openai_data.get("id", f"msg_{int(time.time() * 1000)}"),
            "type": "message",
            "role": "assistant",
            "model": openai_data.get("model", ""),
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": usage.get("completion_tokens", 0) or 0,
                "cache_read_input_tokens": cached_tokens,
            },
        }

    def openai_to_gemini_response(self, openai_data: dict) -> dict:
        """将 OpenAI 响应转换为 Gemini 响应。"""
        usage = openai_data.get("usage", {})
        candidates = []
        for choice in openai_data.get("choices", []):
            message = choice.get("message", {})
            text = cu.openai_content_to_text(message.get("content", ""))
            parts: List[dict] = [{"text": text}]
            # tool_calls → functionCall parts
            for tc in cu.safe_list(message.get("tool_calls")):
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {"raw": raw_args}
                else:
                    args = raw_args
                parts.append({"functionCall": {"name": func.get("name", "tool"), "args": args}})
            finish_reason = cu.openai_finish_to_gemini(choice.get("finish_reason"))
            candidates.append({
                "index": choice.get("index", 0),
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            })
        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            },
            "modelVersion": openai_data.get("model", ""),
        }

    # ------------------------------------------------------------------
    # Token 提取
    # ------------------------------------------------------------------

    def extract_tokens_from_provider_response(self, provider_data: dict, provider_format: str, openai_data: Optional[dict] = None) -> int:
        """从 provider 响应中提取总 token 数。"""
        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            return provider_data.get("usage", {}).get("total_tokens", 0)
        if provider_format == PROVIDER_FORMAT_CLAUDE:
            usage = provider_data.get("usage", {})
            return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if provider_format == PROVIDER_FORMAT_GEMINI:
            return provider_data.get("usageMetadata", {}).get("totalTokenCount", 0)
        if openai_data:
            return openai_data.get("usage", {}).get("total_tokens", 0)
        return provider_data.get("usage", {}).get("total_tokens", 0)
