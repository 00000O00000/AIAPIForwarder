"""流式处理模块。

处理 SSE 流的解析、chunk 数据收集、以及客户端流式 payload 生成。
"""

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import content_utils as cu
from .format_converter import (
    CLIENT_FORMAT_CLAUDE,
    CLIENT_FORMAT_GEMINI,
    PROVIDER_FORMAT_CLAUDE,
    PROVIDER_FORMAT_GEMINI,
    PROVIDER_FORMAT_OPENAI_RESPONSE,
)


class StreamHandler:
    """SSE 流解析与客户端流式 payload 生成。"""

    # ------------------------------------------------------------------
    # 上游 chunk 解析
    # ------------------------------------------------------------------

    def parse_stream_chunk(
        self, provider_format: str, current_event: str, chunk: dict
    ) -> Tuple[str, Optional[str], Dict[str, int], Dict[str, Any]]:
        """解析上游 provider 的单个 SSE chunk。

        返回 (text_delta, finish_reason, usage, meta)。
        meta 中可能包含 `_tool_calls_delta` / `_thinking_delta`（OpenAI 格式时）。
        """
        usage: Dict[str, int] = {}
        meta: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        text_delta = ""

        if provider_format == PROVIDER_FORMAT_CLAUDE:
            if current_event == "message_start":
                message = chunk.get("message", {})
                meta["id"] = message.get("id")
                meta["model"] = message.get("model")
                c_usage = message.get("usage", {})
                usage = {
                    "prompt_tokens": c_usage.get("input_tokens", 0),
                    "completion_tokens": c_usage.get("output_tokens", 0),
                    "total_tokens": c_usage.get("input_tokens", 0) + c_usage.get("output_tokens", 0),
                }
            elif current_event == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_delta = delta.get("text", "")
            elif current_event == "message_delta":
                delta = chunk.get("delta", {})
                finish_reason = cu.normalize_finish_reason(delta.get("stop_reason"))
                c_usage = chunk.get("usage", {})
                if c_usage:
                    output_tokens = c_usage.get("output_tokens", 0)
                    usage["completion_tokens"] = output_tokens
                    usage["total_tokens"] = usage.get("prompt_tokens", 0) + output_tokens
            return text_delta, finish_reason, usage, meta

        if provider_format == PROVIDER_FORMAT_GEMINI:
            for candidate in chunk.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text_delta += part.get("text", "")
                if candidate.get("finishReason"):
                    finish_reason = cu.normalize_finish_reason(candidate.get("finishReason"))
            g_usage = chunk.get("usageMetadata", {})
            if g_usage:
                usage = {
                    "prompt_tokens": g_usage.get("promptTokenCount", 0),
                    "completion_tokens": g_usage.get("candidatesTokenCount", 0),
                    "total_tokens": g_usage.get("totalTokenCount", 0),
                }
            if chunk.get("modelVersion"):
                meta["model"] = chunk.get("modelVersion")
            return text_delta, finish_reason, usage, meta

        # Responses API 格式（openai-response）：事件类型在 data.type 中
        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            event_type = chunk.get("type", "")

            if event_type == "response.output_text.delta":
                text_delta = chunk.get("delta", "")
            elif event_type == "response.completed":
                resp = chunk.get("response", {})
                has_func = any(
                    isinstance(o, dict) and o.get("type") == "function_call"
                    for o in cu.safe_list(resp.get("output"))
                )
                finish_reason = "tool_calls" if has_func else "stop"
                resp_usage = resp.get("usage", {})
                if resp_usage:
                    usage = {
                        "prompt_tokens": resp_usage.get("input_tokens", 0),
                        "completion_tokens": resp_usage.get("output_tokens", 0),
                        "total_tokens": resp_usage.get("total_tokens", 0),
                    }
                meta["id"] = resp.get("id", "")
                meta["model"] = resp.get("model", "")
            elif event_type == "response.output_item.added":
                item = chunk.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id") or item.get("id", "")
                    meta["_tool_calls_delta"] = [{
                        "index": 0,
                        "id": call_id,
                        "function": {"name": item.get("name", ""), "arguments": ""},
                        "type": "function",
                    }]
            elif event_type == "response.function_call_arguments.delta":
                meta["_tool_calls_delta"] = [{
                    "index": 0,
                    "function": {"arguments": chunk.get("delta", "")},
                }]
            elif event_type == "response.reasoning_summary_text.delta":
                meta["_thinking_delta"] = {"content": chunk.get("delta", "")}
            elif event_type == "response.reasoning_summary_part.done":
                if chunk.get("part"):
                    meta["_thinking_delta"] = {"signature": chunk.get("item_id", "")}

            return text_delta, finish_reason, usage, meta

        # OpenAI 格式
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text_delta += delta.get("content", "")
            if choice.get("finish_reason"):
                finish_reason = cu.normalize_finish_reason(choice.get("finish_reason"))
            # 收集 tool_calls 增量数据
            if delta.get("tool_calls"):
                meta["_tool_calls_delta"] = delta["tool_calls"]
            # CCR: 收集 thinking 增量数据
            if delta.get("thinking"):
                meta["_thinking_delta"] = delta["thinking"]
        o_usage = chunk.get("usage", {})
        if o_usage:
            usage = {
                "prompt_tokens": o_usage.get("prompt_tokens", 0),
                "completion_tokens": o_usage.get("completion_tokens", 0),
                "total_tokens": o_usage.get("total_tokens", 0),
            }
            # CCR: 传递 prompt_tokens_details 以计算 cache_read_input_tokens
            if o_usage.get("prompt_tokens_details"):
                usage["prompt_tokens_details"] = o_usage["prompt_tokens_details"]
        if chunk.get("id"):
            meta["id"] = chunk.get("id")
        if chunk.get("model"):
            meta["model"] = chunk.get("model")
        if chunk.get("created"):
            meta["created"] = chunk.get("created")
        return text_delta, finish_reason, usage, meta

    # ------------------------------------------------------------------
    # 流式内容 → OpenAI 非流式响应
    # ------------------------------------------------------------------

    def build_openai_response_from_stream_content(
        self, content: str, finish_reason: str, usage: Dict[str, int], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从收集的流式内容构建 OpenAI 格式的完整响应。"""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        if completion_tokens == 0 and total_tokens > 0 and prompt_tokens == 0:
            completion_tokens = total_tokens

        message: Dict[str, Any] = {"role": "assistant", "content": content}
        # 将收集到的 tool_calls 放入 message
        tool_calls = meta.pop("_tool_calls", None)
        if tool_calls:
            message["tool_calls"] = tool_calls
        # CCR: 将收集到的 thinking 放入 message
        thinking = meta.pop("_thinking", None)
        if thinking:
            message["thinking"] = thinking

        result_usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        # CCR: 传递 prompt_tokens_details 以支持 cache_read_input_tokens
        if usage.get("prompt_tokens_details"):
            result_usage["prompt_tokens_details"] = usage["prompt_tokens_details"]

        return {
            "id": meta.get("id", f"chatcmpl_{int(time.time() * 1000)}"),
            "object": "chat.completion",
            "created": meta.get("created", int(time.time())),
            "model": meta.get("model", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop"}],
            "usage": result_usage,
        }

    # ------------------------------------------------------------------
    # 非流式响应 → 客户端 SSE 流
    # ------------------------------------------------------------------

    def iter_stream_payload(self, response_data: dict, client_format: str) -> Iterable[str]:
        """将完整的 OpenAI 响应转换为 SSE 流 payload。"""
        if client_format == CLIENT_FORMAT_CLAUDE:
            yield from self._iter_claude_stream(response_data)
            return

        if client_format == CLIENT_FORMAT_GEMINI:
            yield from self._iter_gemini_stream(response_data)
            return

        yield from self._iter_openai_stream(response_data)

    def _iter_claude_stream(self, response_data: dict) -> Iterable[str]:
        """生成 Claude 格式的 SSE 流。"""
        choice = (response_data.get("choices") or [{}])[0]
        usage = response_data.get("usage", {})
        message = choice.get("message", {})
        text = cu.openai_content_to_text(message.get("content", ""))
        tool_calls = cu.safe_list(message.get("tool_calls"))
        thinking = message.get("thinking")

        yield "event: message_start\n"
        yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': response_data.get('id'), 'type': 'message', 'role': 'assistant', 'model': response_data.get('model'), 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': 0}}})}\n\n"

        block_index = 0

        # CCR: thinking content block（在 text 之前输出）
        if isinstance(thinking, dict) and (thinking.get("content") or thinking.get("signature")):
            yield "event: content_block_start\n"
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
            if thinking.get("content"):
                yield "event: content_block_delta\n"
                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': thinking['content']}})}\n\n"
            if thinking.get("signature"):
                yield "event: content_block_delta\n"
                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'signature_delta', 'signature': thinking['signature']}})}\n\n"
            yield "event: content_block_stop\n"
            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            block_index += 1

        # 文本内容块
        if text:
            yield "event: content_block_start\n"
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield "event: content_block_delta\n"
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield "event: content_block_stop\n"
            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            block_index += 1

        # tool_use 内容块
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function", {})
            tool_id = tc.get("id", f"toolu_{int(time.time() * 1000)}_{block_index}")
            tool_name = function.get("name", "tool")
            raw_args = function.get("arguments", "{}")

            yield "event: content_block_start\n"
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}})}\n\n"
            yield "event: content_block_delta\n"
            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': raw_args}})}\n\n"
            yield "event: content_block_stop\n"
            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            block_index += 1

        # 如果没有任何内容块，输出一个空文本块
        if block_index == 0:
            yield "event: content_block_start\n"
            yield "data: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n"
            yield "event: content_block_stop\n"
            yield "data: {\"type\": \"content_block_stop\", \"index\": 0}\n\n"

        # 确定 stop_reason：有 tool_calls 时为 tool_use
        finish = choice.get("finish_reason")
        if tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = cu.openai_finish_to_claude(finish)

        # CCR: usage 中包含 cache_read_input_tokens
        prompt_tokens_details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
        input_tokens = (usage.get("prompt_tokens", 0) or 0) - cached_tokens

        yield "event: message_delta\n"
        yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'input_tokens': input_tokens, 'output_tokens': usage.get('completion_tokens', 0) or 0, 'cache_read_input_tokens': cached_tokens}})}\n\n"
        yield "event: message_stop\n"
        yield "data: {\"type\": \"message_stop\"}\n\n"

    def _iter_gemini_stream(self, response_data: dict) -> Iterable[str]:
        """生成 Gemini 格式的 SSE 流。"""
        usage = response_data.get("usage", {})
        for choice in response_data.get("choices", []):
            message = choice.get("message", {})
            text = cu.openai_content_to_text(message.get("content", ""))
            parts: list = []
            if text:
                parts.append({"text": text})

            # 将 tool_calls 转为 Gemini functionCall parts
            for tc in cu.safe_list(message.get("tool_calls")):
                if not isinstance(tc, dict):
                    continue
                function = tc.get("function", {})
                raw_args = function.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {"raw": raw_args}
                else:
                    parsed_args = raw_args
                parts.append({"functionCall": {"name": function.get("name", ""), "args": parsed_args}})

            if not parts:
                parts.append({"text": ""})

            chunk = {
                "candidates": [
                    {
                        "index": choice.get("index", 0),
                        "content": {"role": "model", "parts": parts},
                        "finishReason": cu.openai_finish_to_gemini(choice.get("finish_reason")),
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": usage.get("prompt_tokens", 0),
                    "candidatesTokenCount": usage.get("completion_tokens", 0),
                    "totalTokenCount": usage.get("total_tokens", 0),
                },
                "modelVersion": response_data.get("model", ""),
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    def _iter_openai_stream(self, response_data: dict) -> Iterable[str]:
        """生成 OpenAI 格式的 SSE 流。"""
        for choice in response_data.get("choices", []):
            message = choice.get("message", {})
            delta: dict = {
                "role": "assistant",
                "content": cu.openai_content_to_text(message.get("content", "")),
            }
            # 将 tool_calls 包含在 delta 中
            tool_calls = cu.safe_list(message.get("tool_calls"))
            if tool_calls:
                delta["tool_calls"] = tool_calls

            chunk = {
                "id": response_data.get("id", ""),
                "object": "chat.completion.chunk",
                "created": response_data.get("created", int(time.time())),
                "model": response_data.get("model", ""),
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": delta,
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Responses API 流式事件 → OpenAI Chat Completion Chunk（逐 chunk 透传）
    # ------------------------------------------------------------------

    def parse_openai_response_stream_event(
        self, data: dict, index_state: dict
    ) -> Optional[str]:
        """将单个 Responses API SSE 事件转换为 OpenAI Chat Completion Chunk SSE 行。

        对齐 CCR openai.responses.transformer.ts 的 transformResponseOut 逻辑。
        index_state 为可变字典，用于跨调用追踪 {current_index, last_event_type}。

        返回值：
        - SSE 行字符串（含 "data: ...\\n\\n"）
        - None 表示该事件无需输出
        """
        event_type = data.get("type", "")

        def _get_index(evt_type: str) -> int:
            """仅在事件类型切换时递增索引（与 CCR 的 getCurrentIndex 对齐）。"""
            if evt_type != index_state.get("last_event_type", ""):
                index_state["current_index"] = index_state.get("current_index", -1) + 1
                index_state["last_event_type"] = evt_type
            return index_state.get("current_index", 0)

        def _make_chunk(chunk_id: str, model: str, choices: list) -> str:
            obj = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": choices,
            }
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        # --- response.output_text.delta → content delta ---
        if event_type == "response.output_text.delta":
            return _make_chunk(
                data.get("item_id", ""),
                (data.get("response") or {}).get("model", ""),
                [{"index": _get_index(event_type), "delta": {"content": data.get("delta", "")}, "finish_reason": None}],
            )

        # --- response.output_item.added (function_call) → tool_calls 开始 ---
        if event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id", "")
                return _make_chunk(
                    call_id,
                    (data.get("response") or {}).get("model", ""),
                    [{
                        "index": _get_index(event_type),
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": 0,
                                "id": call_id,
                                "function": {"name": item.get("name", ""), "arguments": ""},
                                "type": "function",
                            }],
                        },
                        "finish_reason": None,
                    }],
                )
            if item.get("type") == "message":
                # 处理初始 message item 中已有的 output_text 内容
                content_parts = cu.safe_list(item.get("content"))
                texts = [p.get("text", "") for p in content_parts if isinstance(p, dict) and p.get("type") == "output_text" and p.get("text")]
                if texts:
                    combined = texts[0] if len(texts) == 1 else "".join(texts)
                    return _make_chunk(
                        item.get("id", ""),
                        (data.get("response") or {}).get("model", ""),
                        [{"index": _get_index(event_type), "delta": {"role": "assistant", "content": combined}, "finish_reason": None}],
                    )
            return None

        # --- response.function_call_arguments.delta → tool_calls arguments 增量 ---
        if event_type == "response.function_call_arguments.delta":
            return _make_chunk(
                data.get("item_id", ""),
                (data.get("response") or {}).get("model", ""),
                [{
                    "index": _get_index(event_type),
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": data.get("delta", "")}}],
                    },
                    "finish_reason": None,
                }],
            )

        # --- response.output_text.annotation.added → annotations ---
        if event_type == "response.output_text.annotation.added":
            ann = data.get("annotation", {})
            return _make_chunk(
                data.get("item_id", ""),
                (data.get("response") or {}).get("model", ""),
                [{
                    "index": _get_index(event_type),
                    "delta": {
                        "annotations": [{
                            "type": "url_citation",
                            "url_citation": {
                                "url": ann.get("url", ""),
                                "title": ann.get("title", ""),
                                "content": "",
                                "start_index": ann.get("start_index", 0),
                                "end_index": ann.get("end_index", 0),
                            },
                        }],
                    },
                    "finish_reason": None,
                }],
            )

        # --- response.reasoning_summary_text.delta → thinking content delta ---
        if event_type == "response.reasoning_summary_text.delta":
            return _make_chunk(
                data.get("item_id", ""),
                (data.get("response") or {}).get("model", ""),
                [{
                    "index": _get_index(event_type),
                    "delta": {"thinking": {"content": data.get("delta", "")}},
                    "finish_reason": None,
                }],
            )

        # --- response.reasoning_summary_part.done → thinking signature ---
        if event_type == "response.reasoning_summary_part.done" and data.get("part"):
            return _make_chunk(
                data.get("item_id", ""),
                (data.get("response") or {}).get("model", ""),
                [{
                    "index": index_state.get("current_index", 0),
                    "delta": {"thinking": {"signature": data.get("item_id", "")}},
                    "finish_reason": None,
                }],
            )

        # --- response.completed → finish_reason ---
        if event_type == "response.completed":
            resp = data.get("response", {})
            has_function_call = any(
                isinstance(o, dict) and o.get("type") == "function_call"
                for o in cu.safe_list(resp.get("output"))
            )
            finish_reason = "tool_calls" if has_function_call else "stop"
            # 同时提取 usage
            usage = resp.get("usage", {})
            chunk = {
                "id": resp.get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": resp.get("model", ""),
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            if usage:
                chunk["usage"] = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 其他事件类型忽略
        return None

