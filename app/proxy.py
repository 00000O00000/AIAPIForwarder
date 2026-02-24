"""请求代理模块。

核心代理逻辑：请求路由、重试、流式处理、错误处理。
格式转换逻辑委托给 format_converter 和 stream_handler 模块。
"""

import json
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import httpx
from flask import Request, Response, stream_with_context

from . import content_utils as cu
from .format_converter import (
    CLIENT_FORMAT_CLAUDE,
    CLIENT_FORMAT_GEMINI,
    CLIENT_FORMAT_OPENAI,
    PROVIDER_FORMAT_CLAUDE,
    PROVIDER_FORMAT_GEMINI,
    PROVIDER_FORMAT_OPENAI,
    PROVIDER_FORMAT_OPENAI_RESPONSE,
    FormatConverter,
)
from .models import ProviderConfig
from .provider_manager import ProviderManager
from .stream_handler import StreamHandler

logger = logging.getLogger(__name__)


class OpenAIProxy:
    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager
        self._converter = FormatConverter()
        self._stream = StreamHandler()

    # ------------------------------------------------------------------
    # 入口 handler
    # ------------------------------------------------------------------

    def handle_chat_completion(self, request: Request) -> Response:
        return self._proxy_request(request, "/chat/completions")

    def handle_completion(self, request: Request) -> Response:
        return self._proxy_request(request, "/completions", forced_client_format=CLIENT_FORMAT_OPENAI)

    def handle_embeddings(self, request: Request) -> Response:
        return self._proxy_request(request, "/embeddings", forced_client_format=CLIENT_FORMAT_OPENAI)

    def handle_claude_messages(self, request: Request) -> Response:
        return self._proxy_request(request, "/chat/completions", forced_client_format=CLIENT_FORMAT_CLAUDE)

    def handle_gemini_content(self, request: Request, model_action: str) -> Response:
        route_model, route_stream = self._parse_gemini_model_action(model_action)
        return self._proxy_request(
            request,
            "/chat/completions",
            forced_client_format=CLIENT_FORMAT_GEMINI,
            route_model=route_model,
            route_stream=route_stream,
        )

    @staticmethod
    def _parse_gemini_model_action(model_action: str) -> Tuple[str, bool]:
        model = model_action
        stream = False
        if ":" in model_action:
            model, action = model_action.split(":", 1)
            stream = action == "streamGenerateContent"
        return model, stream

    # ------------------------------------------------------------------
    # 主请求路由
    # ------------------------------------------------------------------

    def _proxy_request(
        self,
        request: Request,
        endpoint: str,
        forced_client_format: Optional[str] = None,
        route_model: Optional[str] = None,
        route_stream: Optional[bool] = None,
    ) -> Response:
        error_client_format = forced_client_format or CLIENT_FORMAT_OPENAI
        try:
            body = request.get_json()
        except Exception as exc:
            return self._error_response_for_client(400, f"Invalid JSON body: {exc}", error_client_format)

        if not body:
            return self._error_response_for_client(400, "Request body is required", error_client_format)
        if not isinstance(body, dict):
            return self._error_response_for_client(400, "Request body must be a JSON object", error_client_format)

        client_format = forced_client_format or self._converter.detect_client_format(body, endpoint)
        canonical_body = self._converter.convert_client_request_to_openai(
            body=body,
            endpoint=endpoint,
            client_format=client_format,
            route_model=route_model,
            route_stream=route_stream,
        )
        canonical_body = self._normalize_request_body(canonical_body)

        model_name = canonical_body.get("model")
        if not model_name:
            return self._error_response_for_client(400, "Model field is required", client_format)

        is_stream = bool(canonical_body.get("stream", False))
        priority_groups = self.provider_manager.get_providers_by_priority(model_name)
        if not priority_groups:
            return self._error_response_for_client(
                502,
                f"No available upstream providers for model: {model_name}",
                client_format,
            )

        last_error = None
        concurrency_limited = False
        queue_overflow_factor = self.provider_manager.config_manager.global_config.queue_overflow_factor
        for priority in sorted(priority_groups.keys()):
            providers = priority_groups[priority]
            remaining = {p.name: p.retry + 1 for p in providers}
            skipped = set()

            while True:
                round_attempted = False
                blocked_by_provider_max_worker = False
                # #1: 按 weight 加权打乱 provider 顺序，让权重高的更可能排在前面
                shuffled = self._weighted_shuffle(providers)
                for provider in shuffled:
                    if provider.name in skipped or remaining[provider.name] <= 0:
                        continue

                    # #2: 原子地完成 rate_limit 检查 + worker 获取
                    acquired, acquire_reason, fail_type = (
                        self.provider_manager.check_and_acquire_provider_worker(model_name, provider)
                    )
                    if not acquired:
                        if fail_type == "rate_limited":
                            skipped.add(provider.name)
                            logger.debug("Provider %s unavailable: %s", provider.name, acquire_reason)
                        elif fail_type == "max_worker":
                            concurrency_limited = True
                            blocked_by_provider_max_worker = True
                            logger.debug("Provider %s skipped by max_worker: %s", provider.name, acquire_reason)
                        continue

                    provider_release_on_close = False
                    try:
                        provider_format = provider.format
                        direct_stream = self._is_direct_stream_compatible(client_format, provider_format)
                        force_non_stream_for_format = is_stream and provider.non_stream_support and not direct_stream
                        need_convert_stream = is_stream and (not provider.stream_support or force_non_stream_for_format)
                        need_convert_non_stream = (not is_stream) and (not provider.non_stream_support)

                        remaining[provider.name] -= 1
                        round_attempted = True

                        result, error_type = self._try_provider_once(
                            provider=provider,
                            body=canonical_body,
                            endpoint=endpoint,
                            model_name=model_name,
                            is_stream=is_stream,
                            need_convert_stream=need_convert_stream,
                            need_convert_non_stream=need_convert_non_stream,
                            client_format=client_format,
                        )
                        if error_type is None:
                            if self._should_hold_provider_worker_until_close(result):
                                self._register_provider_worker_release_on_close(
                                    response=result,
                                    release_callback=lambda m=model_name, p=provider.name: self.provider_manager.release_provider_worker(m, p),
                                    safety_timeout=provider.timeout * 3,
                                )
                                provider_release_on_close = True
                            return result

                        last_error = result
                        if error_type == "client_error":
                            return result
                        if error_type == "auth_error":
                            skipped.add(provider.name)
                    finally:
                        if not provider_release_on_close:
                            self.provider_manager.release_provider_worker(model_name, provider.name)

                if not round_attempted:
                    if blocked_by_provider_max_worker:
                        wait_timeout = self._get_priority_queue_wait_timeout(providers)
                        queued, queue_reason = self.provider_manager.wait_for_priority_capacity(
                            model_name=model_name,
                            priority=priority,
                            providers=providers,
                            wait_timeout=wait_timeout,
                            queue_overflow_factor=queue_overflow_factor,
                        )
                        if queued:
                            logger.debug(
                                "Priority %s queue wakeup for model %s: %s",
                                priority,
                                model_name,
                                queue_reason,
                            )
                            continue
                        logger.debug(
                            "Priority %s queue bypass for model %s: %s",
                            priority,
                            model_name,
                            queue_reason,
                        )
                    break

        if last_error:
            return last_error
        if concurrency_limited:
            return self._error_response_for_client(429, "All available providers are busy", client_format)
        return self._error_response_for_client(502, "No available upstream provider", client_format)

    # ------------------------------------------------------------------
    # 单 Provider 请求
    # ------------------------------------------------------------------

    def _try_provider_once(
        self,
        provider: ProviderConfig,
        body: dict,
        endpoint: str,
        model_name: str,
        is_stream: bool,
        need_convert_stream: bool,
        need_convert_non_stream: bool,
        client_format: str,
    ) -> Tuple[Response, Optional[str]]:
        upstream_body = body.copy()
        upstream_body["model"] = provider.model
        if need_convert_stream:
            upstream_body["stream"] = False
        elif need_convert_non_stream:
            upstream_body["stream"] = True

        try:
            response, status_code, tokens = self._send_request(
                provider=provider,
                body=upstream_body,
                endpoint=endpoint,
                is_stream=is_stream and not need_convert_stream,
                need_convert_stream=need_convert_stream,
                need_convert_non_stream=need_convert_non_stream,
                model_name=model_name,
                client_format=client_format,
            )
            if 200 <= status_code < 300:
                self.provider_manager.record_success(model_name, provider.name, tokens)
                return response, None
            return response, self._classify_error(status_code)
        except httpx.TimeoutException:
            return self._error_response_for_client(504, "Upstream timeout", client_format), "timeout"
        except Exception as exc:
            logger.exception("Unexpected provider error")
            return self._error_response_for_client(502, f"Upstream error: {exc}", client_format), "server_error"

    @staticmethod
    def _classify_error(status_code: int) -> str:
        if status_code == 400:
            return "client_error"
        if status_code in (401, 403):
            return "auth_error"
        if status_code == 429:
            return "rate_limited"
        return "server_error"

    def _send_request(
        self,
        provider: ProviderConfig,
        body: dict,
        endpoint: str,
        is_stream: bool,
        need_convert_stream: bool,
        need_convert_non_stream: bool,
        model_name: Optional[str],
        client_format: str,
    ) -> Tuple[Response, int, int]:
        provider_format = provider.format
        upstream_body, upstream_endpoint = self._build_upstream_request(provider, body, endpoint, is_stream)
        url = self._build_upstream_url(provider.endpoint, upstream_endpoint, provider_format, is_stream)
        headers = self._build_headers(provider)

        if is_stream:
            return self._handle_stream_request(
                url=url,
                headers=headers,
                body=upstream_body,
                timeout=provider.timeout,
                model_name=model_name,
                provider_name=provider.name,
                provider_format=provider_format,
                client_format=client_format,
            )
        if need_convert_non_stream:
            return self._convert_to_non_stream(
                provider=provider,
                body=upstream_body,
                endpoint=upstream_endpoint,
                client_format=client_format,
                provider_format=provider_format,
                model_name=body.get("model"),
            )

        with httpx.Client(timeout=provider.timeout) as client:
            response = client.post(url, json=upstream_body, headers=headers)

        if not (200 <= response.status_code < 300):
            return Response(response.content, status=response.status_code, content_type="application/json"), response.status_code, 0

        if endpoint != "/chat/completions":
            tokens = 0
            try:
                tokens = response.json().get("usage", {}).get("total_tokens", 0)
            except Exception:
                pass
            if need_convert_stream:
                try:
                    return self._convert_to_stream(response.json(), client_format), 200, tokens
                except Exception:
                    return self._error_response_for_client(
                        502,
                        "Cannot convert non-JSON response to stream",
                        client_format,
                    ), 502, 0
            return Response(response.content, status=response.status_code, content_type="application/json"), response.status_code, tokens

        try:
            provider_data = response.json()
        except Exception:
            return Response(response.content, status=response.status_code, content_type="application/json"), response.status_code, 0

        openai_data = self._converter.provider_response_to_openai(provider_data, provider_format, body.get("model"))
        tokens = self._converter.extract_tokens_from_provider_response(provider_data, provider_format, openai_data)

        if need_convert_stream:
            return self._convert_to_stream(openai_data, client_format), 200, tokens

        client_data = self._converter.convert_openai_response_to_client(openai_data, client_format)
        return self._json_response(client_data), 200, tokens

    # ------------------------------------------------------------------
    # 流式请求处理
    # ------------------------------------------------------------------

    def _handle_stream_request(
        self,
        url: str,
        headers: dict,
        body: dict,
        timeout: int,
        model_name: Optional[str],
        provider_name: Optional[str],
        provider_format: str,
        client_format: str,
    ) -> Tuple[Response, int, int]:
        client = httpx.Client(timeout=timeout)
        try:
            req = client.build_request("POST", url, json=body, headers=headers)
            upstream_response = client.send(req, stream=True)
            if upstream_response.status_code != 200:
                error_body = upstream_response.read()
                upstream_response.close()
                client.close()
                return Response(error_body, status=upstream_response.status_code, content_type="application/json"), upstream_response.status_code, 0

            provider_mgr = self.provider_manager
            direct = self._is_direct_stream_compatible(client_format, provider_format)
            stream_handler = self._stream

            def generate():
                usage: Dict[str, int] = {}
                current_event = ""
                text_buffer = ""
                finish_reason = "stop"
                meta: Dict[str, Any] = {"model": body.get("model")}
                # 用于收集流式中的 tool_calls 增量数据
                collected_tool_calls: Dict[int, dict] = {}
                # 用于收集流式中的 thinking 增量数据（CCR 兼容）
                thinking_content = ""
                thinking_signature = ""
                try:
                    if direct:
                        for chunk in upstream_response.iter_raw():
                            if chunk:
                                yield chunk
                        return

                    # --- Responses API 流式：逐 chunk 实时透传（对齐 CCR） ---
                    if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
                        index_state: Dict[str, Any] = {"current_index": -1, "last_event_type": ""}
                        for line in upstream_response.iter_lines():
                            if line is None:
                                continue
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if stripped.startswith("event:"):
                                continue
                            if not stripped.startswith("data:"):
                                continue
                            payload = stripped[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                data = json.loads(payload)
                            except Exception:
                                continue

                            # 将 Responses API 事件实时转换为 OpenAI Chat Completion Chunk
                            sse_line = stream_handler.parse_openai_response_stream_event(data, index_state)
                            if sse_line is not None:
                                # 检查是否为 response.completed （含 usage）
                                if data.get("type") == "response.completed":
                                    resp = data.get("response", {})
                                    resp_usage = resp.get("usage", {})
                                    if resp_usage:
                                        usage = {
                                            "prompt_tokens": resp_usage.get("input_tokens", 0),
                                            "completion_tokens": resp_usage.get("output_tokens", 0),
                                            "total_tokens": resp_usage.get("total_tokens", 0),
                                        }

                                yield sse_line.encode("utf-8")

                        yield b"data: [DONE]\n\n"
                        return

                    # --- 非 Responses API：保持原有逻辑（收集后一次性输出）---
                    for line in upstream_response.iter_lines():
                        if line is None:
                            continue
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("event:"):
                            current_event = stripped.split(":", 1)[1].strip()
                            continue
                        if not stripped.startswith("data:"):
                            continue

                        payload = stripped[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except Exception:
                            continue

                        piece, finish, usage_update, meta_update = stream_handler.parse_stream_chunk(provider_format, current_event, data)
                        if piece:
                            text_buffer += piece
                        if finish:
                            finish_reason = finish
                        if usage_update:
                            usage.update(usage_update)
                        if meta_update:
                            self._merge_tool_calls_delta(collected_tool_calls, meta_update)
                            thinking_content, thinking_signature = self._merge_thinking_delta(
                                thinking_content, thinking_signature, meta_update
                            )
                            meta.update(meta_update)

                    # 将收集到的 tool_calls 放入 meta
                    if collected_tool_calls:
                        meta["_tool_calls"] = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]
                    # 将收集到的 thinking 放入 meta
                    if thinking_content or thinking_signature:
                        meta["_thinking"] = {"content": thinking_content, "signature": thinking_signature}
                    openai_data = stream_handler.build_openai_response_from_stream_content(text_buffer, finish_reason, usage, meta)
                    # iter_stream_payload 内部按 client_format 做格式转换，直接传 openai 格式数据
                    for item in stream_handler.iter_stream_payload(openai_data, client_format):
                        yield item.encode("utf-8")
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")
                finally:
                    upstream_response.close()
                    client.close()
                    total_tokens = usage.get("total_tokens", 0)
                    if model_name and provider_name and total_tokens > 0:
                        provider_mgr.record_stream_tokens(model_name, provider_name, total_tokens)

            response = Response(
                stream_with_context(generate()),
                status=200,
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
            setattr(response, "_hold_provider_worker_until_close", True)
            return response, 200, 0
        except Exception:
            client.close()
            raise

    # ------------------------------------------------------------------
    # 流转非流
    # ------------------------------------------------------------------

    def _convert_to_non_stream(
        self,
        provider: ProviderConfig,
        body: dict,
        endpoint: str,
        client_format: str,
        provider_format: str,
        model_name: Optional[str],
    ) -> Tuple[Response, int, int]:
        url = self._build_upstream_url(provider.endpoint, endpoint, provider_format, is_stream=True)
        headers = self._build_headers(provider)

        # 提前初始化，避免 with 块内变量遮蔽导致潜在的 UnboundLocalError
        usage: Dict[str, int] = {}
        content = ""
        finish_reason = "stop"
        meta: Dict[str, Any] = {"model": model_name or body.get("model")}
        collected_tool_calls: Dict[int, dict] = {}
        thinking_content = ""
        thinking_signature = ""

        try:
            with httpx.Client(timeout=provider.timeout) as client:
                with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code != 200:
                        return Response(response.read(), status=response.status_code, content_type="application/json"), response.status_code, 0

                    # --- Responses API 流式：从 response.completed 事件获取完整响应 ---
                    if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
                        completed_response: Optional[dict] = None
                        for line in response.iter_lines():
                            if line is None:
                                continue
                            stripped = line.strip()
                            if not stripped or stripped.startswith("event:"):
                                continue
                            if not stripped.startswith("data:"):
                                continue
                            payload = stripped[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                data = json.loads(payload)
                            except Exception:
                                continue
                            # response.completed 包含完整的响应数据
                            if data.get("type") == "response.completed":
                                completed_response = data.get("response", {})

                        if completed_response:
                            openai_data = self._converter.provider_response_to_openai(
                                completed_response, provider_format, model_name
                            )
                        else:
                            # 兜底：如果没有收到 completed 事件
                            openai_data = {
                                "id": f"chatcmpl_{int(time.time() * 1000)}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": model_name or "",
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            }

                        usage = openai_data.get("usage", {})
                        client_data = self._converter.convert_openai_response_to_client(openai_data, client_format)
                        return self._json_response(client_data), 200, usage.get("total_tokens", 0)

                    # --- 非 Responses API：保持原有逻辑 ---
                    current_event = ""

                    for line in response.iter_lines():
                        if line is None:
                            continue
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("event:"):
                            current_event = stripped.split(":", 1)[1].strip()
                            continue
                        if not stripped.startswith("data:"):
                            continue

                        payload = stripped[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except Exception:
                            continue

                        piece, finish, usage_update, meta_update = self._stream.parse_stream_chunk(provider_format, current_event, data)
                        if piece:
                            content += piece
                        if finish:
                            finish_reason = finish
                        if usage_update:
                            usage.update(usage_update)
                        if meta_update:
                            self._merge_tool_calls_delta(collected_tool_calls, meta_update)
                            thinking_content, thinking_signature = self._merge_thinking_delta(
                                thinking_content, thinking_signature, meta_update
                            )
                            meta.update(meta_update)

            if not usage.get("total_tokens"):
                estimated = max(1, len(content) // 4) if content else 0
                usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", estimated),
                    "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + estimated),
                }

            if collected_tool_calls:
                meta["_tool_calls"] = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]
            if thinking_content or thinking_signature:
                meta["_thinking"] = {"content": thinking_content, "signature": thinking_signature}
            openai_data = self._stream.build_openai_response_from_stream_content(content, finish_reason, usage, meta)
            client_data = self._converter.convert_openai_response_to_client(openai_data, client_format)
            return self._json_response(client_data), 200, usage.get("total_tokens", 0)
        except Exception as exc:
            return self._error_response_for_client(502, f"Stream conversion error: {exc}", client_format), 502, 0

    # ------------------------------------------------------------------
    # 非流转流
    # ------------------------------------------------------------------

    def _convert_to_stream(self, response_data: dict, client_format: str) -> Response:
        return Response(
            stream_with_context(self._stream.iter_stream_payload(response_data, client_format)),
            status=200,
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # 上游请求构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_upstream_url(base: str, endpoint: str, provider_format: str, is_stream: bool) -> str:
        url = f"{base.rstrip('/')}{endpoint}"
        if provider_format == PROVIDER_FORMAT_GEMINI and is_stream:
            parsed = urlparse(url)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query.setdefault("alt", "sse")
            parsed = parsed._replace(query=urlencode(query))
            return urlunparse(parsed)
        return url

    def _build_upstream_request(self, provider: ProviderConfig, body: dict, endpoint: str, is_stream: bool) -> Tuple[dict, str]:
        provider_format = provider.format
        if endpoint != "/chat/completions":
            return body, endpoint

        if provider_format == PROVIDER_FORMAT_OPENAI:
            # 清理内部标记字段，避免泄漏到上游
            cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
            return cleaned, endpoint
        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            return self._converter.openai_to_provider_request_openai_response(body), "/responses"
        if provider_format == PROVIDER_FORMAT_CLAUDE:
            return self._converter.openai_to_provider_request_claude(body), "/messages"
        if provider_format == PROVIDER_FORMAT_GEMINI:
            safe_model = quote(str(body.get("model", "")).replace("models/", ""), safe="-_.~")
            action = "streamGenerateContent" if is_stream else "generateContent"
            return self._converter.openai_to_provider_request_gemini(body), f"/models/{safe_model}:{action}"
        return body, endpoint

    @staticmethod
    def _build_headers(provider: ProviderConfig) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if provider.format == PROVIDER_FORMAT_CLAUDE:
            headers["x-api-key"] = provider.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider.format == PROVIDER_FORMAT_GEMINI:
            headers["x-goog-api-key"] = provider.api_key
        else:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        if provider.custom_headers:
            headers.update(provider.custom_headers)
        return headers

    # ------------------------------------------------------------------
    # 请求体标准化
    # ------------------------------------------------------------------

    def _normalize_request_body(self, body: dict) -> dict:
        normalized = body.copy()
        if "stop" in normalized:
            normalized_stop = cu.normalize_stop(normalized.get("stop"))
            if normalized_stop is None:
                normalized.pop("stop", None)
            else:
                normalized["stop"] = normalized_stop
        return normalized

    # ------------------------------------------------------------------
    # Worker 管理
    # ------------------------------------------------------------------

    @staticmethod
    def _is_direct_stream_compatible(client_format: str, provider_format: str) -> bool:
        if client_format == CLIENT_FORMAT_OPENAI and provider_format == PROVIDER_FORMAT_OPENAI:
            return True
        if client_format == CLIENT_FORMAT_CLAUDE and provider_format == PROVIDER_FORMAT_CLAUDE:
            return True
        if client_format == CLIENT_FORMAT_GEMINI and provider_format == PROVIDER_FORMAT_GEMINI:
            return True
        return False

    @staticmethod
    def _should_hold_provider_worker_until_close(response: Response) -> bool:
        return bool(getattr(response, "_hold_provider_worker_until_close", False))

    def _register_provider_worker_release_on_close(
        self, response: Response, release_callback: Callable[[], None], safety_timeout: float = 180.0,
    ):
        released = threading.Event()

        def _release_once():
            if released.is_set():
                return
            released.set()
            release_callback()

        response.call_on_close(_release_once)

        # #5: 安全超时兜底——防止客户端异常断开导致 worker 永久泄漏
        def _safety_release():
            if not released.wait(timeout=max(60.0, safety_timeout)):
                logger.warning("流式 worker 安全超时释放触发（%.0fs），可能客户端连接异常", safety_timeout)
                _release_once()

        timer = threading.Thread(target=_safety_release, daemon=True)
        timer.start()

    # ------------------------------------------------------------------
    # 流式增量数据辅助合并
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_tool_calls_delta(collected: Dict[int, dict], meta_update: Dict[str, Any]) -> None:
        """将 meta_update 中的 _tool_calls_delta 合并到 collected 字典。"""
        tc_deltas = meta_update.pop("_tool_calls_delta", None)
        if not tc_deltas:
            return
        for tc in tc_deltas:
            idx = tc.get("index", 0)
            if idx not in collected:
                collected[idx] = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            entry = collected[idx]
            if tc.get("id"):
                entry["id"] = tc["id"]
            func = tc.get("function", {})
            if func.get("name"):
                entry["function"]["name"] += func["name"]
            if func.get("arguments"):
                entry["function"]["arguments"] += func["arguments"]

    @staticmethod
    def _merge_thinking_delta(
        thinking_content: str, thinking_signature: str, meta_update: Dict[str, Any],
    ) -> Tuple[str, str]:
        """将 meta_update 中的 _thinking_delta 合并，返回更新后的 (content, signature)。"""
        thinking_delta = meta_update.pop("_thinking_delta", None)
        if isinstance(thinking_delta, dict):
            if thinking_delta.get("content"):
                thinking_content += thinking_delta["content"]
            if thinking_delta.get("signature"):
                thinking_signature = thinking_delta["signature"]
        return thinking_content, thinking_signature

    # ------------------------------------------------------------------
    # 错误响应
    # ------------------------------------------------------------------

    def _error_response_for_client(self, status_code: int, message: str, client_format: str) -> Response:
        if client_format == CLIENT_FORMAT_CLAUDE:
            return self._claude_error_response(status_code, message)
        if client_format == CLIENT_FORMAT_GEMINI:
            return self._gemini_error_response(status_code, message)
        return self._error_response(status_code, message)

    @staticmethod
    def _claude_error_response(status_code: int, message: str) -> Response:
        if status_code == 429:
            error_type = "rate_limit_error"
        elif status_code in (401, 403):
            error_type = "authentication_error"
        elif status_code >= 500:
            error_type = "api_error"
        else:
            error_type = "invalid_request_error"

        body = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        return Response(json.dumps(body), status=status_code, content_type="application/json")

    @staticmethod
    def _gemini_error_response(status_code: int, message: str) -> Response:
        status_text = OpenAIProxy._http_status_to_gemini_status(status_code)
        body = {
            "error": {
                "code": status_code,
                "message": message,
                "status": status_text,
            }
        }
        return Response(json.dumps(body), status=status_code, content_type="application/json")

    @staticmethod
    def _http_status_to_gemini_status(status_code: int) -> str:
        mapping = {
            400: "INVALID_ARGUMENT",
            401: "UNAUTHENTICATED",
            403: "PERMISSION_DENIED",
            404: "NOT_FOUND",
            408: "DEADLINE_EXCEEDED",
            409: "ABORTED",
            429: "RESOURCE_EXHAUSTED",
            500: "INTERNAL",
            501: "UNIMPLEMENTED",
            502: "UNAVAILABLE",
            503: "UNAVAILABLE",
            504: "DEADLINE_EXCEEDED",
        }
        return mapping.get(status_code, "UNKNOWN")

    @staticmethod
    def _error_response(status_code: int, message: str) -> Response:
        return Response(json.dumps({"error": {"message": message, "type": "api_error", "code": status_code}}), status=status_code, content_type="application/json")

    @staticmethod
    def _json_response(body: dict, status_code: int = 200) -> Response:
        return Response(json.dumps(body), status=status_code, content_type="application/json")

    # ------------------------------------------------------------------
    # 调度工具
    # ------------------------------------------------------------------

    @staticmethod
    def _get_priority_queue_wait_timeout(providers: List[ProviderConfig]) -> float:
        """Wait timeout for a priority queue fallback cycle."""
        # #6: 使用 min(timeout) 避免单个高 timeout provider 导致排队等待过长
        timeouts = [float(p.timeout) for p in providers if getattr(p, "timeout", None)]
        if not timeouts:
            return 60.0
        return max(1.0, min(timeouts))

    @staticmethod
    def _weighted_shuffle(providers: List[ProviderConfig]) -> List[ProviderConfig]:
        """
        按 weight 加权打乱 provider 列表顺序。
        权重越高的 provider 越可能排在前面。
        """
        if len(providers) <= 1:
            return providers
        weights = [max(1, p.weight) for p in providers]
        result = []
        candidates = list(zip(providers, weights))
        while candidates:
            selected = random.choices(range(len(candidates)), weights=[w for _, w in candidates], k=1)[0]
            result.append(candidates[selected][0])
            candidates.pop(selected)
        return result
