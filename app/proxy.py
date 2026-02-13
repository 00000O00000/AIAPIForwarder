"""Request proxy module."""

import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import httpx
from flask import Request, Response, stream_with_context

from .models import ProviderConfig, ProviderStatus
from .provider_manager import ProviderManager

logger = logging.getLogger(__name__)

CLIENT_FORMAT_OPENAI = "openai"
CLIENT_FORMAT_CLAUDE = "claude"
CLIENT_FORMAT_GEMINI = "gemini"

PROVIDER_FORMAT_OPENAI = "openai"
PROVIDER_FORMAT_OPENAI_RESPONSE = "openai-response"
PROVIDER_FORMAT_CLAUDE = "claude"
PROVIDER_FORMAT_GEMINI = "gemini"


class OpenAIProxy:
    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager

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

        client_format = forced_client_format or self._detect_client_format(body, endpoint)
        canonical_body = self._convert_client_request_to_openai(
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

        acquired, acquire_reason = self.provider_manager.try_acquire_model_worker(model_name)
        if not acquired:
            logger.info("Skip model %s due to max_worker limit: %s", model_name, acquire_reason)
            return self._error_response_for_client(429, acquire_reason, client_format)

        release_on_close = False
        try:
            is_stream = bool(canonical_body.get("stream", False))
            priority_groups = self.provider_manager.get_providers_by_priority(model_name)
            if not priority_groups:
                return self._error_response_for_client(
                    502,
                    f"No available upstream providers for model: {model_name}",
                    client_format,
                )

            last_error = None
            for priority in sorted(priority_groups.keys()):
                providers = priority_groups[priority]
                remaining = {p.name: p.retry + 1 for p in providers}
                skipped = set()

                while True:
                    round_attempted = False
                    for provider in providers:
                        if provider.name in skipped or remaining[provider.name] <= 0:
                            continue

                        status, reason = self.provider_manager.rate_limiter.check_availability(model_name, provider)
                        if status != ProviderStatus.AVAILABLE:
                            skipped.add(provider.name)
                            logger.debug("Provider %s unavailable: %s", provider.name, reason)
                            continue

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
                            if self._should_release_worker_on_close(result):
                                self._register_worker_release_on_close(result, model_name)
                                release_on_close = True
                            return result

                        last_error = result
                        if error_type == "client_error":
                            return result
                        if error_type == "auth_error":
                            skipped.add(provider.name)

                    if not round_attempted:
                        break

            if last_error:
                return last_error
            return self._error_response_for_client(502, "No available upstream provider", client_format)
        finally:
            if not release_on_close:
                self.provider_manager.release_model_worker(model_name)

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
        if status_code in (401, 403, 429):
            return "auth_error"
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

        openai_data = self._provider_response_to_openai(provider_data, provider_format, body.get("model"))
        tokens = self._extract_tokens_from_provider_response(provider_data, provider_format, openai_data)

        if need_convert_stream:
            return self._convert_to_stream(openai_data, client_format), 200, tokens

        client_data = self._convert_openai_response_to_client(openai_data, client_format)
        return self._json_response(client_data), 200, tokens

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

            def generate():
                usage: Dict[str, int] = {}
                current_event = ""
                text_buffer = ""
                finish_reason = "stop"
                meta: Dict[str, Any] = {"model": body.get("model")}
                try:
                    if direct:
                        for chunk in upstream_response.iter_raw():
                            if chunk:
                                yield chunk
                        return

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

                        piece, finish, usage_update, meta_update = self._parse_stream_chunk(provider_format, current_event, data)
                        if piece:
                            text_buffer += piece
                        if finish:
                            finish_reason = finish
                        if usage_update:
                            usage.update(usage_update)
                        if meta_update:
                            meta.update(meta_update)

                    openai_data = self._build_openai_response_from_stream_content(text_buffer, finish_reason, usage, meta)
                    for item in self._iter_stream_payload(openai_data, client_format):
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
            return response, 200, 0
        except Exception:
            client.close()
            raise

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

        usage: Dict[str, int] = {}
        content = ""
        finish_reason = "stop"
        meta: Dict[str, Any] = {"model": model_name or body.get("model")}
        current_event = ""

        try:
            with httpx.Client(timeout=provider.timeout) as client:
                with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code != 200:
                        return Response(response.read(), status=response.status_code, content_type="application/json"), response.status_code, 0

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

                        piece, finish, usage_update, meta_update = self._parse_stream_chunk(provider_format, current_event, data)
                        if piece:
                            content += piece
                        if finish:
                            finish_reason = finish
                        if usage_update:
                            usage.update(usage_update)
                        if meta_update:
                            meta.update(meta_update)

            if not usage.get("total_tokens"):
                estimated = max(1, len(content) // 4) if content else 0
                usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", estimated),
                    "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + estimated),
                }

            openai_data = self._build_openai_response_from_stream_content(content, finish_reason, usage, meta)
            client_data = self._convert_openai_response_to_client(openai_data, client_format)
            return self._json_response(client_data), 200, usage.get("total_tokens", 0)
        except Exception as exc:
            return self._error_response_for_client(502, f"Stream conversion error: {exc}", client_format), 502, 0

    def _convert_to_stream(self, response_data: dict, client_format: str) -> Response:
        return Response(
            stream_with_context(self._iter_stream_payload(response_data, client_format)),
            status=200,
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    def _iter_stream_payload(self, response_data: dict, client_format: str) -> Iterable[str]:
        if client_format == CLIENT_FORMAT_CLAUDE:
            choice = (response_data.get("choices") or [{}])[0]
            usage = response_data.get("usage", {})
            text = self._openai_content_to_text(choice.get("message", {}).get("content", ""))

            yield "event: message_start\n"
            yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': response_data.get('id'), 'type': 'message', 'role': 'assistant', 'model': response_data.get('model'), 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': 0}}})}\n\n"
            yield "event: content_block_start\n"
            yield "data: {\"type\": \"content_block_start\", \"index\": 0, \"content_block\": {\"type\": \"text\", \"text\": \"\"}}\n\n"
            if text:
                yield "event: content_block_delta\n"
                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            yield "event: content_block_stop\n"
            yield "data: {\"type\": \"content_block_stop\", \"index\": 0}\n\n"
            yield "event: message_delta\n"
            yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': self._openai_finish_to_claude(choice.get('finish_reason')), 'stop_sequence': None}, 'usage': {'output_tokens': usage.get('completion_tokens', 0)}})}\n\n"
            yield "event: message_stop\n"
            yield "data: {\"type\": \"message_stop\"}\n\n"
            return

        if client_format == CLIENT_FORMAT_GEMINI:
            usage = response_data.get("usage", {})
            for choice in response_data.get("choices", []):
                text = self._openai_content_to_text(choice.get("message", {}).get("content", ""))
                chunk = {
                    "candidates": [
                        {
                            "index": choice.get("index", 0),
                            "content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": self._openai_finish_to_gemini(choice.get("finish_reason")),
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
            return

        for choice in response_data.get("choices", []):
            chunk = {
                "id": response_data.get("id", ""),
                "object": "chat.completion.chunk",
                "created": response_data.get("created", int(time.time())),
                "model": response_data.get("model", ""),
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": {
                            "role": "assistant",
                            "content": self._openai_content_to_text(choice.get("message", {}).get("content", "")),
                        },
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

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
            return body, endpoint
        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            return self._openai_to_provider_request_openai_response(body), "/responses"
        if provider_format == PROVIDER_FORMAT_CLAUDE:
            return self._openai_to_provider_request_claude(body), "/messages"
        if provider_format == PROVIDER_FORMAT_GEMINI:
            safe_model = quote(str(body.get("model", "")).replace("models/", ""), safe="-_.~")
            action = "streamGenerateContent" if is_stream else "generateContent"
            return self._openai_to_provider_request_gemini(body), f"/models/{safe_model}:{action}"
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

    def _parse_stream_chunk(self, provider_format: str, current_event: str, chunk: dict) -> Tuple[str, Optional[str], Dict[str, int], Dict[str, Any]]:
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
                finish_reason = self._normalize_finish_reason(delta.get("stop_reason"))
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
                    finish_reason = self._normalize_finish_reason(candidate.get("finishReason"))
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

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            text_delta += delta.get("content", "")
            if choice.get("finish_reason"):
                finish_reason = self._normalize_finish_reason(choice.get("finish_reason"))
        o_usage = chunk.get("usage", {})
        if o_usage:
            usage = {
                "prompt_tokens": o_usage.get("prompt_tokens", 0),
                "completion_tokens": o_usage.get("completion_tokens", 0),
                "total_tokens": o_usage.get("total_tokens", 0),
            }
        if chunk.get("id"):
            meta["id"] = chunk.get("id")
        if chunk.get("model"):
            meta["model"] = chunk.get("model")
        if chunk.get("created"):
            meta["created"] = chunk.get("created")
        return text_delta, finish_reason, usage, meta

    def _build_openai_response_from_stream_content(self, content: str, finish_reason: str, usage: Dict[str, int], meta: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        if completion_tokens == 0 and total_tokens > 0 and prompt_tokens == 0:
            completion_tokens = total_tokens

        return {
            "id": meta.get("id", f"chatcmpl_{int(time.time() * 1000)}"),
            "object": "chat.completion",
            "created": meta.get("created", int(time.time())),
            "model": meta.get("model", ""),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason or "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
        }

    def _extract_tokens_from_provider_response(self, provider_data: dict, provider_format: str, openai_data: Optional[dict] = None) -> int:
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

    def _provider_response_to_openai(self, data: dict, provider_format: str, model_fallback: Optional[str]) -> dict:
        if provider_format == PROVIDER_FORMAT_OPENAI:
            return data

        if provider_format == PROVIDER_FORMAT_OPENAI_RESPONSE:
            output_text = data.get("output_text")
            if isinstance(output_text, list):
                text = "".join([str(item) for item in output_text])
            elif isinstance(output_text, str):
                text = output_text
            else:
                text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") in ("output_text", "text"):
                                text += content.get("text", "")
                    elif item.get("type") == "output_text":
                        text += item.get("text", "")
            usage = data.get("usage", {})
            return {
                "id": data.get("id", f"chatcmpl_{int(time.time() * 1000)}"),
                "object": "chat.completion",
                "created": data.get("created_at", int(time.time())),
                "model": data.get("model", model_fallback or ""),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
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
                "choices": [{"index": 0, "message": message_payload, "finish_reason": self._normalize_finish_reason(data.get("stop_reason", "end_turn"))}],
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
                "choices": [{"index": 0, "message": message_payload, "finish_reason": self._normalize_finish_reason(candidate.get("finishReason", "STOP"))}],
                "usage": {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                },
            }

        return data

    def _convert_openai_response_to_client(self, openai_data: dict, client_format: str) -> dict:
        if client_format == CLIENT_FORMAT_CLAUDE:
            return self._openai_to_claude_response(openai_data)
        if client_format == CLIENT_FORMAT_GEMINI:
            return self._openai_to_gemini_response(openai_data)
        return openai_data

    def _openai_to_claude_response(self, openai_data: dict) -> dict:
        choice = (openai_data.get("choices") or [{}])[0]
        usage = openai_data.get("usage", {})
        message = choice.get("message", {})
        text = self._openai_content_to_text(message.get("content", ""))
        content_blocks = [{"type": "text", "text": text}]
        for tool_call in self._safe_list(message.get("tool_calls")):
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
        return {
            "id": openai_data.get("id", f"msg_{int(time.time() * 1000)}"),
            "type": "message",
            "role": "assistant",
            "model": openai_data.get("model", ""),
            "content": content_blocks,
            "stop_reason": self._openai_finish_to_claude(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
        }

    def _openai_to_gemini_response(self, openai_data: dict) -> dict:
        usage = openai_data.get("usage", {})
        candidates = []
        for choice in openai_data.get("choices", []):
            message = choice.get("message", {})
            parts = self._openai_content_to_gemini_parts(message.get("content", ""))
            for tool_call in self._safe_list(message.get("tool_calls")):
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
                parts.append(
                    {
                        "functionCall": {
                            "name": function.get("name", "tool"),
                            "args": parsed_arguments,
                        }
                    }
                )
            candidates.append(
                {
                    "index": choice.get("index", 0),
                    "content": {"role": "model", "parts": parts},
                    "finishReason": self._openai_finish_to_gemini(choice.get("finish_reason")),
                }
            )
        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            },
            "modelVersion": openai_data.get("model", ""),
        }

    def _detect_client_format(self, body: dict, endpoint: str) -> str:
        if endpoint != "/chat/completions":
            return CLIENT_FORMAT_OPENAI
        if "contents" in body or "generationConfig" in body or "systemInstruction" in body:
            return CLIENT_FORMAT_GEMINI
        if "anthropic_version" in body or "stop_sequences" in body or "anthropic_beta" in body:
            return CLIENT_FORMAT_CLAUDE
        if "system" in body and "messages" in body:
            return CLIENT_FORMAT_CLAUDE
        if self._looks_like_claude_messages(body.get("messages")):
            return CLIENT_FORMAT_CLAUDE
        return CLIENT_FORMAT_OPENAI

    @staticmethod
    def _looks_like_claude_messages(messages: Any) -> bool:
        if not isinstance(messages, list) or not messages:
            return False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("text", "image", "tool_use", "tool_result"):
                        return True
        return False

    def _convert_client_request_to_openai(
        self,
        body: dict,
        endpoint: str,
        client_format: str,
        route_model: Optional[str],
        route_stream: Optional[bool],
    ) -> dict:
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
            "model", "messages", "system", "max_tokens", "temperature", "top_p", "top_k", "stream", "stop_sequences", "tools", "tool_choice"
        }
        result = self._passthrough_fields(body, known_keys)
        result["model"] = body.get("model") or route_model
        result["messages"] = []
        if "system" in body:
            result["messages"].append({"role": "system", "content": self._claude_content_to_openai(body.get("system"))})
        for msg in self._safe_list(body.get("messages")):
            if not isinstance(msg, dict):
                continue
            result["messages"].extend(self._claude_message_to_openai_messages(msg))

        for key in ("max_tokens", "temperature", "top_p", "top_k"):
            if key in body:
                result[key] = body.get(key)
        if "stop_sequences" in body:
            result["stop"] = body.get("stop_sequences")
        if "tools" in body:
            result["tools"] = self._claude_tools_to_openai_tools(body.get("tools") or [])
        if "tool_choice" in body:
            result["tool_choice"] = self._convert_tool_choice_to_openai(body.get("tool_choice"))
        result["stream"] = bool(body.get("stream", route_stream if route_stream is not None else False))
        return result

    def _openai_request_from_gemini(self, body: dict, route_model: Optional[str], route_stream: Optional[bool]) -> dict:
        known_keys = {
            "model", "contents", "systemInstruction", "generationConfig", "tools", "toolConfig", "safetySettings", "cachedContent", "stream"
        }
        result = self._passthrough_fields(body, known_keys)
        result["model"] = body.get("model") or route_model
        result["messages"] = []

        system_instruction = body.get("systemInstruction")
        if system_instruction is not None:
            if isinstance(system_instruction, dict):
                content = self._gemini_parts_to_openai_content(system_instruction.get("parts", []))
            else:
                content = str(system_instruction)
            result["messages"].append({"role": "system", "content": content})

        for item in self._safe_list(body.get("contents")):
            if not isinstance(item, dict):
                continue
            result["messages"].extend(self._gemini_item_to_openai_messages(item))

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
            result["tools"] = self._gemini_tools_to_openai_tools(body.get("tools") or [])
        if body.get("toolConfig"):
            result["tool_choice"] = self._gemini_tool_config_to_openai(body.get("toolConfig"))
        if "safetySettings" in body:
            result["safetySettings"] = body.get("safetySettings")
        if "cachedContent" in body:
            result["cachedContent"] = body.get("cachedContent")
        if cfg:
            result["gemini_generation_config"] = cfg
        result["stream"] = bool(body.get("stream", route_stream if route_stream is not None else False))
        return result

    def _openai_to_provider_request_openai_response(self, body: dict) -> dict:
        known = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "stream", "tools", "tool_choice", "n", "stop"}
        result = self._passthrough_fields(body, known)
        result["model"] = body.get("model")
        result["input"] = []

        instructions = []
        for msg in self._safe_list(body.get("messages")):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role == "system":
                instructions.append(self._openai_content_to_text(msg.get("content", "")))
                continue
            result["input"].append({"role": role if role in ("user", "assistant", "developer") else "user", "content": self._openai_content_to_response_input(msg.get("content", ""))})
        if instructions:
            result["instructions"] = "\n\n".join([x for x in instructions if x])

        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is not None:
            result["max_output_tokens"] = max_tokens
        for key in ("temperature", "top_p", "stream", "tools", "tool_choice", "stop"):
            if key in body:
                result[key] = body.get(key)
        return result

    def _openai_to_provider_request_claude(self, body: dict) -> dict:
        known = {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "stream", "stop", "tools", "tool_choice", "metadata", "system"}
        result = self._passthrough_fields(body, known)
        result["model"] = body.get("model")

        system_blocks, messages = self._openai_messages_to_claude(body.get("messages", []))
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
            result["tools"] = self._openai_tools_to_claude_tools(body.get("tools") or [])
        if "tool_choice" in body:
            result["tool_choice"] = self._convert_tool_choice_to_claude(body.get("tool_choice"))
        result.setdefault("anthropic_version", "2023-06-01")
        return result

    def _openai_to_provider_request_gemini(self, body: dict) -> dict:
        known = {"model", "messages", "stream", "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens", "n", "stop", "presence_penalty", "frequency_penalty", "tools", "tool_choice", "safetySettings", "cachedContent", "gemini_generation_config"}
        result = self._passthrough_fields(body, known)

        system_parts, contents = self._openai_messages_to_gemini(body.get("messages", []))
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
            result["tools"] = self._openai_tools_to_gemini_tools(body.get("tools") or [])
        if "tool_choice" in body:
            tool_config = self._convert_tool_choice_to_gemini_tool_config(body.get("tool_choice"))
            if tool_config:
                result["toolConfig"] = tool_config
        if "safetySettings" in body:
            result["safetySettings"] = body.get("safetySettings")
        if "cachedContent" in body:
            result["cachedContent"] = body.get("cachedContent")
        return result

    def _openai_messages_to_claude(self, messages: Any) -> Tuple[List[dict], List[dict]]:
        system_blocks: List[dict] = []
        result_messages: List[dict] = []
        for msg in self._safe_list(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_blocks.extend(self._openai_content_to_claude_blocks(content))
                continue
            if role == "tool":
                result_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", "tool"), "content": self._openai_content_to_text(content)}]})
                continue
            blocks = self._openai_content_to_claude_blocks(content)
            if role == "assistant":
                for tool_call in self._safe_list(msg.get("tool_calls")):
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

    def _openai_messages_to_gemini(self, messages: Any) -> Tuple[List[dict], List[dict]]:
        system_parts: List[dict] = []
        contents: List[dict] = []
        for msg in self._safe_list(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.extend(self._openai_content_to_gemini_parts(content))
                continue
            if role == "tool":
                parts = [{"functionResponse": {"name": msg.get("name", "tool"), "response": {"content": self._openai_content_to_text(content)}}}]
                contents.append({"role": "user", "parts": parts})
                continue

            parts = self._openai_content_to_gemini_parts(content)
            if role == "assistant":
                for tool_call in self._safe_list(msg.get("tool_calls")):
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

    def _claude_message_to_openai_messages(self, message: dict) -> List[dict]:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, list):
            return [{"role": "assistant" if role == "assistant" else "user", "content": self._claude_content_to_openai(content)}]

        text_blocks: List[dict] = []
        tool_calls: List[dict] = []
        tool_results: List[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            b_type = block.get("type")
            if b_type == "text":
                text_blocks.append({"type": "text", "text": block.get("text", "")})
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
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", "tool"),
                        "content": self._openai_content_to_text(block.get("content", "")),
                    }
                )

        messages: List[dict] = []
        if role == "assistant":
            assistant_msg: dict = {"role": "assistant"}
            if len(text_blocks) == 1 and text_blocks[0].get("type") == "text":
                assistant_msg["content"] = text_blocks[0].get("text", "")
            else:
                assistant_msg["content"] = text_blocks
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
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

    def _gemini_item_to_openai_messages(self, item: dict) -> List[dict]:
        role = "assistant" if item.get("role") == "model" else "user"
        parts = self._safe_list(item.get("parts"))
        text_parts: List[dict] = []
        tool_calls: List[dict] = []
        tool_messages: List[dict] = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text_parts.append({"type": "text", "text": part.get("text", "")})
            elif "inlineData" in part or "fileData" in part:
                converted = self._gemini_parts_to_openai_content([part])
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

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _openai_content_to_response_input(content: Any) -> List[dict]:
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

    def _claude_content_to_openai(self, content: Any) -> Any:
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
                parts.append({"type": "text", "text": block.get("text", "")})
            elif b_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    mime = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            elif b_type == "tool_result":
                parts.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
        if len(parts) == 1 and parts[0].get("type") == "text":
            return parts[0].get("text", "")
        return parts or ""

    def _gemini_parts_to_openai_content(self, parts: List[dict]) -> Any:
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

    def _openai_content_to_claude_blocks(self, content: Any) -> List[dict]:
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
                parsed = self._parse_data_url(image_url)
                if parsed:
                    mime_type, data = parsed
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}})
                else:
                    blocks.append({"type": "text", "text": str(image_url)})
        return blocks or [{"type": "text", "text": ""}]

    def _openai_content_to_gemini_parts(self, content: Any) -> List[dict]:
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
                parsed = self._parse_data_url(image_url)
                if parsed:
                    mime_type, data = parsed
                    parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
                else:
                    parts.append({"fileData": {"mimeType": "image/*", "fileUri": image_url}})
        return parts or [{"text": ""}]

    @staticmethod
    def _parse_data_url(value: Any) -> Optional[Tuple[str, str]]:
        if not isinstance(value, str):
            return None
        if not value.startswith("data:") or ";base64," not in value:
            return None
        prefix, data = value.split(";base64,", 1)
        mime_type = prefix[5:] or "application/octet-stream"
        return mime_type, data

    @staticmethod
    def _openai_content_to_text(content: Any) -> str:
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

    @staticmethod
    def _openai_tools_to_claude_tools(tools: List[dict]) -> List[dict]:
        result = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            result.append({"name": fn.get("name"), "description": fn.get("description"), "input_schema": fn.get("parameters", {"type": "object", "properties": {}})})
        return result

    @staticmethod
    def _claude_tools_to_openai_tools(tools: List[dict]) -> List[dict]:
        result = []
        for tool in tools:
            result.append({"type": "function", "function": {"name": tool.get("name"), "description": tool.get("description"), "parameters": tool.get("input_schema", {"type": "object", "properties": {}})}})
        return result

    @staticmethod
    def _openai_tools_to_gemini_tools(tools: List[dict]) -> List[dict]:
        declarations = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            declarations.append({"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters", {"type": "object", "properties": {}})})
        return [{"functionDeclarations": declarations}] if declarations else []

    @staticmethod
    def _gemini_tools_to_openai_tools(tools: List[dict]) -> List[dict]:
        result = []
        for group in tools:
            for declaration in group.get("functionDeclarations", []):
                result.append({"type": "function", "function": {"name": declaration.get("name"), "description": declaration.get("description"), "parameters": declaration.get("parameters", {"type": "object", "properties": {}})}})
        return result

    @staticmethod
    def _convert_tool_choice_to_claude(tool_choice: Any) -> Any:
        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "any"):
                return {"type": "auto" if tool_choice in ("auto", "none") else "any"}
            return {"type": "tool", "name": tool_choice}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function = tool_choice.get("function", {})
                return {"type": "tool", "name": function.get("name")}
            if tool_choice.get("type") in ("auto", "any"):
                return {"type": tool_choice.get("type")}
        return {"type": "auto"}

    @staticmethod
    def _convert_tool_choice_to_openai(tool_choice: Any) -> Any:
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool" and tool_choice.get("name"):
                return {"type": "function", "function": {"name": tool_choice.get("name")}}
            if tool_choice.get("type") in ("auto", "any"):
                return "auto"
        if isinstance(tool_choice, str):
            return tool_choice
        return "auto"

    @staticmethod
    def _convert_tool_choice_to_gemini_tool_config(tool_choice: Any) -> Optional[dict]:
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

    @staticmethod
    def _gemini_tool_config_to_openai(tool_config: dict) -> Any:
        mode = (tool_config or {}).get("functionCallingConfig", {}).get("mode", "AUTO")
        if mode == "NONE":
            return "none"
        if mode == "ANY":
            allowed = (tool_config or {}).get("functionCallingConfig", {}).get("allowedFunctionNames", [])
            if allowed:
                return {"type": "function", "function": {"name": allowed[0]}}
            return "required"
        return "auto"

    @staticmethod
    def _normalize_finish_reason(value: Optional[str]) -> str:
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

    @staticmethod
    def _openai_finish_to_claude(value: Optional[str]) -> str:
        normalized = OpenAIProxy._normalize_finish_reason(value)
        if normalized == "length":
            return "max_tokens"
        if normalized == "tool_calls":
            return "tool_use"
        return "end_turn"

    @staticmethod
    def _openai_finish_to_gemini(value: Optional[str]) -> str:
        normalized = OpenAIProxy._normalize_finish_reason(value)
        if normalized == "length":
            return "MAX_TOKENS"
        return "STOP"

    @staticmethod
    def _passthrough_fields(source: dict, excluded: set) -> dict:
        return {k: v for k, v in source.items() if k not in excluded}

    @staticmethod
    def _json_response(body: dict, status_code: int = 200) -> Response:
        return Response(json.dumps(body), status=status_code, content_type="application/json")

    @staticmethod
    def _is_direct_stream_compatible(client_format: str, provider_format: str) -> bool:
        return (
            (client_format == CLIENT_FORMAT_OPENAI and provider_format == PROVIDER_FORMAT_OPENAI)
            or (client_format == CLIENT_FORMAT_CLAUDE and provider_format == PROVIDER_FORMAT_CLAUDE)
            or (client_format == CLIENT_FORMAT_GEMINI and provider_format == PROVIDER_FORMAT_GEMINI)
        )

    @staticmethod
    def _should_release_worker_on_close(response: Response) -> bool:
        return bool(getattr(response, "is_streamed", False))

    def _register_worker_release_on_close(self, response: Response, model_name: str):
        released = False

        def _release_once():
            nonlocal released
            if released:
                return
            released = True
            self.provider_manager.release_model_worker(model_name)

        response.call_on_close(_release_once)

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

    def _normalize_request_body(self, body: dict) -> dict:
        normalized = body.copy()
        if "stop" in normalized:
            normalized_stop = self._normalize_stop(normalized.get("stop"))
            if normalized_stop is None:
                normalized.pop("stop", None)
            else:
                normalized["stop"] = normalized_stop
        return normalized

    @staticmethod
    def _normalize_stop(stop_value: Any) -> Optional[Any]:
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
