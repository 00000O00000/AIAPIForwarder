"""
代理转发模块 - 处理请求转发和响应处理
"""

import json
import logging
import httpx
from typing import Optional, Dict, List, Tuple
from flask import Request, Response, stream_with_context

from .provider_manager import ProviderManager
from .models import ProviderConfig, ProviderStatus

logger = logging.getLogger(__name__)


class OpenAIProxy:
    """OpenAI API 代理"""
    
    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager
    
    def handle_chat_completion(self, request: Request) -> Response:
        """处理 chat completion 请求"""
        return self._proxy_request(request, "/chat/completions")
    
    def handle_completion(self, request: Request) -> Response:
        """处理 completion 请求"""
        return self._proxy_request(request, "/completions")
    
    def handle_embeddings(self, request: Request) -> Response:
        """处理 embeddings 请求"""
        return self._proxy_request(request, "/embeddings")
    
    def _proxy_request(self, request: Request, endpoint: str) -> Response:
        """
        通用代理请求处理
        
        故障转移逻辑：
        1. 按优先级分组，先尝试高优先级组（数字越小优先级越高）
        2. 组内轮询：所有 provider 各尝试一次为一轮，循环多轮直到重试次数耗尽
        3. 当前优先级组全部失败后，切换到下一优先级组
        4. 所有 provider 均失败时返回 502
        """
        try:
            body = request.get_json()
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")
        
        if not body:
            return self._error_response(400, "Request body is required")
        
        model_name = body.get("model")
        if not model_name:
            return self._error_response(400, "Model field is required")
        
        is_stream = body.get("stream", False)
        
        # 按优先级获取所有提供商
        priority_groups = self.provider_manager.get_providers_by_priority(model_name)
        
        if not priority_groups:
            return self._error_response(
                502, f"No available upstream providers for model: {model_name}"
            )
        
        last_error = None
        
        # 按优先级从高到低遍历（数字越小优先级越高）
        for priority in sorted(priority_groups.keys()):
            providers = priority_groups[priority]
            
            # 每个 provider 的剩余尝试次数（1次初始 + retry次重试）
            remaining = {p.name: p.retry + 1 for p in providers}
            # 已永久跳过的 provider（认证/限流等不可恢复错误）
            skipped = set()
            
            # 轮询直到所有 provider 的尝试次数耗尽
            while True:
                round_attempted = False
                
                for provider in providers:
                    if provider.name in skipped:
                        continue
                    if remaining[provider.name] <= 0:
                        continue
                    
                    # 检查限额可用性
                    status, reason = self.provider_manager.rate_limiter.check_availability(
                        model_name, provider
                    )
                    if status != ProviderStatus.AVAILABLE:
                        skipped.add(provider.name)
                        logger.debug(f"Provider {provider.name} unavailable: {reason}")
                        continue
                    
                    remaining[provider.name] -= 1
                    round_attempted = True
                    
                    logger.info(
                        f"Trying provider {provider.name} "
                        f"(priority={priority}, remaining={remaining[provider.name]})"
                    )
                    
                    # 确定流式转换需求
                    need_convert_stream = is_stream and not provider.stream_support
                    need_convert_non_stream = not is_stream and not provider.non_stream_support
                    
                    # 尝试一次请求
                    result, error_type = self._try_provider_once(
                        provider, body, endpoint, model_name,
                        is_stream, need_convert_stream, need_convert_non_stream
                    )
                    
                    if error_type is None:
                        return result
                    
                    last_error = result
                    
                    if error_type == "client_error":
                        # 400 客户端请求错误，任何 provider 都无法处理
                        return result
                    elif error_type == "auth_error":
                        # 认证/限流错误，跳过此 provider 不再重试
                        skipped.add(provider.name)
                        logger.warning(
                            f"Provider {provider.name} skipped: {error_type}"
                        )
                    else:
                        # server_error / timeout → 继续轮询下一个 provider
                        logger.warning(
                            f"Provider {provider.name} failed ({error_type}), trying next"
                        )
                
                if not round_attempted:
                    break
            
            logger.info(f"Priority group {priority} exhausted, trying next group")
        
        # 所有 provider 都失败
        if last_error:
            return last_error
        return self._error_response(502, "当前暂无可用上游供应商")
    
    def _try_provider_once(
        self, provider: ProviderConfig, body: dict,
        endpoint: str, model_name: str,
        is_stream: bool, need_convert_stream: bool,
        need_convert_non_stream: bool
    ) -> Tuple[Response, Optional[str]]:
        """
        尝试使用指定提供商发送一次请求
        
        Returns:
            (response, error_type)
            error_type: None=成功, "client_error", "auth_error", "server_error", "timeout"
        """
        upstream_body = body.copy()
        upstream_body["model"] = provider.model
        
        if need_convert_stream:
            upstream_body["stream"] = False
        elif need_convert_non_stream:
            upstream_body["stream"] = True
        
        try:
            response, status_code, tokens = self._send_request(
                provider, upstream_body, endpoint,
                is_stream and not need_convert_stream,
                need_convert_stream, need_convert_non_stream,
                model_name
            )
            
            if 200 <= status_code < 300:
                self.provider_manager.record_success(model_name, provider.name, tokens)
                return response, None
            
            error_type = self._classify_error(status_code)
            logger.warning(
                f"Provider {provider.name} returned {status_code}, "
                f"error_type={error_type}"
            )
            return response, error_type
            
        except httpx.TimeoutException:
            logger.error(f"Timeout when calling {provider.name}")
            return self._error_response(504, "Upstream timeout"), "timeout"
            
        except Exception as e:
            logger.error(f"Error calling {provider.name}: {e}")
            return self._error_response(502, f"Upstream error: {str(e)}"), "server_error"
    
    @staticmethod
    def _classify_error(status_code: int) -> str:
        """
        分类 HTTP 错误类型
        
        Returns:
            "client_error"  - 400，客户端请求错误，不应重试
            "auth_error"    - 401/403/429，认证或限流错误，跳过此 provider
            "server_error"  - 5xx 等，可重试
        """
        if status_code == 400:
            return "client_error"
        if status_code in (401, 403, 429):
            return "auth_error"
        return "server_error"
    
    def _send_request(
        self, provider: ProviderConfig, body: dict,
        endpoint: str, is_stream: bool,
        need_convert_stream: bool, need_convert_non_stream: bool,
        model_name: str = None
    ) -> Tuple[Response, int, int]:
        """
        发送实际请求
        
        Returns:
            (response, status_code, tokens_used)
        """
        url = f"{provider.endpoint.rstrip('/')}{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        
        if provider.custom_headers:
            headers.update(provider.custom_headers)
        
        if is_stream:
            # 流式请求
            return self._handle_stream_request(
                url, headers, body, provider.timeout,
                model_name, provider.name
            )
        elif need_convert_non_stream:
            # 需要将流式响应转换为非流式
            return self._convert_to_non_stream(provider, body, endpoint)
        else:
            # 标准非流式请求
            with httpx.Client(timeout=provider.timeout) as client:
                response = client.post(url, json=body, headers=headers)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    usage = response_data.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                    
                    if need_convert_stream:
                        return self._convert_to_stream(response_data), 200, tokens_used
                    
                    return Response(
                        response.content,
                        status=response.status_code,
                        content_type="application/json"
                    ), response.status_code, tokens_used
                else:
                    return Response(
                        response.content,
                        status=response.status_code,
                        content_type="application/json"
                    ), response.status_code, 0
    
    def _handle_stream_request(
        self, url: str, headers: dict, body: dict, timeout: int,
        model_name: str = None, provider_name: str = None
    ) -> Tuple[Response, int, int]:
        """
        处理流式请求
        
        - 请求伊始出错（HTTP 状态码非 200）：返回实际状态码，触发故障转移
        - 流式传输中途出错：正常中断流，不触发故障转移
        - 流结束后异步记录 token 使用量
        """
        client = httpx.Client(timeout=timeout)
        
        try:
            req = client.build_request("POST", url, json=body, headers=headers)
            upstream_response = client.send(req, stream=True)
            
            if upstream_response.status_code != 200:
                # 请求伊始就报错，返回实际状态码以触发故障转移
                error_body = upstream_response.read()
                upstream_response.close()
                client.close()
                return Response(
                    error_body,
                    status=upstream_response.status_code,
                    content_type="application/json"
                ), upstream_response.status_code, 0
            
            # 状态码 200，开始流式传输
            provider_mgr = self.provider_manager
            _model_name = model_name
            _provider_name = provider_name
            
            def generate():
                tokens = 0
                try:
                    for line in upstream_response.iter_lines():
                        if line:
                            yield line + "\n"
                            if line.startswith("data: ") and not line.endswith("[DONE]"):
                                try:
                                    data = json.loads(line[6:])
                                    if "usage" in data:
                                        tokens = data["usage"].get("total_tokens", 0)
                                except:
                                    pass
                except Exception as e:
                    # 中途出错，写入错误信息后正常结束流
                    logger.error(f"Stream interrupted: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    upstream_response.close()
                    client.close()
                    # 流结束后记录 token 使用量
                    if _model_name and _provider_name and tokens > 0:
                        provider_mgr.record_stream_tokens(
                            _model_name, _provider_name, tokens
                        )
            
            return Response(
                stream_with_context(generate()),
                status=200,
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            ), 200, 0
            
        except Exception:
            client.close()
            raise
    
    def _convert_to_stream(self, response_data: dict) -> Response:
        """将非流式响应转换为流式格式"""
        
        def generate():
            choices = response_data.get("choices", [])
            for choice in choices:
                chunk = {
                    "id": response_data.get("id", ""),
                    "object": "chat.completion.chunk",
                    "created": response_data.get("created", 0),
                    "model": response_data.get("model", ""),
                    "choices": [{
                        "index": choice.get("index", 0),
                        "delta": {
                            "role": "assistant",
                            "content": choice.get("message", {}).get("content", "")
                        },
                        "finish_reason": choice.get("finish_reason")
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return Response(
            stream_with_context(generate()),
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    def _convert_to_non_stream(
        self, provider: ProviderConfig,
        body: dict, endpoint: str
    ) -> Tuple[Response, int, int]:
        """将流式响应转换为非流式格式"""
        url = f"{provider.endpoint.rstrip('/')}{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        
        if provider.custom_headers:
            headers.update(provider.custom_headers)
        
        full_content = ""
        response_data = {}
        
        try:
            with httpx.Client(timeout=provider.timeout) as client:
                with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code != 200:
                        return Response(
                            response.read(),
                            status=response.status_code,
                            content_type="application/json"
                        ), response.status_code, 0
                    
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if not response_data:
                                    response_data = {
                                        "id": chunk.get("id", ""),
                                        "object": "chat.completion",
                                        "created": chunk.get("created", 0),
                                        "model": chunk.get("model", ""),
                                        "choices": [{
                                            "index": 0,
                                            "message": {"role": "assistant", "content": ""},
                                            "finish_reason": None
                                        }],
                                        "usage": {
                                            "prompt_tokens": 0,
                                            "completion_tokens": 0,
                                            "total_tokens": 0
                                        }
                                    }
                                
                                for choice in chunk.get("choices", []):
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                                    if choice.get("finish_reason"):
                                        response_data["choices"][0]["finish_reason"] = choice["finish_reason"]
                                
                                # 使用流中的准确 usage（如果有）
                                if "usage" in chunk:
                                    response_data["usage"] = chunk["usage"]
                            except json.JSONDecodeError:
                                continue
            
            if not response_data:
                return self._error_response(502, "Empty stream response"), 502, 0
            
            response_data["choices"][0]["message"]["content"] = full_content
            
            # 如果没有从流中获取到准确 usage，则估算
            if response_data["usage"]["total_tokens"] == 0:
                estimated_tokens = len(full_content) // 4
                response_data["usage"]["completion_tokens"] = estimated_tokens
                response_data["usage"]["total_tokens"] = estimated_tokens
            
            total_tokens = response_data["usage"].get("total_tokens", 0)
            
            return Response(
                json.dumps(response_data),
                status=200,
                content_type="application/json"
            ), 200, total_tokens
            
        except Exception as e:
            return self._error_response(502, f"Stream conversion error: {e}"), 502, 0
    
    @staticmethod
    def _error_response(status_code: int, message: str) -> Response:
        """生成错误响应"""
        error_body = {
            "error": {
                "message": message,
                "type": "api_error",
                "code": status_code
            }
        }
        return Response(
            json.dumps(error_body),
            status=status_code,
            content_type="application/json"
        )