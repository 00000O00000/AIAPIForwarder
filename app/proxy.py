"""
代理转发模块 - 处理请求转发和响应处理
"""

import json
import time
import logging
import httpx
from typing import Generator, Optional, Dict, Any, Tuple
from flask import Request, Response, stream_with_context

from .provider_manager import ProviderManager
from .models import ProviderConfig

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
        """通用代理请求处理"""
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
        
        # 获取可用提供商
        excluded_providers = []
        last_error = None
        
        while True:
            # 选择提供商（考虑流式需求，但不严格要求）
            provider = self.provider_manager.select_provider(
                model_name, 
                exclude_providers=excluded_providers
            )
            
            if not provider:
                if last_error:
                    return last_error
                return self._error_response(
                    502, 
                    f"No available upstream providers for model: {model_name}"
                )
            
            # 检查流式支持
            need_convert_stream = False
            need_convert_non_stream = False
            
            if is_stream and not provider.stream_support:
                # 需要将非流式响应转换为流式
                need_convert_stream = True
                logger.info(f"Provider {provider.name} doesn't support stream, will convert")
            elif not is_stream and not provider.non_stream_support:
                # 需要将流式响应转换为非流式
                need_convert_non_stream = True
                logger.info(f"Provider {provider.name} doesn't support non-stream, will convert")
            
            # 尝试请求
            result, should_switch = self._try_provider(
                provider, body, endpoint, model_name,
                is_stream, need_convert_stream, need_convert_non_stream
            )
            
            if not should_switch:
                return result
            
            # 需要切换提供商
            excluded_providers.append(provider.name)
            last_error = result
            logger.info(f"Switching provider, excluded: {excluded_providers}")
    
    def _try_provider(self, provider: ProviderConfig, body: dict, 
                     endpoint: str, model_name: str,
                     is_stream: bool, need_convert_stream: bool,
                     need_convert_non_stream: bool) -> Tuple[Response, bool]:
        """
        尝试使用指定提供商发送请求
        
        Returns:
            (response, should_switch_provider)
        """
        attempt = 0
        max_attempts = provider.retry + 1
        
        # 修改请求体中的模型名为上游模型名
        upstream_body = body.copy()
        upstream_body["model"] = provider.model
        
        # 如果需要转换流式，则请求非流式
        if need_convert_stream:
            upstream_body["stream"] = False
        # 如果需要转换非流式，则请求流式
        elif need_convert_non_stream:
            upstream_body["stream"] = True
        
        while attempt < max_attempts:
            attempt += 1
            logger.info(f"Attempting request to {provider.name}, attempt {attempt}/{max_attempts}")
            
            try:
                response, status_code, tokens = self._send_request(
                    provider, upstream_body, endpoint,
                    is_stream and not need_convert_stream,
                    need_convert_stream, need_convert_non_stream
                )
                
                if 200 <= status_code < 300:
                    # 成功
                    self.provider_manager.record_success(model_name, provider.name, tokens)
                    return response, False
                
                # 错误处理
                self.provider_manager.record_error(model_name, provider.name)
                should_retry, should_switch = self.provider_manager.should_retry(
                    model_name, provider.name, status_code, attempt
                )
                
                if should_switch:
                    return response, True
                
                if not should_retry:
                    return response, False
                
                # 重试前等待
                time.sleep(min(2 ** attempt, 10))
                
            except httpx.TimeoutException:
                logger.error(f"Timeout when calling {provider.name}")
                self.provider_manager.record_error(model_name, provider.name)
                if attempt >= max_attempts:
                    return self._error_response(504, "Upstream timeout"), True
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error calling {provider.name}: {e}")
                self.provider_manager.record_error(model_name, provider.name)
                return self._error_response(502, f"Upstream error: {str(e)}"), True
        
        return self._error_response(502, "Max retries exceeded"), True
    
    def _send_request(self, provider: ProviderConfig, body: dict, 
                     endpoint: str, is_stream: bool,
                     need_convert_stream: bool,
                     need_convert_non_stream: bool) -> Tuple[Response, int, int]:
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
        
        # 添加自定义头
        if provider.custom_headers:
            headers.update(provider.custom_headers)
        
        tokens_used = 0
        
        if is_stream:
            # 流式请求
            return self._handle_stream_request(
                url, headers, body, provider.timeout
            )
        else:
            # 非流式请求
            with httpx.Client(timeout=provider.timeout) as client:
                response = client.post(url, json=body, headers=headers)
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # 提取 token 使用量
                    usage = response_data.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                    
                    if need_convert_stream:
                        # 将非流式响应转换为流式
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
    
    def _handle_stream_request(self, url: str, headers: dict, 
                               body: dict, timeout: int) -> Tuple[Response, int, int]:
        """处理流式请求"""
        
        def generate():
            tokens = 0
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", url, json=body, headers=headers) as response:
                        if response.status_code != 200:
                            # 非成功响应，直接返回
                            yield response.read()
                            return
                        
                        for line in response.iter_lines():
                            if line:
                                yield line + "\n"
                                # 尝试从流中提取 token 信息（如果有的话）
                                if line.startswith("data: ") and not line.endswith("[DONE]"):
                                    try:
                                        data = json.loads(line[6:])
                                        if "usage" in data:
                                            tokens = data["usage"].get("total_tokens", 0)
                                    except:
                                        pass
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        ), 200, 0  # 流式请求的 token 统计可能不准确
    
    def _convert_to_stream(self, response_data: dict) -> Response:
        """将非流式响应转换为流式格式"""
        
        def generate():
            # 发送内容
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
            
            # 发送结束标记
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
    
    def _convert_to_non_stream(self, provider: ProviderConfig,
                               body: dict, endpoint: str) -> Tuple[Response, int, int]:
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
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}],
                                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                                    }
                                
                                for choice in chunk.get("choices", []):
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                                    if choice.get("finish_reason"):
                                        response_data["choices"][0]["finish_reason"] = choice["finish_reason"]
                            except json.JSONDecodeError:
                                continue
            
            response_data["choices"][0]["message"]["content"] = full_content
            
            # 估算 token（简单估算）
            estimated_tokens = len(full_content) // 4
            response_data["usage"]["completion_tokens"] = estimated_tokens
            response_data["usage"]["total_tokens"] = estimated_tokens
            
            return Response(
                json.dumps(response_data),
                status=200,
                content_type="application/json"
            ), 200, estimated_tokens
            
        except Exception as e:
            return self._error_response(502, f"Stream conversion error: {e}"), 502, 0
    
    def _error_response(self, status_code: int, message: str) -> Response:
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