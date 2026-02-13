"""
Flask 应用主入口
"""

import os
import logging
from flask import Flask, request, jsonify
from functools import wraps

from .config import ConfigManager, UsageManager
from .provider_manager import ProviderManager
from .proxy import OpenAIProxy
from .scheduler import UsageResetScheduler

# 配置日志
handlers = [logging.StreamHandler()]
try:
    os.makedirs('/app/logs', exist_ok=True)
    handlers.append(logging.FileHandler('/app/logs/gateway.log'))
except Exception:
    pass

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# 初始化组件
config_manager = ConfigManager()
usage_manager = UsageManager()
provider_manager = ProviderManager(config_manager, usage_manager)
proxy = OpenAIProxy(provider_manager)
scheduler = UsageResetScheduler(config_manager, usage_manager)

# 创建 Flask 应用
app = Flask(__name__)


def _extract_gateway_api_key() -> str:
    """从常见位置提取调用方网关密钥。"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    for header_name in ("x-api-key", "x-goog-api-key"):
        header_value = request.headers.get(header_name)
        if header_value:
            return header_value.strip()

    query_key = request.args.get("key")
    if query_key:
        return query_key.strip()
    return ""


def require_api_key(f):
    """API Key 验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        gateway_api_key = config_manager.global_config.api_key
        if gateway_api_key:
            provided_key = _extract_gateway_api_key()
            if not provided_key:
                return jsonify({
                    "error": {
                        "message": "Missing API key",
                        "type": "invalid_request_error",
                        "code": 401
                    }
                }), 401
            
            if provided_key != gateway_api_key:
                return jsonify({
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                        "code": 401
                    }
                }), 401
        
        return f(*args, **kwargs)
    return decorated_function


# ==================== 路由定义 ====================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({"status": "healthy", "version": "1.0.0"})


@app.route('/v1/models', methods=['GET'])
@require_api_key
def list_models():
    """列出所有可用模型"""
    models = provider_manager.get_available_models()
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "api-gateway"
            }
            for model in models
        ]
    })


@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    """Chat Completion 接口"""
    return proxy.handle_chat_completion(request)


@app.route('/v1/messages', methods=['POST'])
@require_api_key
def claude_messages():
    """Claude Messages 接口"""
    return proxy.handle_claude_messages(request)


@app.route('/v1beta/models/<path:model_action>', methods=['POST'])
@require_api_key
def gemini_v1beta_generate_content(model_action: str):
    """Gemini generateContent/streamGenerateContent 接口 (v1beta)"""
    return proxy.handle_gemini_content(request, model_action)


@app.route('/v1/models/<path:model_action>', methods=['POST'])
@require_api_key
def gemini_v1_generate_content(model_action: str):
    """Gemini generateContent/streamGenerateContent 接口 (v1)"""
    return proxy.handle_gemini_content(request, model_action)


@app.route('/v1/completions', methods=['POST'])
@require_api_key
def completions():
    """Completion 接口"""
    return proxy.handle_completion(request)


@app.route('/v1/embeddings', methods=['POST'])
@require_api_key
def embeddings():
    """Embeddings 接口"""
    return proxy.handle_embeddings(request)


# ==================== 管理接口 ====================

@app.route('/admin/stats', methods=['GET'])
@require_api_key
def get_stats():
    """获取使用统计（简单实现，生产环境应该加认证）"""
    stats = {}
    for model in provider_manager.get_available_models():
        stats[model] = {}
        for provider in config_manager.get_providers(model):
            usage = usage_manager.get_usage(model, provider.name)
            stats[model][provider.name] = {
                "requests": usage.requests,
                "tokens": usage.tokens,
                "last_reset": usage.last_reset,
                "limit_requests": provider.rate_limit.requests_per_period if provider.rate_limit else None,
                "limit_tokens": provider.rate_limit.tokens_per_period if provider.rate_limit else None
            }
    return jsonify(stats)


@app.route('/admin/reload', methods=['POST'])
@require_api_key
def reload_config():
    """重新加载配置"""
    try:
        config_manager.reload()
        scheduler.reload()
        return jsonify({"status": "ok", "message": "Configuration reloaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/admin/providers/<model_name>', methods=['GET'])
@require_api_key
def get_model_providers(model_name: str):
    """获取模型的提供商信息"""
    providers = config_manager.get_providers(model_name)
    if not providers:
        return jsonify({"error": "Model not found"}), 404
    
    result = []
    for p in providers:
        status, reason = provider_manager.rate_limiter.check_availability(model_name, p)
        result.append({
            "name": p.name,
            "priority": p.priority,
            "weight": p.weight,
            "enabled": p.enabled,
            "status": status.value,
            "status_reason": reason,
            "stream_support": p.stream_support,
            "non_stream_support": p.non_stream_support
        })
    
    return jsonify(result)


# ==================== 启动 ====================

# 启动调度器
if os.getenv("ENABLE_SCHEDULER", "true").lower() in ("1", "true", "yes", "on"):
    scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6010, debug=True)
