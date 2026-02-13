"""Flask application entrypoint."""

import logging
import os
import hmac
from functools import wraps

from flask import Flask, jsonify, request

from .config import ConfigManager, UsageManager
from .provider_manager import ProviderManager
from .proxy import OpenAIProxy
from .scheduler import UsageResetScheduler


handlers = [logging.StreamHandler()]
try:
    os.makedirs("/app/logs", exist_ok=True)
    handlers.append(logging.FileHandler("/app/logs/gateway.log"))
except Exception:
    # Keep console logging when filesystem path is unavailable.
    pass

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)


config_manager = ConfigManager()
usage_manager = UsageManager()
provider_manager = ProviderManager(config_manager, usage_manager)
proxy = OpenAIProxy(provider_manager)
scheduler = UsageResetScheduler(config_manager, usage_manager)

app = Flask(__name__)


def _extract_gateway_api_key() -> str:
    """Extract gateway API key from common request locations."""
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


def require_api_key(func):
    """Route decorator for optional gateway API key auth."""

    @wraps(func)
    def decorated_function(*args, **kwargs):
        gateway_api_key = config_manager.global_config.api_key
        if gateway_api_key:
            provided_key = _extract_gateway_api_key()
            if not provided_key:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Missing API key",
                                "type": "invalid_request_error",
                                "code": 401,
                            }
                        }
                    ),
                    401,
                )

            if not hmac.compare_digest(provided_key, gateway_api_key):
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Invalid API key",
                                "type": "invalid_request_error",
                                "code": 401,
                            }
                        }
                    ),
                    401,
                )

        return func(*args, **kwargs)

    return decorated_function


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})


@app.route("/v1/models", methods=["GET"])
@require_api_key
def list_models():
    models = provider_manager.get_available_models()
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "api-gateway",
                }
                for model in models
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
@require_api_key
def chat_completions():
    return proxy.handle_chat_completion(request)


@app.route("/v1/messages", methods=["POST"])
@require_api_key
def claude_messages():
    return proxy.handle_claude_messages(request)


@app.route("/v1beta/models/<path:model_action>", methods=["POST"])
@require_api_key
def gemini_v1beta_generate_content(model_action: str):
    return proxy.handle_gemini_content(request, model_action)


@app.route("/v1/models/<path:model_action>", methods=["POST"])
@require_api_key
def gemini_v1_generate_content(model_action: str):
    return proxy.handle_gemini_content(request, model_action)


@app.route("/v1/completions", methods=["POST"])
@require_api_key
def completions():
    return proxy.handle_completion(request)


@app.route("/v1/embeddings", methods=["POST"])
@require_api_key
def embeddings():
    return proxy.handle_embeddings(request)


@app.route("/admin/stats", methods=["GET"])
@require_api_key
def get_stats():
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
                "limit_tokens": provider.rate_limit.tokens_per_period if provider.rate_limit else None,
                "limit_max_worker": provider.rate_limit.max_worker if provider.rate_limit else None,
            }
    return jsonify(stats)


@app.route("/admin/reload", methods=["POST"])
@require_api_key
def reload_config():
    try:
        config_manager.reload()
        scheduler.reload()
        return jsonify({"status": "ok", "message": "Configuration reloaded"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/admin/providers/<model_name>", methods=["GET"])
@require_api_key
def get_model_providers(model_name: str):
    providers = config_manager.get_providers(model_name)
    if not providers:
        return jsonify({"error": "Model not found"}), 404
    queue_overflow_factor = config_manager.global_config.queue_overflow_factor
    priority_groups = {}
    for provider in providers:
        priority_groups.setdefault(provider.priority, []).append(provider)

    priority_queue_limits = {
        priority: provider_manager.get_priority_queue_limit(group, queue_overflow_factor)
        for priority, group in priority_groups.items()
    }
    priority_waiting_workers = {
        priority: provider_manager.get_priority_waiting_workers(model_name, priority)
        for priority in priority_groups
    }

    result = []
    for provider in providers:
        status, reason = provider_manager.rate_limiter.check_availability(model_name, provider)
        provider_max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
        provider_running_workers = provider_manager.get_provider_running_workers(model_name, provider.name)
        provider_worker_available = (
            provider_max_worker is None or provider_running_workers < provider_max_worker
        )
        result.append(
            {
                "name": provider.name,
                "priority": provider.priority,
                "weight": provider.weight,
                "enabled": provider.enabled,
                "status": status.value,
                "status_reason": reason,
                "stream_support": provider.stream_support,
                "non_stream_support": provider.non_stream_support,
                "provider_max_worker": provider_max_worker,
                "provider_running_workers": provider_running_workers,
                "provider_worker_available": provider_worker_available,
                "priority_queue_overflow_factor": queue_overflow_factor,
                "priority_queue_limit": priority_queue_limits.get(provider.priority),
                "priority_waiting_workers": priority_waiting_workers.get(provider.priority, 0),
            }
        )

    return jsonify(result)


if os.getenv("ENABLE_SCHEDULER", "true").lower() in ("1", "true", "yes", "on"):
    scheduler.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6010, debug=os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes", "on"))
