"""
限额管理器
"""

import logging
from typing import Tuple
from .config import ConfigManager, UsageManager
from .models import ProviderConfig, ProviderStatus

logger = logging.getLogger(__name__)


class RateLimiter:
    """限额检查器"""
    
    def __init__(self, config_manager: ConfigManager, usage_manager: UsageManager):
        self.config_manager = config_manager
        self.usage_manager = usage_manager
    
    def check_availability(self, model_name: str, 
                          provider: ProviderConfig) -> Tuple[ProviderStatus, str]:
        """
        检查提供商是否可用
        返回: (状态, 原因)
        """
        if not provider.enabled:
            return ProviderStatus.DISABLED, "Provider is disabled"
        
        rate_limit = provider.rate_limit
        if not rate_limit:
            return ProviderStatus.AVAILABLE, "No rate limit configured"
        
        usage = self.usage_manager.get_usage(model_name, provider.name)
        
        # 检查请求数限制
        if rate_limit.requests_per_period is not None:
            if usage.requests >= rate_limit.requests_per_period:
                return (ProviderStatus.RATE_LIMITED, 
                       f"Request limit exceeded: {usage.requests}/{rate_limit.requests_per_period}")
        
        # 检查token数限制
        if rate_limit.tokens_per_period is not None:
            if usage.tokens >= rate_limit.tokens_per_period:
                return (ProviderStatus.RATE_LIMITED,
                       f"Token limit exceeded: {usage.tokens}/{rate_limit.tokens_per_period}")
        
        return ProviderStatus.AVAILABLE, "OK"
    
    def record_usage(self, model_name: str, provider_name: str,
                     requests: int = 1, tokens: int = 0):
        """记录使用量"""
        self.usage_manager.update_usage(
            model_name, provider_name,
            requests_delta=requests,
            tokens_delta=tokens
        )
        logger.debug(f"Recorded usage for {model_name}/{provider_name}: "
                    f"requests={requests}, tokens={tokens}")
    
    def get_usage_stats(self, model_name: str, provider_name: str) -> dict:
        """获取使用统计"""
        usage = self.usage_manager.get_usage(model_name, provider_name)
        provider = None
        
        for p in self.config_manager.get_providers(model_name):
            if p.name == provider_name:
                provider = p
                break
        
        if not provider or not provider.rate_limit:
            return {
                "requests": usage.requests,
                "tokens": usage.tokens,
                "requests_limit": None,
                "tokens_limit": None
            }
        
        return {
            "requests": usage.requests,
            "tokens": usage.tokens,
            "requests_limit": provider.rate_limit.requests_per_period,
            "tokens_limit": provider.rate_limit.tokens_per_period,
            "last_reset": usage.last_reset
        }