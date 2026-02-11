"""
提供商管理器 - 负责选择和管理提供商
"""

import random
import logging
from typing import Optional, List, Dict, Tuple
from .config import ConfigManager, UsageManager
from .rate_limiter import RateLimiter
from .models import ProviderConfig, ProviderStatus

logger = logging.getLogger(__name__)


class ProviderManager:
    """提供商管理器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.usage_manager = UsageManager()
        self.rate_limiter = RateLimiter(config_manager, self.usage_manager)
    
    def get_available_models(self) -> List[str]:
        """获取所有可用模型"""
        return self.config_manager.get_all_models()
    
    def select_provider(self, model_name: str, 
                       exclude_providers: List[str] = None,
                       require_stream: bool = None) -> Optional[ProviderConfig]:
        """
        选择一个可用的提供商
        
        Args:
            model_name: 模型名称
            exclude_providers: 排除的提供商列表
            require_stream: 是否需要流式支持 (True=需要流式, False=需要非流式, None=不限)
        
        Returns:
            选中的提供商配置，如果没有可用的则返回 None
        """
        exclude_providers = exclude_providers or []
        providers = self.config_manager.get_providers(model_name)
        
        if not providers:
            logger.warning(f"No providers configured for model: {model_name}")
            return None
        
        # 按优先级分组
        priority_groups: dict = {}
        for provider in providers:
            if provider.name in exclude_providers:
                continue
            
            # 检查流式支持
            if require_stream is True and not provider.stream_support:
                continue
            if require_stream is False and not provider.non_stream_support:
                continue
            
            # 检查可用性
            status, reason = self.rate_limiter.check_availability(model_name, provider)
            if status != ProviderStatus.AVAILABLE:
                logger.debug(f"Provider {provider.name} unavailable: {reason}")
                continue
            
            priority = provider.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(provider)
        
        if not priority_groups:
            logger.warning(f"No available providers for model: {model_name}")
            return None
        
        # 选择优先级最高的组（数字最小）
        best_priority = min(priority_groups.keys())
        candidates = priority_groups[best_priority]
        
        # 在同优先级中按权重随机选择
        selected = self._weighted_random_choice(candidates)
        logger.info(f"Selected provider {selected.name} for model {model_name}")
        
        return selected
    
    def _weighted_random_choice(self, providers: List[ProviderConfig]) -> ProviderConfig:
        """根据权重随机选择"""
        if len(providers) == 1:
            return providers[0]
        
        total_weight = sum(p.weight for p in providers)
        r = random.uniform(0, total_weight)
        
        current = 0
        for provider in providers:
            current += provider.weight
            if r <= current:
                return provider
        
        return providers[-1]
    
    def get_provider_by_name(self, model_name: str, 
                             provider_name: str) -> Optional[ProviderConfig]:
        """根据名称获取提供商"""
        providers = self.config_manager.get_providers(model_name)
        for p in providers:
            if p.name == provider_name:
                return p
        return None
    
    def record_success(self, model_name: str, provider_name: str, tokens: int = 0):
        """记录成功请求"""
        self.rate_limiter.record_usage(model_name, provider_name, 
                                       requests=1, tokens=tokens)
    
    def record_stream_tokens(self, model_name: str, provider_name: str, tokens: int):
        """记录流式请求结束后的 token 使用量（不增加请求数）"""
        self.rate_limiter.record_usage(model_name, provider_name,
                                       requests=0, tokens=tokens)
    
    def get_providers_by_priority(self, model_name: str) -> Dict[int, List[ProviderConfig]]:
        """
        获取按优先级分组的提供商列表
        
        Returns:
            {priority: [provider_configs]} 字典
        """
        providers = self.config_manager.get_providers(model_name)
        groups: Dict[int, List[ProviderConfig]] = {}
        for p in providers:
            if p.priority not in groups:
                groups[p.priority] = []
            groups[p.priority].append(p)
        return groups