"""
配置加载和管理
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from filelock import FileLock
from .models import (
    ProviderConfig, ModelConfig, GlobalConfig, 
    RateLimitConfig, UsageData
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "/app/config/provider.json"):
        self.config_path = config_path
        self.data_dir = "/app/data/usage"
        self._config: Dict[str, Any] = {}
        self._global_config: GlobalConfig = GlobalConfig()
        self._models: Dict[str, ModelConfig] = {}
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            # 解析全局配置
            if "_global" in self._config:
                self._global_config = GlobalConfig(**self._config["_global"])
            
            # 解析模型配置
            for model_name, model_data in self._config.items():
                if model_name.startswith("_"):
                    continue
                    
                providers = []
                for p in model_data.get("providers", []):
                    # 处理 rate_limit
                    if p.get("rate_limit"):
                        p["rate_limit"] = RateLimitConfig(**p["rate_limit"])
                    providers.append(ProviderConfig(**p))
                
                self._models[model_name] = ModelConfig(providers=providers)
                
            logger.info(f"Loaded configuration for {len(self._models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    @property
    def global_config(self) -> GlobalConfig:
        return self._global_config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self._models.get(model_name)
    
    def get_all_models(self) -> list:
        """获取所有模型名称"""
        return list(self._models.keys())
    
    def get_providers(self, model_name: str) -> list:
        """获取模型的所有提供商"""
        model_config = self._models.get(model_name)
        if model_config:
            return [p for p in model_config.providers if p.enabled]
        return []


class UsageManager:
    """使用量管理"""
    
    def __init__(self, data_dir: str = "/app/data/usage"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_usage_path(self, model_name: str, provider_name: str) -> str:
        """获取使用量数据文件路径"""
        safe_name = f"{model_name}_{provider_name}".replace("/", "_")
        return os.path.join(self.data_dir, f"{safe_name}.json")
    
    def get_usage(self, model_name: str, provider_name: str) -> UsageData:
        """获取使用量"""
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"
        
        try:
            with FileLock(lock_path, timeout=5):
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        return UsageData(**data)
        except Exception as e:
            logger.warning(f"Failed to read usage data: {e}")
        
        return UsageData()
    
    def update_usage(self, model_name: str, provider_name: str, 
                     requests_delta: int = 0, tokens_delta: int = 0):
        """更新使用量"""
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"
        
        try:
            with FileLock(lock_path, timeout=5):
                usage = UsageData()
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            usage = UsageData(**data)
                    except Exception as e:
                        logger.warning(f"Failed to read usage data when updating: {e}")
                usage.requests += requests_delta
                usage.tokens += tokens_delta
                
                with open(path, 'w') as f:
                    json.dump(usage.model_dump(), f)
                    
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
    
    def reset_usage(self, model_name: str, provider_name: str):
        """重置使用量"""
        from datetime import datetime
        
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"
        
        try:
            with FileLock(lock_path, timeout=5):
                usage = UsageData(
                    requests=0,
                    tokens=0,
                    last_reset=datetime.now().isoformat()
                )
                with open(path, 'w') as f:
                    json.dump(usage.model_dump(), f)
                    
            logger.info(f"Reset usage for {model_name}/{provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset usage: {e}")
