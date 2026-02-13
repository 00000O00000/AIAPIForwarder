"""
配置加载和管理
"""

import json
import os
import logging
import re
from typing import Dict, Any, Optional, List
from filelock import FileLock
from .models import (
    ProviderConfig, ModelConfig, GlobalConfig, 
    RateLimitConfig, UsageData
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.data_dir = self._resolve_usage_dir()
        self._config: Dict[str, Any] = {}
        self._global_config: GlobalConfig = GlobalConfig()
        self._models: Dict[str, ModelConfig] = {}
        self._load_config()

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> str:
        if config_path:
            return config_path
        candidate = os.getenv("CONFIG_PATH", "/app/config/provider.json")
        if os.path.exists(candidate):
            return candidate
        local_candidate = os.path.join("config", "provider.json")
        if os.path.exists(local_candidate):
            return local_candidate
        return candidate

    @staticmethod
    def _resolve_usage_dir() -> str:
        candidate = os.getenv("USAGE_DATA_DIR", "/app/data/usage")
        if os.path.isdir(candidate):
            return candidate
        local_candidate = os.path.join("data", "usage")
        return local_candidate if os.path.isdir("data") else candidate
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8-sig') as f:
                config_data = json.load(f)

            if not isinstance(config_data, dict):
                raise ValueError("Configuration root must be a JSON object")

            self._config = config_data
            self._global_config = GlobalConfig()
            self._models = {}

            # 解析全局配置
            if "_global" in self._config:
                self._global_config = GlobalConfig(**self._config["_global"])

            # 解析模型配置
            for model_name, model_data in self._config.items():
                if model_name.startswith("_"):
                    continue

                if not isinstance(model_data, dict):
                    logger.warning(f"Skip invalid model config '{model_name}': model config must be an object")
                    continue

                providers: List[ProviderConfig] = []
                for p in model_data.get("providers", []):
                    if not isinstance(p, dict):
                        logger.warning(f"Skip invalid provider in model '{model_name}': provider entry must be an object")
                        continue
                    provider_data = dict(p)
                    # 处理 rate_limit
                    if provider_data.get("rate_limit"):
                        provider_data["rate_limit"] = RateLimitConfig(**provider_data["rate_limit"])
                    providers.append(ProviderConfig(**provider_data))

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
    
    def get_all_models(self) -> List[str]:
        """获取所有模型名称"""
        return list(self._models.keys())
    
    def get_providers(self, model_name: str) -> List[ProviderConfig]:
        """获取模型的所有提供商"""
        model_config = self._models.get(model_name)
        if model_config:
            return [p for p in model_config.providers if p.enabled]
        return []


class UsageManager:
    """使用量管理"""
    
    def __init__(self, data_dir: Optional[str] = None):
        default_dir = os.getenv("USAGE_DATA_DIR", "/app/data/usage")
        if data_dir:
            self.data_dir = data_dir
        elif os.path.isdir(default_dir):
            self.data_dir = default_dir
        else:
            self.data_dir = os.path.join("data", "usage")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_usage_path(self, model_name: str, provider_name: str) -> str:
        """获取使用量数据文件路径"""
        raw_name = f"{model_name}_{provider_name}"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
        return os.path.join(self.data_dir, f"{safe_name}.json")
    
    def get_usage(self, model_name: str, provider_name: str) -> UsageData:
        """获取使用量"""
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"
        
        try:
            with FileLock(lock_path, timeout=5):
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
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
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            usage = UsageData(**data)
                    except Exception as e:
                        logger.warning(f"Failed to read usage data when updating: {e}")
                usage.requests = max(0, usage.requests + requests_delta)
                usage.tokens = max(0, usage.tokens + tokens_delta)
                
                with open(path, 'w', encoding='utf-8') as f:
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
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(usage.model_dump(), f)
                    
            logger.info(f"Reset usage for {model_name}/{provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset usage: {e}")
