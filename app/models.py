"""
数据模型定义
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum


class RateLimitConfig(BaseModel):
    """限额配置"""
    requests_per_period: Optional[int] = None  # 每周期请求数限制
    tokens_per_period: Optional[int] = None    # 每周期token数限制
    period_cron: str = "0 0 * * *"             # cron表达式，默认每天0点刷新
    

class ProviderConfig(BaseModel):
    """单个提供商配置"""
    name: str                                    # 提供商名称
    endpoint: str                                # API端点
    api_key: str                                 # API密钥
    model: str                                   # 上游模型名
    priority: int = 1                            # 优先级（数字越小优先级越高）
    weight: int = 10                             # 轮询权重
    rate_limit: Optional[RateLimitConfig] = None # 限额配置
    retry: int = 3                               # 重试次数
    timeout: int = 60                            # 超时时间（秒）
    stream_support: bool = True                  # 是否支持流式
    non_stream_support: bool = True              # 是否支持非流式
    enabled: bool = True                         # 是否启用
    custom_headers: Optional[Dict[str, str]] = None  # 自定义请求头
    max_context_length: Optional[int] = None     # 最大上下文长度
    

class ModelConfig(BaseModel):
    """模型配置"""
    providers: List[ProviderConfig]


class GlobalConfig(BaseModel):
    """全局配置"""
    default_timeout: int = 120
    default_retry: int = 3
    log_requests: bool = True
    api_key: Optional[str] = None  # 网关自身的API Key验证


class UsageData(BaseModel):
    """使用量数据"""
    requests: int = 0
    tokens: int = 0
    last_reset: str = ""
    

class ProviderStatus(Enum):
    """提供商状态"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"