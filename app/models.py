"""
Data model definitions.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RateLimitConfig(BaseModel):
    requests_per_period: Optional[int] = Field(default=None, ge=0)
    tokens_per_period: Optional[int] = Field(default=None, ge=0)
    period_cron: str = "0 0 * * *"


class ProviderConfig(BaseModel):
    name: str
    endpoint: str
    api_key: str
    model: str
    format: Literal["openai", "openai-response", "claude", "gemini"] = "openai"
    priority: int = Field(default=1, ge=1)
    weight: int = Field(default=10, ge=1)
    rate_limit: Optional[RateLimitConfig] = None
    retry: int = Field(default=3, ge=0)
    timeout: int = Field(default=60, gt=0)
    stream_support: bool = True
    non_stream_support: bool = True
    enabled: bool = True
    custom_headers: Optional[Dict[str, str]] = None
    max_context_length: Optional[int] = Field(default=None, ge=1)

    @field_validator("name", "endpoint", "model", mode="before")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        if value is None:
            raise ValueError("must not be null")
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class ModelConfig(BaseModel):
    providers: List[ProviderConfig]


class GlobalConfig(BaseModel):
    default_timeout: int = Field(default=120, gt=0)
    default_retry: int = Field(default=3, ge=0)
    log_requests: bool = True
    api_key: Optional[str] = None


class UsageData(BaseModel):
    requests: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    last_reset: str = ""


class ProviderStatus(Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"
