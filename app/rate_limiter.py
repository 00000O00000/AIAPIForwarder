"""Provider-level rate limit checks and usage accounting."""

import logging
from typing import Dict, Optional, Tuple

from .config import ConfigManager, UsageManager
from .models import ProviderConfig, ProviderStatus

logger = logging.getLogger(__name__)


class RateLimiter:
    """Check provider availability against configured rate limits."""

    def __init__(self, config_manager: ConfigManager, usage_manager: UsageManager):
        self.config_manager = config_manager
        self.usage_manager = usage_manager

    def check_availability(self, model_name: str, provider: ProviderConfig) -> Tuple[ProviderStatus, str]:
        """Return provider availability and reason."""
        if not provider.enabled:
            return ProviderStatus.DISABLED, "Provider is disabled"

        rate_limit = provider.rate_limit
        if not rate_limit:
            return ProviderStatus.AVAILABLE, "No rate limit configured"

        usage = self.usage_manager.get_usage(model_name, provider.name)

        if rate_limit.requests_per_period is not None and usage.requests >= rate_limit.requests_per_period:
            return (
                ProviderStatus.RATE_LIMITED,
                f"Request limit exceeded: {usage.requests}/{rate_limit.requests_per_period}",
            )

        if rate_limit.tokens_per_period is not None and usage.tokens >= rate_limit.tokens_per_period:
            return (
                ProviderStatus.RATE_LIMITED,
                f"Token limit exceeded: {usage.tokens}/{rate_limit.tokens_per_period}",
            )

        return ProviderStatus.AVAILABLE, "OK"

    def record_usage(self, model_name: str, provider_name: str, requests: int = 1, tokens: int = 0):
        """Persist usage deltas for a provider."""
        try:
            safe_requests = max(0, int(requests))
        except (TypeError, ValueError):
            safe_requests = 0

        try:
            safe_tokens = max(0, int(tokens))
        except (TypeError, ValueError):
            safe_tokens = 0

        if safe_requests == 0 and safe_tokens == 0:
            return

        self.usage_manager.update_usage(
            model_name,
            provider_name,
            requests_delta=safe_requests,
            tokens_delta=safe_tokens,
        )
        logger.debug(
            "Recorded usage for %s/%s: requests=%s, tokens=%s",
            model_name,
            provider_name,
            safe_requests,
            safe_tokens,
        )

    def get_usage_stats(self, model_name: str, provider_name: str) -> Dict[str, Optional[int]]:
        """Return usage and configured limits for one provider."""
        usage = self.usage_manager.get_usage(model_name, provider_name)
        provider = next((p for p in self.config_manager.get_providers(model_name) if p.name == provider_name), None)

        if not provider or not provider.rate_limit:
            return {
                "requests": usage.requests,
                "tokens": usage.tokens,
                "requests_limit": None,
                "tokens_limit": None,
            }

        return {
            "requests": usage.requests,
            "tokens": usage.tokens,
            "requests_limit": provider.rate_limit.requests_per_period,
            "tokens_limit": provider.rate_limit.tokens_per_period,
            "last_reset": usage.last_reset,
        }
