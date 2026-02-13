"""Provider selection and runtime model-concurrency management."""

import logging
import random
import threading
from typing import Dict, List, Optional, Tuple

from .config import ConfigManager, UsageManager
from .models import ProviderConfig, ProviderStatus
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ProviderManager:
    """Manage provider availability checks, selection and usage tracking."""

    def __init__(self, config_manager: ConfigManager, usage_manager: Optional[UsageManager] = None):
        self.config_manager = config_manager
        self.usage_manager = usage_manager or UsageManager()
        self.rate_limiter = RateLimiter(config_manager, self.usage_manager)

        # Runtime in-flight counters for model-level max_worker limit.
        self._model_running_workers: Dict[str, int] = {}
        self._model_workers_lock = threading.Lock()

    def get_available_models(self) -> List[str]:
        """Return all configured model names."""
        return self.config_manager.get_all_models()

    def try_acquire_model_worker(self, model_name: str) -> Tuple[bool, str]:
        """Try to reserve one worker slot for a model."""
        max_worker = self.config_manager.get_model_max_worker(model_name)
        if max_worker is None:
            return True, "No max_worker limit configured"

        with self._model_workers_lock:
            running = self._model_running_workers.get(model_name, 0)
            if running >= max_worker:
                return False, f"Model concurrency limit reached: {running}/{max_worker}"
            self._model_running_workers[model_name] = running + 1
            return True, f"Model worker acquired: {running + 1}/{max_worker}"

    def release_model_worker(self, model_name: str):
        """Release one worker slot for a model."""
        with self._model_workers_lock:
            running = self._model_running_workers.get(model_name, 0)
            if running <= 0:
                logger.debug("Model worker release ignored for %s: no running workers", model_name)
                return
            if running == 1:
                self._model_running_workers.pop(model_name, None)
                return
            self._model_running_workers[model_name] = running - 1

    def get_model_running_workers(self, model_name: str) -> int:
        """Return current in-flight request count for a model."""
        with self._model_workers_lock:
            return self._model_running_workers.get(model_name, 0)

    def select_provider(
        self,
        model_name: str,
        exclude_providers: Optional[List[str]] = None,
        require_stream: Optional[bool] = None,
    ) -> Optional[ProviderConfig]:
        """Select one available provider by priority and weight."""
        exclude = set(exclude_providers or [])
        providers = self.config_manager.get_providers(model_name)

        if not providers:
            logger.warning("No providers configured for model: %s", model_name)
            return None

        priority_groups: Dict[int, List[ProviderConfig]] = {}
        for provider in providers:
            if provider.name in exclude:
                continue

            if require_stream is True and not provider.stream_support:
                continue
            if require_stream is False and not provider.non_stream_support:
                continue

            status, reason = self.rate_limiter.check_availability(model_name, provider)
            if status != ProviderStatus.AVAILABLE:
                logger.debug("Provider %s unavailable: %s", provider.name, reason)
                continue

            priority_groups.setdefault(provider.priority, []).append(provider)

        if not priority_groups:
            logger.warning("No available providers for model: %s", model_name)
            return None

        best_priority = min(priority_groups.keys())
        selected = self._weighted_random_choice(priority_groups[best_priority])
        logger.info("Selected provider %s for model %s", selected.name, model_name)
        return selected

    def _weighted_random_choice(self, providers: List[ProviderConfig]) -> ProviderConfig:
        """Return one provider selected by configured weight."""
        if not providers:
            raise ValueError("providers must not be empty")
        if len(providers) == 1:
            return providers[0]

        weights = [max(0, p.weight) for p in providers]
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(providers)

        return random.choices(providers, weights=weights, k=1)[0]

    def get_provider_by_name(self, model_name: str, provider_name: str) -> Optional[ProviderConfig]:
        """Return provider config by provider name."""
        for provider in self.config_manager.get_providers(model_name):
            if provider.name == provider_name:
                return provider
        return None

    def record_success(self, model_name: str, provider_name: str, tokens: int = 0):
        """Record one successful request."""
        self.rate_limiter.record_usage(model_name, provider_name, requests=1, tokens=tokens)

    def record_stream_tokens(self, model_name: str, provider_name: str, tokens: int):
        """Record stream-token usage without incrementing request count."""
        self.rate_limiter.record_usage(model_name, provider_name, requests=0, tokens=tokens)

    def get_providers_by_priority(self, model_name: str) -> Dict[int, List[ProviderConfig]]:
        """Return enabled providers grouped by priority, sorted by weight desc."""
        providers = self.config_manager.get_providers(model_name)
        groups: Dict[int, List[ProviderConfig]] = {}
        for provider in providers:
            groups.setdefault(provider.priority, []).append(provider)

        for priority in groups:
            groups[priority].sort(key=lambda item: item.weight, reverse=True)

        return groups
