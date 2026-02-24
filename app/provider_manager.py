"""Provider selection and runtime provider-concurrency management."""

import logging
import math
import random
import threading
import time
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

        # Runtime in-flight counters for provider-level max_worker limit.
        self._provider_running_workers: Dict[str, int] = {}
        self._provider_workers_lock = threading.Lock()
        self._provider_workers_condition = threading.Condition(self._provider_workers_lock)
        self._priority_waiting_workers: Dict[str, int] = {}

    def get_available_models(self) -> List[str]:
        """Return all configured model names."""
        return self.config_manager.get_all_models()

    @staticmethod
    def _provider_worker_key(model_name: str, provider_name: str) -> str:
        return f"{model_name}::{provider_name}"

    @staticmethod
    def _priority_queue_key(model_name: str, priority: int) -> str:
        return f"{model_name}::priority::{priority}"

    def try_acquire_provider_worker(self, model_name: str, provider: ProviderConfig) -> Tuple[bool, str]:
        """Try to reserve one worker slot for a provider."""
        max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
        if max_worker is None:
            return True, "No provider max_worker limit configured"

        key = self._provider_worker_key(model_name, provider.name)
        with self._provider_workers_condition:
            running = self._provider_running_workers.get(key, 0)
            if running >= max_worker:
                return False, f"Provider concurrency limit reached: {provider.name} {running}/{max_worker}"
            self._provider_running_workers[key] = running + 1
            return True, f"Provider worker acquired: {provider.name} {running + 1}/{max_worker}"

    def check_and_acquire_provider_worker(
        self, model_name: str, provider: ProviderConfig,
    ) -> Tuple[bool, str, str]:
        """
        先检查 rate_limit 可用性（锁外），再在锁内获取 worker。
        返回 (acquired, reason, fail_type):
          - fail_type: "rate_limited" / "max_worker" / "" (成功时为空)
        """
        status, reason = self.rate_limiter.check_availability(model_name, provider)
        if status != ProviderStatus.AVAILABLE:
            return False, reason, "rate_limited"

        max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
        if max_worker is None:
            return True, "No provider max_worker limit configured", ""

        key = self._provider_worker_key(model_name, provider.name)
        with self._provider_workers_condition:
            running = self._provider_running_workers.get(key, 0)
            if running >= max_worker:
                return False, f"Provider concurrency limit reached: {provider.name} {running}/{max_worker}", "max_worker"
            self._provider_running_workers[key] = running + 1
            return True, f"Provider worker acquired: {provider.name} {running + 1}/{max_worker}", ""

    def is_provider_worker_available(self, model_name: str, provider: ProviderConfig) -> Tuple[bool, str]:
        """Check whether provider worker slots are available without mutating state."""
        max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
        if max_worker is None:
            return True, "No provider max_worker limit configured"

        key = self._provider_worker_key(model_name, provider.name)
        with self._provider_workers_lock:
            running = self._provider_running_workers.get(key, 0)
        if running >= max_worker:
            return False, f"Provider concurrency limit reached: {provider.name} {running}/{max_worker}"
        return True, f"Provider worker available: {provider.name} {running}/{max_worker}"

    def release_provider_worker(self, model_name: str, provider_name: str):
        """Release one worker slot for a provider."""
        key = self._provider_worker_key(model_name, provider_name)
        with self._provider_workers_condition:
            running = self._provider_running_workers.get(key, 0)
            if running <= 0:
                logger.debug("Provider worker release ignored for %s/%s: no running workers", model_name, provider_name)
                return
            if running == 1:
                self._provider_running_workers.pop(key, None)
            else:
                self._provider_running_workers[key] = running - 1
            self._provider_workers_condition.notify()

    def get_provider_running_workers(self, model_name: str, provider_name: str) -> int:
        """Return current in-flight request count for a provider."""
        key = self._provider_worker_key(model_name, provider_name)
        with self._provider_workers_lock:
            return self._provider_running_workers.get(key, 0)

    @staticmethod
    def _normalize_queue_overflow_factor(value: Optional[float]) -> float:
        if value is None:
            return 2.0
        try:
            factor = float(value)
        except (TypeError, ValueError):
            return 2.0
        if not math.isfinite(factor):
            return 2.0
        return 1.0 if factor < 1.0 else factor

    def _calculate_priority_queue_limit(self, providers: List[ProviderConfig], queue_overflow_factor: Optional[float]) -> Optional[int]:
        """
        Queue limit for one priority.
        Returns:
          - None: no finite queue limit can be calculated (e.g. unlimited provider max_worker)
          - 0: queue disabled (factor <= 1)
          - >0: max waiting requests allowed for this priority
        """
        total_max_worker = 0
        for provider in providers:
            max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
            if max_worker is None:
                logger.warning(
                    "排队机制被静默禁用：provider '%s' 未配置 max_worker，"
                    "同优先级内混用有/无 max_worker 的 provider 会导致排队不生效",
                    provider.name,
                )
                return None
            total_max_worker += max_worker

        factor = self._normalize_queue_overflow_factor(queue_overflow_factor)
        if factor <= 1.0:
            return 0
        # Strictly match "overflow when waiting_count > total_max_worker * factor".
        # Waiting count is integer, so admitted upper bound is floor(total_max_worker * factor).
        raw_limit = total_max_worker * factor
        return max(0, int(math.floor(raw_limit)))

    def get_priority_waiting_workers(self, model_name: str, priority: int) -> int:
        key = self._priority_queue_key(model_name, priority)
        with self._provider_workers_lock:
            return self._priority_waiting_workers.get(key, 0)

    def get_priority_queue_limit(self, providers: List[ProviderConfig], queue_overflow_factor: Optional[float]) -> Optional[int]:
        return self._calculate_priority_queue_limit(providers, queue_overflow_factor)

    def _any_provider_capacity_locked(self, model_name: str, providers: List[ProviderConfig]) -> bool:
        for provider in providers:
            max_worker = provider.rate_limit.max_worker if provider.rate_limit else None
            if max_worker is None:
                return True
            key = self._provider_worker_key(model_name, provider.name)
            running = self._provider_running_workers.get(key, 0)
            if running < max_worker:
                return True
        return False

    def wait_for_priority_capacity(
        self,
        model_name: str,
        priority: int,
        providers: List[ProviderConfig],
        wait_timeout: float,
        queue_overflow_factor: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Wait in priority queue until at least one provider slot becomes available.
        Returns:
          (True, reason): admitted and woken with potential capacity
          (False, reason): queue disabled / overflow / timeout
        """
        queue_limit = self._calculate_priority_queue_limit(providers, queue_overflow_factor)
        if queue_limit is None:
            return False, "Priority queue disabled: provider max_worker is not fully configured"
        if queue_limit <= 0:
            return False, "Priority queue disabled by queue_overflow_factor"

        queue_key = self._priority_queue_key(model_name, priority)
        timeout_seconds = max(0.0, float(wait_timeout))
        deadline = time.monotonic() + timeout_seconds

        with self._provider_workers_condition:
            waiting = self._priority_waiting_workers.get(queue_key, 0)
            if waiting + 1 > queue_limit:
                return False, f"Priority queue overflow: {waiting}/{queue_limit}"

            self._priority_waiting_workers[queue_key] = waiting + 1
            try:
                while True:
                    if self._any_provider_capacity_locked(model_name, providers):
                        return True, "Priority queue woke up with available provider capacity"

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False, "Priority queue wait timeout"
                    self._provider_workers_condition.wait(timeout=min(0.2, remaining))
            finally:
                current_waiting = self._priority_waiting_workers.get(queue_key, 0)
                if current_waiting <= 1:
                    self._priority_waiting_workers.pop(queue_key, None)
                else:
                    self._priority_waiting_workers[queue_key] = current_waiting - 1

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
            available, available_reason = self.is_provider_worker_available(model_name, provider)
            if not available:
                logger.debug("Provider %s runtime unavailable: %s", provider.name, available_reason)
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

        weights = [max(1, p.weight) for p in providers]
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
