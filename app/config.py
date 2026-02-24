"""Configuration and usage persistence management."""

import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from filelock import FileLock

from .models import GlobalConfig, ModelConfig, ProviderConfig, RateLimitConfig, UsageData

logger = logging.getLogger(__name__)


class ConfigManager:
    """Load and expose gateway configuration."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
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

    def _load_config(self):
        """Load provider configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8-sig") as f:
                config_data = json.load(f)

            if not isinstance(config_data, dict):
                raise ValueError("Configuration root must be a JSON object")

            self._config = config_data
            self._global_config = GlobalConfig()
            self._models = {}

            if "_global" in self._config:
                self._global_config = GlobalConfig(**self._config["_global"])

            for model_name, model_data in self._config.items():
                if model_name.startswith("_"):
                    continue

                if not isinstance(model_data, dict):
                    logger.warning("Skip invalid model config '%s': model config must be an object", model_name)
                    continue
                if "max_worker" in model_data:
                    logger.warning(
                        "Model-level max_worker in '%s' is ignored. "
                        "Please set rate_limit.max_worker on each provider.",
                        model_name,
                    )

                providers_raw = model_data.get("providers", [])
                if not isinstance(providers_raw, list):
                    logger.warning("Skip invalid providers for model '%s': providers must be an array", model_name)
                    providers_raw = []

                providers: List[ProviderConfig] = []
                for p in providers_raw:
                    if not isinstance(p, dict):
                        logger.warning("Skip invalid provider in model '%s': provider entry must be an object", model_name)
                        continue

                    provider_data = dict(p)
                    try:
                        if provider_data.get("rate_limit"):
                            provider_data["rate_limit"] = RateLimitConfig(**provider_data["rate_limit"])
                        providers.append(ProviderConfig(**provider_data))
                    except Exception as provider_exc:
                        provider_name = provider_data.get("name", "<unknown>")
                        logger.warning(
                            "Skip invalid provider '%s' in model '%s': %s",
                            provider_name,
                            model_name,
                            provider_exc,
                        )
                        continue

                self._models[model_name] = ModelConfig(providers=providers)

            logger.info("Loaded configuration for %s models", len(self._models))

        except Exception as exc:
            logger.error("Failed to load config: %s", exc)
            raise

    def reload(self):
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")

    @property
    def global_config(self) -> GlobalConfig:
        return self._global_config

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by model name."""
        return self._models.get(model_name)

    def get_all_models(self) -> List[str]:
        """Return all configured model names."""
        return list(self._models.keys())

    def get_providers(self, model_name: str) -> List[ProviderConfig]:
        """Return all enabled providers for a model."""
        model_config = self._models.get(model_name)
        if model_config:
            return [p for p in model_config.providers if p.enabled]
        return []


class UsageManager:
    """Manage persisted usage counters per model/provider."""

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
        """Return safe usage file path for model/provider."""
        raw_name = f"{model_name}_{provider_name}"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name)
        return os.path.join(self.data_dir, f"{safe_name}.json")

    @staticmethod
    def _atomic_write_json(path: str, data: Dict[str, Any]):
        """Atomically write json file to avoid partial-write corruption."""
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_usage_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def get_usage(self, model_name: str, provider_name: str) -> UsageData:
        """Read usage counters from disk."""
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"

        try:
            with FileLock(lock_path, timeout=5):
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return UsageData(**data)
        except Exception as exc:
            logger.warning("Failed to read usage data: %s", exc)

        return UsageData()

    def update_usage(self, model_name: str, provider_name: str, requests_delta: int = 0, tokens_delta: int = 0):
        """Update usage counters and persist to disk."""
        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"

        try:
            with FileLock(lock_path, timeout=5):
                usage = UsageData()
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            usage = UsageData(**data)
                    except Exception as exc:
                        logger.warning("Failed to read usage data when updating: %s", exc)

                usage.requests = max(0, usage.requests + requests_delta)
                usage.tokens = max(0, usage.tokens + tokens_delta)

                self._atomic_write_json(path, usage.model_dump())
        except Exception as exc:
            logger.error("Failed to update usage: %s", exc)

    def reset_usage(self, model_name: str, provider_name: str):
        """Reset usage counters for a model/provider."""
        from datetime import datetime

        path = self._get_usage_path(model_name, provider_name)
        lock_path = f"{path}.lock"

        try:
            with FileLock(lock_path, timeout=5):
                usage = UsageData(
                    requests=0,
                    tokens=0,
                    last_reset=datetime.now().isoformat(),
                )
                self._atomic_write_json(path, usage.model_dump())

            logger.info("Reset usage for %s/%s", model_name, provider_name)
        except Exception as exc:
            logger.error("Failed to reset usage: %s", exc)
