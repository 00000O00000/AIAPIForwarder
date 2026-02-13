"""Schedule periodic usage resets for provider rate limits."""

import logging
from datetime import datetime
from typing import Set

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from croniter import croniter

from .config import ConfigManager, UsageManager

logger = logging.getLogger(__name__)


class UsageResetScheduler:
    """Background scheduler for resetting provider usage counters."""

    def __init__(self, config_manager: ConfigManager, usage_manager: UsageManager):
        self.config_manager = config_manager
        self.usage_manager = usage_manager
        self.scheduler = BackgroundScheduler()
        self._jobs: Set[str] = set()

    def start(self):
        """Start scheduler and register jobs once."""
        if self.scheduler.running:
            logger.info("Usage reset scheduler already running")
            return

        self._setup_jobs()
        self.scheduler.start()
        logger.info("Usage reset scheduler started")

    def stop(self):
        """Stop scheduler if running."""
        if not self.scheduler.running:
            logger.info("Usage reset scheduler already stopped")
            return

        self.scheduler.shutdown()
        logger.info("Usage reset scheduler stopped")

    def _setup_jobs(self):
        """Register reset jobs for all providers with rate-limit cron configured."""
        for model_name in self.config_manager.get_all_models():
            for provider in self.config_manager.get_providers(model_name):
                if provider.rate_limit and provider.rate_limit.period_cron:
                    self._add_reset_job(model_name, provider.name, provider.rate_limit.period_cron)

    def _add_reset_job(self, model_name: str, provider_name: str, cron_expr: str):
        """Add one reset job for a model/provider pair."""
        job_id = f"reset_{model_name}_{provider_name}"
        cron_expr = (cron_expr or "").strip()

        try:
            if not croniter.is_valid(cron_expr):
                logger.error("Invalid cron expression: %s", cron_expr)
                return

            parts = cron_expr.split()
            if len(parts) != 5:
                logger.error("Invalid cron expression: %s", cron_expr)
                return

            minute, hour, day, month, day_of_week = parts
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
            )

            self.scheduler.add_job(
                self._reset_usage,
                trigger=trigger,
                args=[model_name, provider_name],
                id=job_id,
                replace_existing=True,
            )
            self._jobs.add(job_id)

            next_run = croniter(cron_expr, datetime.now()).get_next(datetime)
            logger.info(
                "Scheduled reset job for %s/%s, cron: %s, next run: %s",
                model_name,
                provider_name,
                cron_expr,
                next_run,
            )
        except Exception as exc:
            logger.error("Failed to add reset job for %s/%s: %s", model_name, provider_name, exc)

    def _reset_usage(self, model_name: str, provider_name: str):
        """Reset usage counters for one provider."""
        logger.info("Resetting usage for %s/%s", model_name, provider_name)
        self.usage_manager.reset_usage(model_name, provider_name)

    def reload(self):
        """Reload scheduler jobs from current config."""
        self._clear_jobs()
        self._setup_jobs()
        logger.info("Scheduler reloaded, %s jobs configured", len(self._jobs))

    def _clear_jobs(self):
        """Remove all known scheduler jobs."""
        for job_id in list(self._jobs):
            try:
                self.scheduler.remove_job(job_id)
                logger.debug("Removed job: %s", job_id)
            except Exception:
                pass
        self._jobs.clear()
