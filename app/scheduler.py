"""
定时任务调度器 - 处理限额刷新
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from croniter import croniter

from .config import ConfigManager, UsageManager

logger = logging.getLogger(__name__)


class UsageResetScheduler:
    """使用量重置调度器"""
    
    def __init__(self, config_manager: ConfigManager, usage_manager: UsageManager):
        self.config_manager = config_manager
        self.usage_manager = usage_manager
        self.scheduler = BackgroundScheduler()
        self._jobs = {}
    
    def start(self):
        """启动调度器"""
        self._setup_jobs()
        self.scheduler.start()
        logger.info("Usage reset scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.scheduler.shutdown()
        logger.info("Usage reset scheduler stopped")
    
    def _setup_jobs(self):
        """设置定时任务"""
        for model_name in self.config_manager.get_all_models():
            for provider in self.config_manager.get_providers(model_name):
                if provider.rate_limit and provider.rate_limit.period_cron:
                    self._add_reset_job(model_name, provider.name, 
                                       provider.rate_limit.period_cron)
    
    def _add_reset_job(self, model_name: str, provider_name: str, cron_expr: str):
        """添加重置任务"""
        job_id = f"reset_{model_name}_{provider_name}"
        
        try:
            # 解析 cron 表达式
            parts = cron_expr.split()
            if len(parts) != 5:
                logger.error(f"Invalid cron expression: {cron_expr}")
                return
            
            minute, hour, day, month, day_of_week = parts
            
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week
            )
            
            self.scheduler.add_job(
                self._reset_usage,
                trigger=trigger,
                args=[model_name, provider_name],
                id=job_id,
                replace_existing=True
            )
            
            # 计算下次执行时间
            cron = croniter(cron_expr, datetime.now())
            next_run = cron.get_next(datetime)
            
            logger.info(f"Scheduled reset job for {model_name}/{provider_name}, "
                       f"cron: {cron_expr}, next run: {next_run}")
            
        except Exception as e:
            logger.error(f"Failed to add reset job for {model_name}/{provider_name}: {e}")
    
    def _reset_usage(self, model_name: str, provider_name: str):
        """执行使用量重置"""
        logger.info(f"Resetting usage for {model_name}/{provider_name}")
        self.usage_manager.reset_usage(model_name, provider_name)
    
    def reload(self):
        """重新加载定时任务"""
        # 移除所有现有任务
        for job_id in list(self._jobs.keys()):
            try:
                self.scheduler.remove_job(job_id)
            except:
                pass
        self._jobs.clear()
        
        # 重新设置任务
        self._setup_jobs()
        logger.info("Scheduler reloaded")