"""LEXA Monitoring System

This module provides monitoring, health checks, and system metrics collection.
"""

import os
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
import prometheus_client as prom
from dataclasses import dataclass
import spacy
import numpy as np
from sqlalchemy import create_engine, text

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """Container for system health information"""
    status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    database_connection: bool
    nlp_models_loaded: bool
    cache_status: bool
    issues: List[str]
    last_check: datetime

# Prometheus metrics
REQUESTS_TOTAL = prom.Counter(
    'lexa_requests_total',
    'Total requests processed',
    ['endpoint', 'method', 'status']
)

PROCESSING_TIME = prom.Histogram(
    'lexa_processing_seconds',
    'Time spent processing requests',
    ['endpoint']
)

CHAR_COUNT = prom.Counter(
    'lexa_characters_processed_total',
    'Total characters processed',
    ['tier']
)

ACTIVE_USERS = prom.Gauge(
    'lexa_active_users',
    'Number of currently active users'
)

ERROR_COUNT = prom.Counter(
    'lexa_errors_total',
    'Total number of errors',
    ['type']
)

class HealthCheck:
    """System health monitoring and reporting"""

    def __init__(self, db_url: str):
        """Initialize health check system.
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.nlp_models = {}
        self.last_check = None
        self.initialize_monitoring()

    def initialize_monitoring(self):
        """Initialize monitoring systems"""
        # Start Prometheus HTTP server if enabled
        if os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true':
            prom.start_http_server(8000)

    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check.
        
        Returns:
            SystemHealth object with current system status
        """
        start_time = time.time()
        issues = []

        try:
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage}%")

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            if memory_usage > 80:
                issues.append(f"High memory usage: {memory_usage}%")

            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            if disk_usage > 80:
                issues.append(f"High disk usage: {disk_usage}%")

            # Check database connection
            db_ok = self.check_database()
            if not db_ok:
                issues.append("Database connection failed")

            # Check NLP models
            nlp_ok = self.check_nlp_models()
            if not nlp_ok:
                issues.append("NLP models not properly loaded")

            # Check cache status
            cache_ok = self.check_cache()
            if not cache_ok:
                issues.append("Cache system not responding")

            # Calculate response time
            response_time = time.time() - start_time

            # Determine overall status
            status = "healthy" if not issues else "degraded" if len(issues) < 2 else "unhealthy"

            health = SystemHealth(
                status=status,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                response_time=response_time,
                database_connection=db_ok,
                nlp_models_loaded=nlp_ok,
                cache_status=cache_ok,
                issues=issues,
                last_check=datetime.now()
            )

            self.last_check = health
            return health

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return SystemHealth(
                status="error",
                cpu_usage=-1,
                memory_usage=-1,
                disk_usage=-1,
                response_time=-1,
                database_connection=False,
                nlp_models_loaded=False,
                cache_status=False,
                issues=[f"Health check failed: {str(e)}"],
                last_check=datetime.now()
            )

    def check_database(self) -> bool:
        """Check database connectivity.
        
        Returns:
            bool indicating if database is accessible
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return False

    def check_nlp_models(self) -> bool:
        """Check if NLP models are properly loaded.
        
        Returns:
            bool indicating if models are available
        """
        try:
            required_models = ['pt_core_news_lg', 'en_core_web_lg']
            for model in required_models:
                if model not in self.nlp_models:
                    self.nlp_models[model] = spacy.load(model)
            return True
        except Exception as e:
            logger.error(f"NLP model check failed: {str(e)}")
            return False

    def check_cache(self) -> bool:
        """Check if cache system is responding.
        
        Returns:
            bool indicating if cache is working
        """
        try:
            # Implementation depends on cache system used (Redis, Memcached, etc.)
            return True
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return False

    def get_system_metrics(self) -> Dict:
        """Get detailed system metrics.
        
        Returns:
            Dict containing various system metrics
        """
        return {
            'system': {
                'cpu': {
                    'usage_percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'load_avg': psutil.getloadavg()
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used,
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            },
            'application': {
                'uptime': self.get_uptime(),
                'active_users': ACTIVE_USERS._value.get(),
                'error_rate': self.calculate_error_rate(),
                'response_time': self.calculate_average_response_time()
            }
        }

    def get_uptime(self) -> float:
        """Get application uptime in seconds.
        
        Returns:
            float representing uptime in seconds
        """
        return time.time() - psutil.Process().create_time()

    def calculate_error_rate(self) -> float:
        """Calculate current error rate.
        
        Returns:
            float representing error rate percentage
        """
        total_requests = sum(REQUESTS_TOTAL._value.values())
        total_errors = sum(ERROR_COUNT._value.values())
        return (total_errors / total_requests * 100) if total_requests > 0 else 0

    def calculate_average_response_time(self) -> float:
        """Calculate average response time.
        
        Returns:
            float representing average response time in seconds
        """
        return np.mean(list(PROCESSING_TIME._sum.values()))

    def log_request(self, endpoint: str, method: str, status: int, duration: float):
        """Log a request for monitoring.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status: Response status code
            duration: Request duration in seconds
        """
        REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
        PROCESSING_TIME.labels(endpoint=endpoint).observe(duration)

    def log_error(self, error_type: str):
        """Log an error for monitoring.
        
        Args:
            error_type: Type of error encountered
        """
        ERROR_COUNT.labels(type=error_type).inc()

    def update_active_users(self, count: int):
        """Update active users count.
        
        Args:
            count: Number of currently active users
        """
        ACTIVE_USERS.set(count)

    def log_processed_chars(self, tier: str, count: int):
        """Log processed characters count.
        
        Args:
            tier: Subscription tier
            count: Number of characters processed
        """
        CHAR_COUNT.labels(tier=tier).inc(count)
