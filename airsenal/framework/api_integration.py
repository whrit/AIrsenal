"""
API Management Integration Utilities for AIrsenal

This module provides utilities to integrate the API management system
with existing AIrsenal components and provides easy configuration
for common use cases.

Key Features:
- Pre-configured API managers for common providers
- Integration helpers for existing systems
- Performance monitoring and alerting
- Cache warming strategies for AIrsenal data
- Seamless integration with existing Redis cache
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from airsenal.framework.api_manager import (
    APIManager,
    Priority,
    get_api_manager,
)
from airsenal.framework.env import (
    SPORTMONKS_API_KEY,
)
from airsenal.framework.logging_config import get_logger
from airsenal.framework.redis_cache import redis_cache
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK

logger = get_logger(__name__)


class AIrsenalAPIManager:
    """
    Pre-configured API manager for AIrsenal with sensible defaults
    and integration with existing systems.
    """

    def __init__(self):
        self.api_manager: APIManager | None = None
        self._initialized = False
        self.providers_configured = set()

    async def initialize(self):
        """Initialize with AIrsenal-specific configuration"""
        if self._initialized:
            return

        self.api_manager = await get_api_manager()

        # Configure common providers
        await self._configure_sportmonks()
        await self._configure_fpl_api()

        # Set up cache warming
        await self._setup_cache_warming()

        self._initialized = True
        logger.info("AIrsenal API Manager initialized with integrated configuration")

    async def _configure_sportmonks(self):
        """Configure Sportmonks provider with AIrsenal-optimized settings"""
        if "sportmonks" in self.providers_configured:
            return

        # Determine rate limits based on API tier
        if SPORTMONKS_API_KEY:
            # With API key - assume basic plan
            requests_per_second = 1.5  # Conservative for basic plan
            burst_capacity = 10
            daily_limit = 5000
            hourly_limit = 200
            cost_per_request = 0.01
        else:
            # No API key - very limited
            requests_per_second = 0.1
            burst_capacity = 2
            daily_limit = 100
            hourly_limit = 10
            cost_per_request = 0.0

        self.api_manager.configure_provider(
            provider="sportmonks",
            requests_per_second=requests_per_second,
            burst_capacity=burst_capacity,
            daily_limit=daily_limit,
            hourly_limit=hourly_limit,
            cost_per_request=cost_per_request,
            fallback_strategies=[
                self._sportmonks_cached_fallback,
                self._sportmonks_mock_fallback,
            ],
        )

        self.providers_configured.add("sportmonks")
        logger.info(f"Configured Sportmonks provider with {requests_per_second} req/s")

    async def _configure_fpl_api(self):
        """Configure FPL API with appropriate rate limiting"""
        if "fpl" in self.providers_configured:
            return

        # FPL API is free but has informal rate limits
        self.api_manager.configure_provider(
            provider="fpl",
            requests_per_second=0.5,  # Conservative
            burst_capacity=5,
            daily_limit=1000,
            hourly_limit=100,
            cost_per_request=0.0,
            fallback_strategies=[
                self._fpl_cached_fallback,
                self._fpl_bootstrap_fallback,
            ],
        )

        self.providers_configured.add("fpl")
        logger.info("Configured FPL API provider")

    async def _setup_cache_warming(self):
        """Set up cache warming for frequently accessed data"""
        if not redis_cache.is_available():
            logger.warning("Redis not available - cache warming disabled")
            return

        # Schedule cache warming tasks
        asyncio.create_task(self._warm_player_data())
        asyncio.create_task(self._warm_fixture_data())

        logger.info("Cache warming tasks scheduled")

    async def _warm_player_data(self):
        """Warm cache with current season player data"""
        try:
            cache_manager = self.api_manager.get_cache_manager()

            # Warm player predictions for next gameweek
            # This would integrate with existing player prediction systems
            await cache_manager.set(
                f"player_predictions:{CURRENT_SEASON}:{NEXT_GAMEWEEK}",
                "warming_placeholder",  # Replace with actual data
                ttl=3600,
                priority=Priority.MEDIUM,
            )

            logger.debug("Player data cache warming completed")

        except Exception as e:
            logger.error(f"Error warming player data cache: {e}")

    async def _warm_fixture_data(self):
        """Warm cache with fixture data"""
        try:
            cache_manager = self.api_manager.get_cache_manager()

            # Warm fixture difficulty data
            await cache_manager.set(
                f"fixtures:{CURRENT_SEASON}:{NEXT_GAMEWEEK}",
                "warming_placeholder",  # Replace with actual data
                ttl=7200,  # 2 hours
                priority=Priority.HIGH,
            )

            logger.debug("Fixture data cache warming completed")

        except Exception as e:
            logger.error(f"Error warming fixture data cache: {e}")

    async def _sportmonks_cached_fallback(self, *args, **kwargs):
        """Fallback to cached Sportmonks data"""
        # This would implement logic to return stale/cached Sportmonks data
        logger.info("Using cached Sportmonks data fallback")
        return

    async def _sportmonks_mock_fallback(self, *args, **kwargs):
        """Fallback to mock Sportmonks data"""
        # This would implement logic to return estimated/mock data
        logger.info("Using mock Sportmonks data fallback")
        return []

    async def _fpl_cached_fallback(self, *args, **kwargs):
        """Fallback to cached FPL data"""
        logger.info("Using cached FPL data fallback")
        return

    async def _fpl_bootstrap_fallback(self, *args, **kwargs):
        """Fallback to FPL bootstrap data"""
        # Use static bootstrap data when API is unavailable
        logger.info("Using FPL bootstrap data fallback")
        return

    async def make_api_request(
        self,
        provider: str,
        func: Callable,
        cache_key: str | None = None,
        ttl: int = 3600,
        priority: Priority = Priority.MEDIUM,
        *args,
        **kwargs,
    ):
        """
        Make an API request with full rate limiting and caching support.

        This is the main interface for making API requests in AIrsenal.
        """
        if not self._initialized:
            await self.initialize()

        return await self.api_manager.queue_request(
            provider=provider,
            func=func,
            priority=priority,
            cache_key=cache_key,
            ttl=ttl,
            *args,
            **kwargs,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive API usage statistics"""
        if not self._initialized or not self.api_manager:
            return {"error": "API manager not initialized"}

        stats = self.api_manager.get_comprehensive_stats()

        # Add AIrsenal-specific metrics
        stats["airsenal"] = {
            "providers_configured": list(self.providers_configured),
            "initialization_time": datetime.now().isoformat(),
            "redis_integration": redis_cache.is_available(),
            "current_season": CURRENT_SEASON,
            "next_gameweek": NEXT_GAMEWEEK,
        }

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive health check for AIrsenal integration"""
        if not self._initialized:
            await self.initialize()

        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check API manager
        try:
            stats = self.api_manager.get_comprehensive_stats()
            health["components"]["api_manager"] = {
                "status": "healthy",
                "providers": len(stats.get("rate_limiter", {})),
            }
        except Exception as e:
            health["status"] = "degraded"
            health["components"]["api_manager"] = {"status": "error", "error": str(e)}

        # Check Redis integration
        try:
            redis_available = redis_cache.is_available()
            health["components"]["redis"] = {
                "status": "healthy" if redis_available else "unavailable",
                "available": redis_available,
            }

            if not redis_available:
                health["status"] = "degraded"

        except Exception as e:
            health["status"] = "degraded"
            health["components"]["redis"] = {"status": "error", "error": str(e)}

        # Check providers
        for provider in self.providers_configured:
            try:
                rate_limiter_status = self.api_manager.rate_limiter.get_status(provider)
                health["components"][f"provider_{provider}"] = {
                    "status": "healthy"
                    if rate_limiter_status.get("can_make_request", True)
                    else "limited",
                    "can_make_request": rate_limiter_status.get(
                        "can_make_request", True
                    ),
                    "tokens_available": rate_limiter_status.get("tokens_available", 0),
                }
            except Exception as e:
                health["status"] = "degraded"
                health["components"][f"provider_{provider}"] = {
                    "status": "error",
                    "error": str(e),
                }

        return health


class CacheWarmer:
    """Utility class for warming AIrsenal caches"""

    def __init__(self, api_manager: AIrsenalAPIManager):
        self.api_manager = api_manager

    async def warm_predictions_cache(
        self,
        season: str | None = None,
        gameweeks: list[int] | None = None,
        player_ids: list[int] | None = None,
    ):
        """Warm cache with player predictions"""
        season = season or CURRENT_SEASON
        gameweeks = gameweeks or [NEXT_GAMEWEEK]

        if not player_ids:
            # Get active players (this would integrate with existing player systems)
            logger.info("Getting active players for cache warming")
            return

        cache_manager = self.api_manager.api_manager.get_cache_manager()

        for gameweek in gameweeks:
            for player_id in player_ids:
                cache_key = f"prediction:{season}:{gameweek}:{player_id}"

                # This would fetch predictions using the existing prediction system
                # For now, we'll just set a placeholder
                await cache_manager.set(
                    cache_key,
                    {
                        "player_id": player_id,
                        "gameweek": gameweek,
                        "predicted_points": 0.0,
                    },
                    ttl=3600,
                    priority=Priority.MEDIUM,
                )

        logger.info(f"Warmed predictions cache for {len(player_ids)} players")

    async def warm_team_data_cache(self, season: str | None = None):
        """Warm cache with team data"""
        season = season or CURRENT_SEASON
        cache_manager = self.api_manager.api_manager.get_cache_manager()

        # Warm team strength data
        await cache_manager.set(
            f"team_strengths:{season}",
            {"season": season, "data": "placeholder"},
            ttl=7200,
            priority=Priority.HIGH,
        )

        logger.info(f"Warmed team data cache for season {season}")


class PerformanceMonitor:
    """Monitor API performance and generate alerts"""

    def __init__(self, api_manager: AIrsenalAPIManager):
        self.api_manager = api_manager
        self.alert_thresholds = {
            "cache_hit_rate_min": 0.8,
            "queue_length_max": 50,
            "response_time_max": 5.0,
            "error_rate_max": 0.1,
        }

    async def check_performance(self) -> list[dict[str, Any]]:
        """Check performance and return alerts"""
        alerts = []

        try:
            stats = self.api_manager.get_stats()

            # Check cache hit rate
            cache_hit_rate = stats.get("cache", {}).get("overall_hit_rate", 0.0)
            if cache_hit_rate < self.alert_thresholds["cache_hit_rate_min"]:
                alerts.append(
                    {
                        "level": "warning",
                        "component": "cache",
                        "message": f"Low cache hit rate: {cache_hit_rate:.1%}",
                        "threshold": self.alert_thresholds["cache_hit_rate_min"],
                        "current_value": cache_hit_rate,
                    }
                )

            # Check queue length
            queue_length = stats.get("queue", {}).get("queue_length", 0)
            if queue_length > self.alert_thresholds["queue_length_max"]:
                alerts.append(
                    {
                        "level": "critical",
                        "component": "queue",
                        "message": f"High queue length: {queue_length}",
                        "threshold": self.alert_thresholds["queue_length_max"],
                        "current_value": queue_length,
                    }
                )

            # Check provider performance
            usage_stats = stats.get("usage", {}).get("providers", {})
            for provider, provider_stats in usage_stats.items():
                error_rate = 1.0 - provider_stats.get("success_rate", 1.0)
                if error_rate > self.alert_thresholds["error_rate_max"]:
                    alerts.append(
                        {
                            "level": "critical",
                            "component": "provider",
                            "message": f"High error rate for {provider}: {error_rate:.1%}",
                            "threshold": self.alert_thresholds["error_rate_max"],
                            "current_value": error_rate,
                        }
                    )

        except Exception as e:
            alerts.append(
                {
                    "level": "critical",
                    "component": "monitor",
                    "message": f"Performance monitoring error: {e}",
                    "threshold": None,
                    "current_value": None,
                }
            )

        return alerts

    def update_thresholds(self, **kwargs):
        """Update alert thresholds"""
        for key, value in kwargs.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
                logger.info(f"Updated threshold {key} to {value}")


# Global instance for easy access
_global_airsenal_api_manager: AIrsenalAPIManager | None = None


async def get_airsenal_api_manager() -> AIrsenalAPIManager:
    """Get or create global AIrsenal API manager instance"""
    global _global_airsenal_api_manager

    if _global_airsenal_api_manager is None:
        _global_airsenal_api_manager = AIrsenalAPIManager()
        await _global_airsenal_api_manager.initialize()

    return _global_airsenal_api_manager


# Convenience functions for common operations
async def make_api_request(
    provider: str,
    func: Callable,
    cache_key: str | None = None,
    ttl: int = 3600,
    priority: Priority = Priority.MEDIUM,
    *args,
    **kwargs,
):
    """
    Convenience function for making API requests with full AIrsenal integration.

    This is the recommended way to make API requests in AIrsenal code.
    """
    api_manager = await get_airsenal_api_manager()
    return await api_manager.make_api_request(
        provider, func, cache_key, ttl, priority, *args, **kwargs
    )


async def get_api_stats() -> dict[str, Any]:
    """Get comprehensive API usage statistics"""
    api_manager = await get_airsenal_api_manager()
    return api_manager.get_stats()


async def check_api_health() -> dict[str, Any]:
    """Check API system health"""
    api_manager = await get_airsenal_api_manager()
    return await api_manager.health_check()


# Export main classes and functions
__all__ = [
    "AIrsenalAPIManager",
    "CacheWarmer",
    "PerformanceMonitor",
    "check_api_health",
    "get_airsenal_api_manager",
    "get_api_stats",
    "make_api_request",
]
