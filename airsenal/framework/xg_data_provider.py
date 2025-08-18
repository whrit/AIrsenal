"""
xG/xA Data Provider Integration for AIrsenal

This module provides a flexible interface for integrating with external xG/xA data providers,
with built-in retry logic, circuit breaker patterns, and comprehensive monitoring.

Designed for Sportmonks but extensible to other providers like FBRef, Opta, etc.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from urllib3.util.retry import Retry

from airsenal.framework.api_manager import Priority, get_api_manager
from airsenal.framework.env import SPORTMONKS_API_KEY
from airsenal.framework.logging_config import get_logger
from airsenal.framework.logging_utils import log_api_call, timed
from airsenal.framework.schema import Fixture, PlayerScore, session_scope

logger = get_logger(__name__)


class ProviderTier(Enum):
    """Sportmonks API tiers with different data availability and delays"""

    BASIC = "basic"  # 12hr delay
    STANDARD = "standard"  # post-match
    ADVANCED = "advanced"  # live data


class CircuitBreakerState(Enum):
    """Circuit breaker states for API resilience"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ProviderConfig:
    """Configuration for xG/xA data providers"""

    api_key: str | None = None
    base_url: str = ""
    rate_limit_per_minute: int = 100
    timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    tier: ProviderTier = ProviderTier.BASIC
    cost_per_request: float = 0.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.rate_limit_per_minute <= 0:
            msg = "Rate limit must be positive"
            raise ValueError(msg)
        if self.timeout_seconds <= 0:
            msg = "Timeout must be positive"
            raise ValueError(msg)


@dataclass
class XGDataPoint:
    """Standardized xG/xA data structure"""

    player_id: int
    fixture_id: int
    match_date: datetime
    expected_goals: float | None = None
    expected_assists: float | None = None
    expected_goal_involvements: float | None = None
    expected_goals_conceded: float | None = None
    # Raw provider data for debugging/analysis
    raw_data: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    data_source_tier: ProviderTier = ProviderTier.BASIC

    def __post_init__(self):
        """Calculate derived metrics"""
        if (
            self.expected_goals is not None
            and self.expected_assists is not None
            and self.expected_goal_involvements is None
        ):
            self.expected_goal_involvements = (
                self.expected_goals + self.expected_assists
            )


@dataclass
class APIUsageMetrics:
    """Track API usage and costs"""

    provider_name: str
    requests_made: int = 0
    requests_failed: int = 0
    total_cost: float = 0.0
    daily_limit: int = 1000
    current_window_start: datetime = field(default_factory=datetime.now)
    current_window_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: datetime | None = None
    rate_limit_hits: int = 0

    def add_request(self, success: bool, response_time: float, cost: float = 0.0):
        """Record a new API request"""
        self.requests_made += 1
        if not success:
            self.requests_failed += 1
        self.total_cost += cost
        self.last_request_time = datetime.now()

        # Update current window for rate limiting
        if (datetime.now() - self.current_window_start).total_seconds() >= 60:
            self.current_window_start = datetime.now()
            self.current_window_requests = 0
        self.current_window_requests += 1

        # Update average response time
        if self.requests_made == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                self.average_response_time * (self.requests_made - 1) + response_time
            ) / self.requests_made

    @property
    def success_rate(self) -> float:
        """Calculate API success rate"""
        if self.requests_made == 0:
            return 1.0
        return (self.requests_made - self.requests_failed) / self.requests_made

    @property
    def requests_remaining_in_window(self) -> int:
        """Calculate remaining requests in current rate limit window"""
        return max(0, self.daily_limit - self.current_window_requests)


class CircuitBreaker:
    """Circuit breaker implementation for API resilience"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = CircuitBreakerState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                msg = "Circuit breaker OPEN. Service unavailable."
                raise Exception(msg)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return (
            datetime.now() - self.last_failure_time
        ).total_seconds() >= self.recovery_timeout

    def _on_success(self):
        """Handle successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def _on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )


class XGDataProvider(ABC):
    """Abstract base class for xG/xA data providers"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.metrics = APIUsageMetrics(provider_name=self.__class__.__name__)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
        )
        self._setup_session()
        logger.info(
            f"Initialized {self.__class__.__name__} with tier {config.tier.value}"
        )

    def _setup_session(self):
        """Configure requests session with retry strategy"""
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,  # Exponential backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default timeout
        self.session.timeout = self.config.timeout_seconds

    @abstractmethod
    async def get_player_xg_data(
        self,
        player_id: int,
        fixture_id: int | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[XGDataPoint]:
        """Fetch xG/xA data for a specific player"""

    @abstractmethod
    async def get_match_xg_data(self, fixture_id: int) -> list[XGDataPoint]:
        """Fetch xG/xA data for all players in a specific match"""

    @abstractmethod
    async def get_bulk_xg_data(
        self, date_range: tuple[datetime, datetime], gameweeks: list[int] | None = None
    ) -> list[XGDataPoint]:
        """Fetch bulk xG/xA data for a date range"""

    def _make_request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make rate-limited API request with error handling"""
        if not self._check_rate_limit():
            msg = "Rate limit exceeded"
            raise Exception(msg)

        if not self.config.api_key and not self._is_credentials_optional():
            logger.warning(
                "No API key configured - some providers may return limited data"
            )

        # Add authentication headers
        request_headers = headers or {}
        if self.config.api_key:
            request_headers.update(self._get_auth_headers())

        start_time = time.time()

        def _request():
            return self.session.get(
                url,
                params=params,
                headers=request_headers,
                timeout=self.config.timeout_seconds,
            )

        try:
            response = self.circuit_breaker.call(_request)
            response_time = time.time() - start_time

            response.raise_for_status()

            # Log successful request
            self.metrics.add_request(
                success=True,
                response_time=response_time,
                cost=self.config.cost_per_request,
            )

            log_api_call(
                provider=self.__class__.__name__,
                endpoint=url,
                status_code=response.status_code,
                response_time=response_time,
            )

            return response.json()

        except (RequestException, HTTPError, ConnectionError, Timeout) as e:
            response_time = time.time() - start_time
            self.metrics.add_request(success=False, response_time=response_time)

            logger.error(
                f"API request failed: {url}",
                extra={
                    "error": str(e),
                    "provider": self.__class__.__name__,
                    "response_time": response_time,
                    "circuit_breaker_state": self.circuit_breaker.state.value,
                },
            )
            raise e

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        if self.metrics.requests_remaining_in_window <= 0:
            logger.warning(f"Rate limit hit for {self.__class__.__name__}")
            self.metrics.rate_limit_hits += 1
            return False
        return True

    @abstractmethod
    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests"""

    @abstractmethod
    def _is_credentials_optional(self) -> bool:
        """Whether the provider works without credentials (with limitations)"""


class SportmonksProvider(XGDataProvider):
    """Sportmonks API implementation for xG/xA data"""

    def __init__(self, config: ProviderConfig | None = None):
        if config is None:
            # Use API key from environment if available
            api_key = SPORTMONKS_API_KEY
            if api_key is None:
                logger.info("No Sportmonks API key found in environment")

            config = ProviderConfig(
                api_key=api_key,
                base_url="https://api.sportmonks.com/v3/football/",
                rate_limit_per_minute=100,  # Adjust based on plan
                timeout_seconds=30,
                tier=ProviderTier.BASIC,
                cost_per_request=0.01,  # Estimate, adjust based on plan
            )

        super().__init__(config)
        self.transform = DataTransformer()
        self._api_manager = None
        self._initialized = False

    async def _ensure_api_manager(self):
        """Ensure API manager is initialized"""
        if not self._initialized:
            self._api_manager = await get_api_manager()

            # Configure the provider in API manager
            await self._api_manager.configure_provider(
                provider="sportmonks",
                requests_per_second=self.config.rate_limit_per_minute / 60.0,
                burst_capacity=10,
                daily_limit=10000,  # Adjust based on plan
                hourly_limit=self.config.rate_limit_per_minute,
                cost_per_request=self.config.cost_per_request,
                fallback_strategies=[self._cached_fallback, self._mock_fallback],
            )

            self._initialized = True
            logger.info("SportmonksProvider integrated with APIManager")

    async def _cached_fallback(self, *args, **kwargs):
        """Fallback to cached data when API is unavailable"""
        # This would implement logic to return cached/stale data
        logger.info("Using cached fallback for Sportmonks API")
        return

    async def _mock_fallback(self, *args, **kwargs):
        """Fallback to mock data when API and cache are unavailable"""
        # This would implement logic to return mock/estimated data
        logger.info("Using mock fallback for Sportmonks API")
        return []

    @timed
    async def get_player_xg_data(
        self,
        player_id: int,
        fixture_id: int | None = None,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> list[XGDataPoint]:
        """Fetch xG/xA data for a specific player from Sportmonks"""
        await self._ensure_api_manager()

        try:
            endpoint = f"players/{player_id}/statistics"
            params = {"include": "statistictype,team"}

            if fixture_id:
                params["filter"] = f"fixture_id:{fixture_id}"
            elif date_range:
                start_date = date_range[0].strftime("%Y-%m-%d")
                end_date = date_range[1].strftime("%Y-%m-%d")
                params["filter"] = f"date:gte:{start_date};date:lte:{end_date}"

            # Generate cache key
            cache_key = f"sportmonks:player_xg:{player_id}:{fixture_id or 'all'}:{date_range or 'all'}"

            url = urljoin(self.config.base_url, endpoint)

            # Use API manager for rate-limited request
            response_data = await self._api_manager.queue_request(
                provider="sportmonks",
                func=self._make_request_sync,
                priority=Priority.MEDIUM,
                cache_key=cache_key,
                ttl=3600,  # Cache for 1 hour
                url=url,
                params=params,
            )

            return self.transform.sportmonks_to_xg_data(response_data, player_id)

        except Exception as e:
            logger.error(
                f"Failed to fetch xG data for player {player_id}",
                extra={"error": str(e), "fixture_id": fixture_id},
            )
            return []

    def _make_request_sync(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for _make_request method"""
        return self._make_request(url, params, headers)

    @timed
    async def get_match_xg_data(self, fixture_id: int) -> list[XGDataPoint]:
        """Fetch xG/xA data for all players in a specific match"""
        await self._ensure_api_manager()

        try:
            endpoint = f"fixtures/{fixture_id}/statistics"
            params = {"include": "statistictype,player,team"}

            # Generate cache key
            cache_key = f"sportmonks:match_xg:{fixture_id}"

            url = urljoin(self.config.base_url, endpoint)

            # Use API manager for rate-limited request
            response_data = await self._api_manager.queue_request(
                provider="sportmonks",
                func=self._make_request_sync,
                priority=Priority.HIGH,  # Match data is high priority
                cache_key=cache_key,
                ttl=7200,  # Cache for 2 hours
                url=url,
                params=params,
            )

            return self.transform.sportmonks_match_to_xg_data(response_data, fixture_id)

        except Exception as e:
            logger.error(
                f"Failed to fetch match xG data for fixture {fixture_id}",
                extra={"error": str(e)},
            )
            return []

    @timed
    async def get_bulk_xg_data(
        self, date_range: tuple[datetime, datetime], gameweeks: list[int] | None = None
    ) -> list[XGDataPoint]:
        """Fetch bulk xG/xA data for a date range"""
        await self._ensure_api_manager()

        try:
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")

            endpoint = "fixtures"
            params = {
                "include": "statistics.statistictype,statistics.player",
                "filter": f"date:gte:{start_date};date:lte:{end_date}",
                "per_page": 100,
            }

            # Generate cache key
            cache_key = (
                f"sportmonks:bulk_xg:{start_date}:{end_date}:{gameweeks or 'all'}"
            )

            url = urljoin(self.config.base_url, endpoint)
            all_data = []
            page = 1

            while True:
                params["page"] = page

                # Use API manager for rate-limited request
                response_data = await self._api_manager.queue_request(
                    provider="sportmonks",
                    func=self._make_request_sync,
                    priority=Priority.LOW,  # Bulk requests are low priority
                    cache_key=f"{cache_key}:page_{page}",
                    ttl=14400,  # Cache for 4 hours
                    url=url,
                    params=params.copy(),
                )

                if not response_data.get("data"):
                    break

                page_xg_data = self.transform.sportmonks_bulk_to_xg_data(
                    response_data, gameweeks
                )
                all_data.extend(page_xg_data)

                # Check if there are more pages
                if not response_data.get("meta", {}).get("has_more_pages", False):
                    break

                page += 1
                # The API manager handles rate limiting, so no manual delay needed

            logger.info(
                f"Fetched {len(all_data)} xG data points for date range {start_date} to {end_date}"
            )
            return all_data

        except Exception as e:
            logger.error(
                f"Failed to fetch bulk xG data for date range {date_range}",
                extra={"error": str(e)},
            )
            return []

    def _get_auth_headers(self) -> dict[str, str]:
        """Get Sportmonks authentication headers"""
        if not self.config.api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _is_credentials_optional(self) -> bool:
        """Sportmonks requires API key for most data"""
        return False


class DataTransformer:
    """Transform provider-specific data to AIrsenal format"""

    def sportmonks_to_xg_data(
        self, response_data: dict[str, Any], player_id: int
    ) -> list[XGDataPoint]:
        """Transform Sportmonks player statistics to XGDataPoint format"""
        xg_data = []

        try:
            statistics = response_data.get("data", [])

            for stat in statistics:
                fixture_id = stat.get("fixture_id")
                if not fixture_id:
                    continue

                # Extract xG/xA from statistics
                xg_stats = self._extract_xg_stats_from_sportmonks(stat)
                if not xg_stats:
                    continue

                xg_point = XGDataPoint(
                    player_id=player_id,
                    fixture_id=fixture_id,
                    match_date=self._parse_sportmonks_date(stat.get("date")),
                    expected_goals=xg_stats.get("expected_goals"),
                    expected_assists=xg_stats.get("expected_assists"),
                    expected_goal_involvements=xg_stats.get(
                        "expected_goal_involvements"
                    ),
                    expected_goals_conceded=xg_stats.get("expected_goals_conceded"),
                    raw_data=stat,
                    provider="Sportmonks",
                    data_source_tier=self.config.tier
                    if hasattr(self, "config")
                    else ProviderTier.BASIC,
                )

                xg_data.append(xg_point)

        except Exception as e:
            logger.error(
                f"Error transforming Sportmonks data for player {player_id}",
                extra={"error": str(e), "response_data": response_data},
            )

        return xg_data

    def sportmonks_match_to_xg_data(
        self, response_data: dict[str, Any], fixture_id: int
    ) -> list[XGDataPoint]:
        """Transform Sportmonks match statistics to XGDataPoint format"""
        xg_data = []

        try:
            statistics = response_data.get("data", [])

            for stat in statistics:
                player_data = stat.get("player")
                if not player_data:
                    continue

                player_id = player_data.get("id")
                if not player_id:
                    continue

                xg_stats = self._extract_xg_stats_from_sportmonks(stat)
                if not xg_stats:
                    continue

                xg_point = XGDataPoint(
                    player_id=player_id,
                    fixture_id=fixture_id,
                    match_date=self._parse_sportmonks_date(stat.get("date")),
                    expected_goals=xg_stats.get("expected_goals"),
                    expected_assists=xg_stats.get("expected_assists"),
                    expected_goal_involvements=xg_stats.get(
                        "expected_goal_involvements"
                    ),
                    expected_goals_conceded=xg_stats.get("expected_goals_conceded"),
                    raw_data=stat,
                    provider="Sportmonks",
                    data_source_tier=self.config.tier
                    if hasattr(self, "config")
                    else ProviderTier.BASIC,
                )

                xg_data.append(xg_point)

        except Exception as e:
            logger.error(
                f"Error transforming Sportmonks match data for fixture {fixture_id}",
                extra={"error": str(e), "response_data": response_data},
            )

        return xg_data

    def sportmonks_bulk_to_xg_data(
        self, response_data: dict[str, Any], gameweeks: list[int] | None = None
    ) -> list[XGDataPoint]:
        """Transform Sportmonks bulk fixture data to XGDataPoint format"""
        xg_data = []

        try:
            fixtures = response_data.get("data", [])

            for fixture in fixtures:
                fixture_id = fixture.get("id")
                if not fixture_id:
                    continue

                # Filter by gameweeks if specified
                if gameweeks:
                    fixture_gameweek = fixture.get("round", {}).get("name")
                    if fixture_gameweek:
                        try:
                            gw_num = int(fixture_gameweek.split()[-1])
                            if gw_num not in gameweeks:
                                continue
                        except (ValueError, IndexError):
                            continue

                statistics = fixture.get("statistics", [])
                for stat in statistics:
                    player_data = stat.get("player")
                    if not player_data:
                        continue

                    player_id = player_data.get("id")
                    if not player_id:
                        continue

                    xg_stats = self._extract_xg_stats_from_sportmonks(stat)
                    if not xg_stats:
                        continue

                    xg_point = XGDataPoint(
                        player_id=player_id,
                        fixture_id=fixture_id,
                        match_date=self._parse_sportmonks_date(
                            fixture.get("starting_at")
                        ),
                        expected_goals=xg_stats.get("expected_goals"),
                        expected_assists=xg_stats.get("expected_assists"),
                        expected_goal_involvements=xg_stats.get(
                            "expected_goal_involvements"
                        ),
                        expected_goals_conceded=xg_stats.get("expected_goals_conceded"),
                        raw_data=stat,
                        provider="Sportmonks",
                        data_source_tier=self.config.tier
                        if hasattr(self, "config")
                        else ProviderTier.BASIC,
                    )

                    xg_data.append(xg_point)

        except Exception as e:
            logger.error(
                "Error transforming Sportmonks bulk data",
                extra={"error": str(e), "response_data": response_data},
            )

        return xg_data

    def _extract_xg_stats_from_sportmonks(
        self, stat: dict[str, Any]
    ) -> dict[str, float]:
        """Extract xG/xA statistics from Sportmonks statistic object"""
        xg_stats = {}

        # Map Sportmonks statistic type IDs to our fields
        # These IDs may need to be updated based on Sportmonks documentation
        stat_type_mapping = {
            "expected_goals": ["expected_goals", "xg"],
            "expected_assists": ["expected_assists", "xa"],
            "expected_goal_involvements": ["expected_goal_involvements", "xgi"],
            "expected_goals_conceded": ["expected_goals_conceded", "xgc"],
        }

        statistic_data = stat.get("data", {})

        for our_field, possible_keys in stat_type_mapping.items():
            for key in possible_keys:
                if key in statistic_data:
                    try:
                        value = float(statistic_data[key])
                        xg_stats[our_field] = value
                        break
                    except (ValueError, TypeError):
                        continue

        return xg_stats

    def _parse_sportmonks_date(self, date_str: str | None) -> datetime:
        """Parse Sportmonks date string to datetime object"""
        if not date_str:
            return datetime.now()

        try:
            # Sportmonks typically uses ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse date: {date_str}")
            return datetime.now()


class APIHealthMonitor:
    """Monitor API health, track costs, and provide dashboards"""

    def __init__(self):
        self.providers: dict[str, XGDataProvider] = {}

    def register_provider(self, name: str, provider: XGDataProvider):
        """Register a provider for monitoring"""
        self.providers[name] = provider
        logger.info(f"Registered provider {name} for monitoring")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of all providers"""
        status = {"timestamp": datetime.now().isoformat(), "providers": {}}

        for name, provider in self.providers.items():
            metrics = provider.metrics
            circuit_breaker = provider.circuit_breaker

            provider_status = {
                "name": name,
                "circuit_breaker_state": circuit_breaker.state.value,
                "success_rate": metrics.success_rate,
                "total_requests": metrics.requests_made,
                "failed_requests": metrics.requests_failed,
                "total_cost": metrics.total_cost,
                "average_response_time": metrics.average_response_time,
                "rate_limit_hits": metrics.rate_limit_hits,
                "requests_remaining_in_window": metrics.requests_remaining_in_window,
                "last_request_time": metrics.last_request_time.isoformat()
                if metrics.last_request_time
                else None,
                "tier": provider.config.tier.value,
                "is_healthy": self._is_provider_healthy(provider),
            }

            status["providers"][name] = provider_status

        return status

    def _is_provider_healthy(self, provider: XGDataProvider) -> bool:
        """Determine if a provider is healthy"""
        metrics = provider.metrics
        circuit_breaker = provider.circuit_breaker

        # Provider is unhealthy if:
        # - Circuit breaker is open
        # - Success rate is below 80%
        # - No requests made and circuit breaker failed recently
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            return False

        return not (metrics.requests_made > 0 and metrics.success_rate < 0.8)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary across all providers"""
        total_cost = 0.0
        total_requests = 0
        provider_costs = {}

        for name, provider in self.providers.items():
            metrics = provider.metrics
            total_cost += metrics.total_cost
            total_requests += metrics.requests_made

            provider_costs[name] = {
                "total_cost": metrics.total_cost,
                "requests": metrics.requests_made,
                "cost_per_request": provider.config.cost_per_request,
                "estimated_monthly_cost": metrics.total_cost * 30
                if metrics.total_cost > 0
                else 0,
            }

        return {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "average_cost_per_request": total_cost / total_requests
            if total_requests > 0
            else 0,
            "provider_breakdown": provider_costs,
            "timestamp": datetime.now().isoformat(),
        }

    def export_dashboard_data(self) -> dict[str, Any]:
        """Export data for monitoring dashboard"""
        return {
            "health_status": self.get_health_status(),
            "cost_summary": self.get_cost_summary(),
            "recommendations": self._get_recommendations(),
        }

    def _get_recommendations(self) -> list[dict[str, str]]:
        """Generate recommendations based on current metrics"""
        recommendations = []

        for name, provider in self.providers.items():
            metrics = provider.metrics

            if metrics.success_rate < 0.9 and metrics.requests_made > 10:
                recommendations.append(
                    {
                        "provider": name,
                        "type": "reliability",
                        "message": f"Low success rate ({metrics.success_rate:.1%}). Consider investigating API issues.",
                    }
                )

            if metrics.rate_limit_hits > 5:
                recommendations.append(
                    {
                        "provider": name,
                        "type": "rate_limiting",
                        "message": f"Frequent rate limit hits ({metrics.rate_limit_hits}). Consider upgrading API plan.",
                    }
                )

            if metrics.average_response_time > 10:
                recommendations.append(
                    {
                        "provider": name,
                        "type": "performance",
                        "message": f"High response times ({metrics.average_response_time:.1f}s). Monitor API performance.",
                    }
                )

        return recommendations


class XGDataManager:
    """High-level manager for xG/xA data operations"""

    def __init__(self):
        self.providers: dict[str, XGDataProvider] = {}
        self.monitor = APIHealthMonitor()
        self.logger = get_logger(__name__)

    def add_provider(self, name: str, provider: XGDataProvider):
        """Add a data provider"""
        self.providers[name] = provider
        self.monitor.register_provider(name, provider)
        self.logger.info(f"Added xG data provider: {name}")

    def setup_sportmonks(self, api_key: str | None = None) -> bool:
        """Setup Sportmonks provider"""
        try:
            config = ProviderConfig(api_key=api_key) if api_key else None
            provider = SportmonksProvider(config)
            self.add_provider("sportmonks", provider)
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup Sportmonks provider: {e}")
            return False

    @timed
    async def sync_xg_data_for_gameweek(self, season: str, gameweek: int) -> int:
        """Sync xG/xA data for a specific gameweek"""
        if not self.providers:
            self.logger.warning("No xG data providers configured")
            return 0

        # Get fixtures for the gameweek
        with session_scope() as session:
            fixtures = (
                session.query(Fixture)
                .filter(Fixture.season == season, Fixture.gameweek == gameweek)
                .all()
            )

        if not fixtures:
            self.logger.warning(f"No fixtures found for {season} GW{gameweek}")
            return 0

        total_updated = 0

        for provider_name, provider in self.providers.items():
            try:
                for fixture in fixtures:
                    # Use async method if available
                    if hasattr(
                        provider, "get_match_xg_data"
                    ) and asyncio.iscoroutinefunction(provider.get_match_xg_data):
                        xg_data = await provider.get_match_xg_data(fixture.fixture_id)
                    else:
                        xg_data = provider.get_match_xg_data(fixture.fixture_id)

                    updated_count = self._update_player_scores(xg_data, session)
                    total_updated += updated_count

                    # No manual delay needed - API manager handles rate limiting

            except Exception as e:
                self.logger.error(
                    f"Error syncing xG data with {provider_name} for {season} GW{gameweek}",
                    extra={"error": str(e)},
                )

        self.logger.info(
            f"Updated xG data for {total_updated} player scores in {season} GW{gameweek}"
        )
        return total_updated

    def _update_player_scores(self, xg_data: list[XGDataPoint], session) -> int:
        """Update PlayerScore records with xG/xA data"""
        updated_count = 0

        for xg_point in xg_data:
            try:
                # Find matching PlayerScore record
                player_score = (
                    session.query(PlayerScore)
                    .filter(
                        PlayerScore.player_id == xg_point.player_id,
                        PlayerScore.fixture_id == xg_point.fixture_id,
                    )
                    .first()
                )

                if player_score:
                    # Update xG/xA fields
                    player_score.expected_goals = xg_point.expected_goals
                    player_score.expected_assists = xg_point.expected_assists
                    player_score.expected_goal_involvements = (
                        xg_point.expected_goal_involvements
                    )
                    player_score.expected_goals_conceded = (
                        xg_point.expected_goals_conceded
                    )

                    updated_count += 1

                else:
                    self.logger.debug(
                        f"No PlayerScore found for player {xg_point.player_id} "
                        f"in fixture {xg_point.fixture_id}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error updating PlayerScore for player {xg_point.player_id}",
                    extra={"error": str(e)},
                )

        session.commit()
        return updated_count

    def get_health_dashboard(self) -> dict[str, Any]:
        """Get monitoring dashboard data"""
        return self.monitor.export_dashboard_data()


# Convenience function for easy integration
def create_xg_manager(sportmonks_api_key: str | None = None) -> XGDataManager:
    """Create and configure xG data manager with Sportmonks provider"""
    manager = XGDataManager()

    if True:  # Allow setup even without key for graceful degradation
        success = manager.setup_sportmonks(sportmonks_api_key)
        if success:
            logger.info("xG Data Manager initialized with Sportmonks provider")
        else:
            logger.warning("xG Data Manager initialized without working providers")

    return manager


# Export main classes for easy imports
__all__ = [
    "APIHealthMonitor",
    "ProviderConfig",
    "ProviderTier",
    "SportmonksProvider",
    "XGDataManager",
    "XGDataPoint",
    "XGDataProvider",
    "create_xg_manager",
]
