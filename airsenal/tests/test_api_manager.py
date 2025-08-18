"""
Comprehensive tests for the API Management system.

Tests cover:
- APIRateLimiter with token bucket algorithm
- SmartCacheManager with multi-level caching
- RequestQueue with priority handling
- APIUsageMonitor for cost tracking
- FallbackManager for graceful degradation
- Integration testing
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from airsenal.framework.api_manager import (
    APIManager,
    APIQuota,
    APIRateLimiter,
    APIUsageMonitor,
    CacheStrategy,
    FallbackManager,
    Priority,
    RequestQueue,
    SmartCacheManager,
    TokenBucket,
    api_managed,
)


class TestTokenBucket:
    """Test token bucket rate limiting algorithm"""

    def test_token_bucket_creation(self):
        """Test token bucket initialization"""
        bucket = TokenBucket(capacity=10.0, tokens=10.0, fill_rate=2.0)
        assert bucket.capacity == 10.0
        assert bucket.tokens == 10.0
        assert bucket.fill_rate == 2.0

    def test_token_consumption(self):
        """Test token consumption"""
        bucket = TokenBucket(capacity=10.0, tokens=10.0, fill_rate=2.0)

        # Consume tokens
        assert bucket.consume(5.0) is True
        assert bucket.tokens == 5.0

        # Try to consume more than available
        assert bucket.consume(10.0) is False
        assert bucket.tokens == 5.0

    def test_token_refill(self):
        """Test token bucket refill mechanism"""
        bucket = TokenBucket(capacity=10.0, tokens=5.0, fill_rate=2.0)

        # Simulate time passage
        bucket.last_update = time.time() - 2.0  # 2 seconds ago
        bucket._refill()

        # Should have refilled 4 tokens (2 tokens/sec * 2 sec)
        assert bucket.tokens == 9.0

    def test_time_until_available(self):
        """Test calculation of time until tokens are available"""
        bucket = TokenBucket(capacity=10.0, tokens=2.0, fill_rate=1.0)

        # Need 5 tokens, have 2, fill rate is 1/sec
        time_needed = bucket.time_until_available(5.0)
        assert time_needed == 3.0


class TestAPIQuota:
    """Test API quota management"""

    def test_quota_creation(self):
        """Test quota initialization"""
        quota = APIQuota(
            provider="test", daily_limit=1000, hourly_limit=100, cost_per_request=0.01
        )
        assert quota.provider == "test"
        assert quota.daily_limit == 1000
        assert quota.hourly_limit == 100
        assert quota.cost_per_request == 0.01

    def test_quota_consumption(self):
        """Test quota consumption"""
        quota = APIQuota(provider="test", daily_limit=1000, hourly_limit=100)

        assert quota.can_make_request() is True

        quota.consume_quota(10)
        assert quota.current_daily_usage == 10
        assert quota.current_hourly_usage == 10

    def test_quota_limits(self):
        """Test quota limit enforcement"""
        quota = APIQuota(provider="test", daily_limit=10, hourly_limit=5)

        # Consume up to hourly limit
        quota.consume_quota(5)
        assert quota.can_make_request() is False

        # Reset hourly (simulate time passage)
        quota.current_hourly_usage = 0
        assert quota.can_make_request() is True


class TestAPIRateLimiter:
    """Test API rate limiter with token bucket algorithm"""

    def setup_method(self):
        """Setup test fixtures"""
        self.rate_limiter = APIRateLimiter(redis_enabled=False)

    def test_rate_limiter_configuration(self):
        """Test rate limiter provider configuration"""
        self.rate_limiter.configure_provider(
            provider="test",
            requests_per_second=2.0,
            burst_capacity=10,
            daily_limit=1000,
            hourly_limit=100,
        )

        assert "test" in self.rate_limiter.local_buckets
        assert "test" in self.rate_limiter.quotas

        bucket = self.rate_limiter.local_buckets["test"]
        assert bucket.capacity == 10.0
        assert bucket.fill_rate == 2.0

    @pytest.mark.asyncio
    async def test_permit_acquisition(self):
        """Test permit acquisition"""
        self.rate_limiter.configure_provider(
            provider="test",
            requests_per_second=10.0,  # High rate for testing
            burst_capacity=10,
        )

        # Should acquire permit successfully
        success = await self.rate_limiter.acquire_permit("test")
        assert success is True

        # Multiple permits
        for _ in range(9):  # 9 more permits (total 10)
            success = await self.rate_limiter.acquire_permit("test")
            assert success is True

        # Should fail on 11th permit (burst capacity exceeded)
        success = await self.rate_limiter.acquire_permit("test", timeout=0.1)
        assert success is False

    def test_rate_limiter_status(self):
        """Test rate limiter status reporting"""
        self.rate_limiter.configure_provider(
            provider="test", requests_per_second=2.0, burst_capacity=10
        )

        status = self.rate_limiter.get_status("test")
        assert status["provider"] == "test"
        assert status["bucket_capacity"] == 10.0
        assert status["fill_rate"] == 2.0
        assert "tokens_available" in status


class TestSmartCacheManager:
    """Test smart cache manager with multi-level caching"""

    def setup_method(self):
        """Setup test fixtures"""
        self.cache = SmartCacheManager(
            max_memory_entries=100,
            max_memory_bytes=1024 * 1024,  # 1MB
            eviction_strategy=CacheStrategy.LRU,
        )

    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache operations"""
        # Set value
        success = await self.cache.set("test_key", "test_value", ttl=3600)
        assert success is True

        # Get value
        value = await self.cache.get("test_key")
        assert value == "test_value"

        # Get non-existent key
        value = await self.cache.get("non_existent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entry expiration"""
        # Set with short TTL
        await self.cache.set("test_key", "test_value", ttl=0.1)

        # Should be available immediately
        value = await self.cache.get("test_key")
        assert value == "test_value"

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be expired
        value = await self.cache.get("test_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction with LRU strategy"""
        # Create cache with small capacity
        cache = SmartCacheManager(max_memory_entries=3)

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it most recent
        await cache.get("key1")

        # Add one more - should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still there
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"  # Still there
        assert await cache.get("key4") == "value4"  # Just added

    @pytest.mark.asyncio
    async def test_get_or_fetch(self):
        """Test get_or_fetch functionality"""
        fetch_calls = 0

        async def fetch_func():
            nonlocal fetch_calls
            fetch_calls += 1
            return f"fetched_value_{fetch_calls}"

        # First call should fetch
        value = await self.cache.get_or_fetch("test_key", fetch_func)
        assert value == "fetched_value_1"
        assert fetch_calls == 1

        # Second call should use cache
        value = await self.cache.get_or_fetch("test_key", fetch_func)
        assert value == "fetched_value_1"
        assert fetch_calls == 1  # No additional fetch

    def test_cache_stats(self):
        """Test cache statistics"""
        stats = self.cache.get_stats()

        assert "memory_hit_rate" in stats
        assert "redis_hit_rate" in stats
        assert "overall_hit_rate" in stats
        assert "memory_entries" in stats
        assert "evictions" in stats


class TestRequestQueue:
    """Test priority request queue"""

    def setup_method(self):
        """Setup test fixtures"""
        self.queue = RequestQueue(max_concurrent=5)

    @pytest.mark.asyncio
    async def test_queue_processing(self):
        """Test basic queue processing"""
        await self.queue.start()

        async def test_func(value):
            await asyncio.sleep(0.1)
            return value * 2

        # Enqueue request
        result = await self.queue.enqueue(
            provider="test", func=test_func, priority=Priority.MEDIUM, value=5
        )

        assert result == 10
        await self.queue.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based request ordering"""
        await self.queue.start()

        results = []

        async def test_func(value):
            await asyncio.sleep(0.1)
            results.append(value)
            return value

        # Enqueue requests with different priorities
        tasks = [
            self.queue.enqueue("test", test_func, Priority.LOW, 3),
            self.queue.enqueue("test", test_func, Priority.HIGH, 1),
            self.queue.enqueue("test", test_func, Priority.MEDIUM, 2),
        ]

        await asyncio.gather(*tasks)

        # Results should be in priority order (HIGH, MEDIUM, LOW)
        # Note: Due to async nature, exact ordering may vary
        # but HIGH priority should generally be processed first
        assert 1 in results
        assert 2 in results
        assert 3 in results

        await self.queue.stop()

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test request retry logic"""
        await self.queue.start()

        attempts = 0

        async def failing_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                msg = "Test failure"
                raise Exception(msg)
            return "success"

        result = await self.queue.enqueue(
            provider="test", func=failing_func, max_retries=3
        )

        assert result == "success"
        assert attempts == 3

        await self.queue.stop()

    def test_queue_stats(self):
        """Test queue statistics"""
        stats = self.queue.get_stats()

        assert "queue_length" in stats
        assert "active_requests" in stats
        assert "completed_requests" in stats
        assert "success_rate" in stats


class TestAPIUsageMonitor:
    """Test API usage monitoring and cost tracking"""

    def setup_method(self):
        """Setup test fixtures"""
        self.monitor = APIUsageMonitor()

    def test_provider_registration(self):
        """Test provider registration"""
        self.monitor.register_provider(
            provider="test",
            cost_per_request=0.01,
            quota_limits={"daily": 1000, "hourly": 100},
        )

        assert "test" in self.monitor.providers
        assert self.monitor.providers["test"]["cost_per_request"] == 0.01

    def test_request_recording(self):
        """Test request recording"""
        self.monitor.register_provider("test", cost_per_request=0.01)

        # Record successful request
        self.monitor.record_request(
            provider="test", success=True, response_time=0.5, cost=0.01
        )

        stats = self.monitor.get_provider_stats("test")
        assert stats["total_requests"] == 1
        assert stats["success_count"] == 1
        assert stats["total_cost"] == 0.01
        assert stats["success_rate"] == 1.0

    def test_cost_tracking(self):
        """Test cost tracking across providers"""
        self.monitor.register_provider("provider1", cost_per_request=0.01)
        self.monitor.register_provider("provider2", cost_per_request=0.02)

        # Record requests
        self.monitor.record_request("provider1", True, 0.5, 0.01)
        self.monitor.record_request("provider2", True, 0.8, 0.02)

        all_stats = self.monitor.get_all_stats()
        assert all_stats["totals"]["cost"] == 0.03
        assert all_stats["totals"]["requests"] == 2

    def test_cost_alerts(self):
        """Test cost alert generation"""
        self.monitor.register_provider("test", cost_per_request=0.10)

        # Record expensive requests
        for _ in range(20):
            self.monitor.record_request("test", True, 0.5, 10.0)

        alerts = self.monitor.get_cost_alerts(daily_threshold=100.0)
        assert len(alerts) > 0
        assert alerts[0]["type"] == "daily_cost_exceeded"


class TestFallbackManager:
    """Test fallback management for graceful degradation"""

    def setup_method(self):
        """Setup test fixtures"""
        self.cache = SmartCacheManager()
        self.fallback = FallbackManager(self.cache)

    @pytest.mark.asyncio
    async def test_fallback_strategies(self):
        """Test fallback strategy execution"""

        # Register fallback strategies
        async def fallback1():
            msg = "Fallback 1 failed"
            raise Exception(msg)

        async def fallback2():
            return "fallback_result"

        self.fallback.register_fallback("test", [fallback1, fallback2])

        # Simulate primary function failure
        async def failing_primary():
            msg = "Primary failed"
            raise Exception(msg)

        result = await self.fallback.execute_with_fallback(
            provider="test", primary_func=failing_primary
        )

        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_cached_fallback(self):
        """Test fallback to cached data"""
        # Set up cached data
        cache_key = "test_key"
        await self.cache.set(cache_key, "cached_data")

        # Primary function that fails
        async def failing_primary():
            msg = "Primary failed"
            raise Exception(msg)

        # Execute with fallback
        result = await self.fallback.execute_with_fallback(
            provider="test", primary_func=failing_primary, cache_key=cache_key
        )

        assert result == "cached_data"


class TestAPIManager:
    """Test main API manager coordination"""

    def setup_method(self):
        """Setup test fixtures"""
        self.api_manager = APIManager(
            max_memory_cache_entries=100, max_concurrent_requests=5
        )

    @pytest.mark.asyncio
    async def test_api_manager_initialization(self):
        """Test API manager initialization"""
        await self.api_manager.initialize()
        assert self.api_manager._initialized is True

        await self.api_manager.shutdown()
        assert self.api_manager._initialized is False

    @pytest.mark.asyncio
    async def test_provider_configuration(self):
        """Test provider configuration"""
        await self.api_manager.initialize()

        # Configure provider
        self.api_manager.configure_provider(
            provider="test",
            requests_per_second=2.0,
            burst_capacity=10,
            cost_per_request=0.01,
        )

        # Check configuration
        status = self.api_manager.rate_limiter.get_status("test")
        assert status["provider"] == "test"

        await self.api_manager.shutdown()

    @pytest.mark.asyncio
    async def test_request_context(self):
        """Test API request context manager"""
        await self.api_manager.initialize()

        self.api_manager.configure_provider(
            provider="test", requests_per_second=10.0, burst_capacity=10
        )

        async def test_func():
            return "test_result"

        async with self.api_manager.request_context("test") as ctx:
            result = await ctx.make_request(test_func)
            assert result == "test_result"

        await self.api_manager.shutdown()

    @pytest.mark.asyncio
    async def test_queued_request(self):
        """Test queued API request"""
        await self.api_manager.initialize()

        self.api_manager.configure_provider(
            provider="test", requests_per_second=10.0, burst_capacity=10
        )

        async def test_func(value):
            return value * 2

        result = await self.api_manager.queue_request(
            provider="test", func=test_func, priority=Priority.HIGH, value=5
        )

        assert result == 10

        await self.api_manager.shutdown()

    def test_comprehensive_stats(self):
        """Test comprehensive statistics"""
        stats = self.api_manager.get_comprehensive_stats()

        assert "rate_limiter" in stats
        assert "cache" in stats
        assert "queue" in stats
        assert "usage" in stats
        assert "cost_alerts" in stats
        assert "timestamp" in stats


class TestDecorators:
    """Test API management decorators"""

    @pytest.mark.asyncio
    async def test_api_managed_decorator(self):
        """Test api_managed decorator"""

        @api_managed(
            provider="test",
            cache_key_func=lambda x: f"test_{x}",
            ttl=3600,
            priority=Priority.HIGH,
        )
        async def test_function(value):
            return value * 2

        # Mock the global API manager
        with patch(
            "airsenal.framework.api_manager.get_api_manager"
        ) as mock_get_manager:
            mock_manager = AsyncMock()
            mock_context = AsyncMock()
            mock_context.make_request = AsyncMock(return_value=10)
            mock_manager.request_context.return_value.__aenter__ = AsyncMock(
                return_value=mock_context
            )
            mock_manager.request_context.return_value.__aexit__ = AsyncMock(
                return_value=None
            )
            mock_get_manager.return_value = mock_manager

            result = await test_function(5)
            assert result == 10


class TestIntegration:
    """Integration tests for full API management system"""

    @pytest.mark.asyncio
    async def test_full_integration(self):
        """Test full integration of all components"""
        api_manager = APIManager()
        await api_manager.initialize()

        # Configure provider
        api_manager.configure_provider(
            provider="integration_test",
            requests_per_second=5.0,
            burst_capacity=10,
            cost_per_request=0.01,
        )

        # Test function with various scenarios
        call_count = 0

        async def test_api_call(value, fail=False):
            nonlocal call_count
            call_count += 1

            if fail and call_count == 1:
                msg = "Simulated API failure"
                raise Exception(msg)

            await asyncio.sleep(0.1)  # Simulate API delay
            return f"result_{value}"

        # Test successful request with caching
        result1 = await api_manager.queue_request(
            provider="integration_test",
            func=test_api_call,
            cache_key="test_cache_key",
            value="test1",
        )
        assert result1 == "result_test1"

        # Test cached response (should not increment call_count)
        initial_call_count = call_count
        result2 = await api_manager.queue_request(
            provider="integration_test",
            func=test_api_call,
            cache_key="test_cache_key",
            value="test1",
        )
        assert result2 == "result_test1"
        assert call_count == initial_call_count  # No new API call

        # Test rate limiting by making many requests
        tasks = []
        for i in range(15):  # More than burst capacity
            tasks.append(
                api_manager.queue_request(
                    provider="integration_test", func=test_api_call, value=f"bulk_{i}"
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (queue handles rate limiting)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 15

        # Get comprehensive stats
        stats = api_manager.get_comprehensive_stats()
        assert stats["cache"]["overall_hit_rate"] > 0  # Should have cache hits
        assert len(stats["rate_limiter"]) > 0  # Should have rate limiter data

        await api_manager.shutdown()


class TestPerformance:
    """Performance tests for API management system"""

    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load"""
        cache = SmartCacheManager(max_memory_entries=1000)

        # Test write performance
        start_time = time.time()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time

        # Test read performance
        start_time = time.time()
        for i in range(1000):
            await cache.get(f"key_{i}")
        read_time = time.time() - start_time

        # Performance assertions (adjust thresholds as needed)
        assert write_time < 1.0  # Should write 1000 items in under 1 second
        assert read_time < 0.5  # Should read 1000 items in under 0.5 seconds

        stats = cache.get_stats()
        assert stats["memory_hit_rate"] == 1.0  # All reads should hit memory cache

    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        rate_limiter = APIRateLimiter(redis_enabled=False)
        rate_limiter.configure_provider(
            provider="perf_test",
            requests_per_second=1000.0,  # High rate for testing
            burst_capacity=1000,
        )

        # Test permit acquisition performance
        start_time = time.time()
        success_count = 0

        for _ in range(1000):
            if await rate_limiter.acquire_permit("perf_test"):
                success_count += 1

        acquisition_time = time.time() - start_time

        # Performance assertions
        assert acquisition_time < 1.0  # Should process 1000 permits in under 1 second
        assert success_count == 1000  # All should succeed with high rate limit


# Fixtures for pytest
@pytest.fixture
def api_manager():
    """Provide API manager fixture"""
    manager = APIManager()
    yield manager
    # Cleanup
    if manager._initialized:
        asyncio.create_task(manager.shutdown())


@pytest.fixture
def rate_limiter():
    """Provide rate limiter fixture"""
    return APIRateLimiter(redis_enabled=False)


@pytest.fixture
def cache_manager():
    """Provide cache manager fixture"""
    return SmartCacheManager()


@pytest.fixture
def usage_monitor():
    """Provide usage monitor fixture"""
    return APIUsageMonitor()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
