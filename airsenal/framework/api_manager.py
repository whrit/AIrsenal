"""
API Rate Limiting and Caching Management for AIrsenal

This module provides sophisticated API management capabilities including:
- Token bucket rate limiting with burst capacity
- Multi-level caching (memory -> Redis -> fallback)
- Priority request queue with intelligent scheduling
- Circuit breaker integration
- Comprehensive cost tracking and monitoring
- Graceful fallback mechanisms

Features:
- Token bucket algorithm for smooth rate limiting
- Distributed rate limiting across multiple instances
- Smart cache eviction with LRU and priority-based policies
- Request queuing with priority handling
- API cost tracking per provider
- Circuit breaker patterns for fault tolerance
- Cache warming strategies
- Monitoring dashboard integration

Usage:
    # Initialize API manager
    api_manager = APIManager()

    # Configure rate limiting
    api_manager.configure_rate_limiter(
        provider="sportmonks",
        requests_per_second=2.0,
        burst_capacity=10
    )

    # Make rate-limited API request
    async with api_manager.request_context("sportmonks", priority="high") as ctx:
        response = await ctx.make_request("https://api.example.com/data")

    # Use smart caching
    cache_manager = api_manager.get_cache_manager()
    cached_data = await cache_manager.get_or_fetch(
        key="player_stats_123",
        fetch_func=fetch_player_stats,
        ttl=3600,
        priority="medium"
    )
"""

import asyncio
import hashlib
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from heapq import heappop, heappush
from typing import (
    Any,
)

import numpy as np

from airsenal.framework.env import (
    AIRSENAL_REDIS_ENABLE,
)
from airsenal.framework.logging_config import get_logger
from airsenal.framework.redis_cache import redis_cache

logger = get_logger(__name__)


class Priority(Enum):
    """Request priority levels"""

    CRITICAL = 1  # User-facing requests
    HIGH = 2  # Time-sensitive operations
    MEDIUM = 3  # Normal operations
    LOW = 4  # Background jobs
    BULK = 5  # Batch operations


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CacheStrategy(Enum):
    """Cache eviction strategies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    PRIORITY = "priority"  # Priority-based
    TTL = "ttl"  # Time To Live based


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""

    capacity: float
    tokens: float
    fill_rate: float  # tokens per second
    last_update: float = field(default_factory=time.time)

    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens from bucket"""
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now

    def time_until_available(self, tokens: float = 1.0) -> float:
        """Time in seconds until tokens are available"""
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed_tokens = tokens - self.tokens
        return needed_tokens / self.fill_rate


@dataclass
class APIQuota:
    """API quota management"""

    provider: str
    daily_limit: int
    hourly_limit: int
    current_daily_usage: int = 0
    current_hourly_usage: int = 0
    daily_reset_time: datetime = field(
        default_factory=lambda: datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        + timedelta(days=1)
    )
    hourly_reset_time: datetime = field(
        default_factory=lambda: datetime.now().replace(
            minute=0, second=0, microsecond=0
        )
        + timedelta(hours=1)
    )
    cost_per_request: float = 0.0
    total_cost: float = 0.0

    def can_make_request(self) -> bool:
        """Check if request can be made within quota"""
        self._reset_if_needed()
        return (
            self.current_daily_usage < self.daily_limit
            and self.current_hourly_usage < self.hourly_limit
        )

    def consume_quota(self, count: int = 1):
        """Consume quota for API requests"""
        self._reset_if_needed()
        self.current_daily_usage += count
        self.current_hourly_usage += count
        self.total_cost += self.cost_per_request * count

    def _reset_if_needed(self):
        """Reset counters if time windows have passed"""
        now = datetime.now()

        if now >= self.daily_reset_time:
            self.current_daily_usage = 0
            self.daily_reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

        if now >= self.hourly_reset_time:
            self.current_hourly_usage = 0
            self.hourly_reset_time = now.replace(
                minute=0, second=0, microsecond=0
            ) + timedelta(hours=1)


@dataclass
class QueuedRequest:
    """Queued API request with priority and metadata"""

    id: str
    provider: str
    priority: Priority
    created_at: float
    future: asyncio.Future
    func: Callable
    args: tuple
    kwargs: dict
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3

    def __lt__(self, other):
        """Compare requests for priority queue (lower priority value = higher precedence)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    ttl: float
    created_at: float
    last_accessed: float
    access_count: int = 0
    priority: Priority = Priority.MEDIUM
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.created_at + self.ttl)

    @property
    def age(self) -> float:
        """Age of cache entry in seconds"""
        return time.time() - self.created_at

    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class APIRateLimiter:
    """Token bucket rate limiter with distributed support"""

    def __init__(self, redis_enabled: bool = True):
        self.redis_enabled = redis_enabled and AIRSENAL_REDIS_ENABLE
        self.local_buckets: dict[str, TokenBucket] = {}
        self.quotas: dict[str, APIQuota] = {}
        self._lock = threading.Lock()

        logger.info(
            "APIRateLimiter initialized with Redis %s",
            "enabled" if self.redis_enabled else "disabled",
        )

    def configure_provider(
        self,
        provider: str,
        requests_per_second: float,
        burst_capacity: int,
        daily_limit: int = 10000,
        hourly_limit: int = 1000,
        cost_per_request: float = 0.0,
    ):
        """Configure rate limiting for a provider"""
        with self._lock:
            # Create token bucket
            self.local_buckets[provider] = TokenBucket(
                capacity=float(burst_capacity),
                tokens=float(burst_capacity),
                fill_rate=requests_per_second,
            )

            # Create quota manager
            self.quotas[provider] = APIQuota(
                provider=provider,
                daily_limit=daily_limit,
                hourly_limit=hourly_limit,
                cost_per_request=cost_per_request,
            )

        logger.info(
            "Configured rate limiter for %s: %s req/s, burst %s, daily %s, hourly %s",
            provider,
            requests_per_second,
            burst_capacity,
            daily_limit,
            hourly_limit,
        )

    async def acquire_permit(
        self, provider: str, tokens: float = 1.0, timeout: float = 30.0
    ) -> bool:
        """Acquire permit to make API request"""
        if provider not in self.local_buckets:
            logger.warning("No rate limiter configured for provider %s", provider)
            return True

        start_time = time.time()

        while True:
            # Check quota first
            quota = self.quotas.get(provider)
            if quota and not quota.can_make_request():
                logger.warning("Quota exceeded for provider %s", provider)
                return False

            # Try to acquire tokens
            if self.redis_enabled:
                success = await self._acquire_distributed_permit(provider, tokens)
            else:
                success = self._acquire_local_permit(provider, tokens)

            if success:
                # Consume quota
                if quota:
                    quota.consume_quota()
                return True

            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning("Rate limiter timeout for provider %s", provider)
                return False

            # Wait before retry
            bucket = self.local_buckets[provider]
            wait_time = min(bucket.time_until_available(tokens), 1.0)
            await asyncio.sleep(wait_time)

    def _acquire_local_permit(self, provider: str, tokens: float) -> bool:
        """Acquire permit using local token bucket"""
        with self._lock:
            bucket = self.local_buckets[provider]
            return bucket.consume(tokens)

    async def _acquire_distributed_permit(self, provider: str, tokens: float) -> bool:
        """Acquire permit using distributed Redis-based rate limiting"""
        if not redis_cache.is_available():
            # Fallback to local rate limiting
            return self._acquire_local_permit(provider, tokens)

        try:
            # Redis Lua script for atomic token bucket operation
            lua_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local fill_rate = tonumber(ARGV[2])
            local tokens_requested = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])

            local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
            local tokens = tonumber(bucket[1]) or capacity
            local last_update = tonumber(bucket[2]) or now

            -- Refill bucket
            local elapsed = now - last_update
            local new_tokens = math.min(capacity, tokens + (elapsed * fill_rate))

            -- Try to consume tokens
            if new_tokens >= tokens_requested then
                new_tokens = new_tokens - tokens_requested
                redis.call('HMSET', key, 'tokens', new_tokens, 'last_update', now)
                redis.call('EXPIRE', key, 3600)  -- Expire in 1 hour
                return 1
            else
                redis.call('HMSET', key, 'tokens', new_tokens, 'last_update', now)
                redis.call('EXPIRE', key, 3600)
                return 0
            end
            """

            bucket = self.local_buckets[provider]
            redis_key = f"rate_limit:{provider}"

            with redis_cache.connection_manager.get_connection() as conn:
                result = conn.eval(
                    lua_script,
                    1,
                    redis_key,
                    str(bucket.capacity),
                    str(bucket.fill_rate),
                    str(tokens),
                    str(time.time()),
                )
                return bool(result)

        except Exception as e:
            logger.error("Distributed rate limiting failed for %s: %s", provider, e)
            # Fallback to local rate limiting
            return self._acquire_local_permit(provider, tokens)

    def get_status(self, provider: str) -> dict[str, Any]:
        """Get rate limiter status for provider"""
        if provider not in self.local_buckets:
            return {"error": f"Provider {provider} not configured"}

        bucket = self.local_buckets[provider]
        quota = self.quotas.get(provider)

        bucket._refill()  # Update tokens

        status = {
            "provider": provider,
            "tokens_available": bucket.tokens,
            "bucket_capacity": bucket.capacity,
            "fill_rate": bucket.fill_rate,
            "next_token_in": bucket.time_until_available(1.0),
        }

        if quota:
            status.update(
                {
                    "daily_usage": quota.current_daily_usage,
                    "daily_limit": quota.daily_limit,
                    "hourly_usage": quota.current_hourly_usage,
                    "hourly_limit": quota.hourly_limit,
                    "total_cost": quota.total_cost,
                    "can_make_request": quota.can_make_request(),
                }
            )

        return status

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all configured providers"""
        return {provider: self.get_status(provider) for provider in self.local_buckets}


class SmartCacheManager:
    """Multi-level cache manager with intelligent eviction"""

    def __init__(
        self,
        max_memory_entries: int = 1000,
        max_memory_bytes: int = 100 * 1024 * 1024,  # 100MB
        eviction_strategy: CacheStrategy = CacheStrategy.LRU,
    ):
        self.max_memory_entries = max_memory_entries
        self.max_memory_bytes = max_memory_bytes
        self.eviction_strategy = eviction_strategy

        # In-memory cache
        self.memory_cache: dict[str, CacheEntry] = {}
        self.memory_size_bytes = 0
        self._memory_lock = threading.Lock()

        # LRU tracking
        self.access_order = deque()
        self.access_set: set[str] = set()

        # Statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "evictions": 0,
            "total_requests": 0,
        }

        logger.info(
            "SmartCacheManager initialized with %s entries, %s bytes",
            max_memory_entries,
            max_memory_bytes,
        )

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level lookup"""
        self.stats["total_requests"] += 1

        # Check memory cache first
        with self._memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired:
                    entry.touch()
                    self._update_access_order(key)
                    self.stats["memory_hits"] += 1
                    return entry.value
                # Remove expired entry
                self._remove_from_memory(key)

        self.stats["memory_misses"] += 1

        # Check Redis cache using existing infrastructure
        if redis_cache.is_available():
            try:
                redis_value = redis_cache.get(key)
                if redis_value is not None:
                    self.stats["redis_hits"] += 1

                    # Promote to memory cache with smart TTL calculation
                    # Use shorter TTL for promoted entries to ensure freshness
                    promote_ttl = min(ttl if "ttl" in locals() else 3600, 1800)
                    await self._set_memory_cache(
                        key, redis_value, ttl=promote_ttl, priority=Priority.MEDIUM
                    )

                    return redis_value
            except Exception as e:
                logger.error("Redis cache error: %s", e)

        self.stats["redis_misses"] += 1
        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float = 3600,
        priority: Priority = Priority.MEDIUM,
        memory_only: bool = False,
    ) -> bool:
        """Set value in cache with configurable levels"""
        success = True

        # Set in memory cache
        await self._set_memory_cache(key, value, ttl, priority)

        # Set in Redis cache (unless memory_only) using existing infrastructure
        if not memory_only and redis_cache.is_available():
            try:
                # Use the existing Redis cache with enhanced key management
                success = redis_cache.set(key, value, int(ttl)) and success
            except Exception as e:
                logger.error("Redis cache set error: %s", e)
                success = False

        return success

    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable[..., Awaitable[Any]],
        ttl: float = 3600,
        priority: Priority = Priority.MEDIUM,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """Get from cache or fetch and cache the result"""
        kwargs = kwargs or {}

        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Fetch and cache
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func(*args, **kwargs)
            else:
                value = fetch_func(*args, **kwargs)

            if value is not None:
                await self.set(key, value, ttl, priority)

            return value

        except Exception as e:
            logger.error("Fetch function failed for key %s: %s", key, e)
            raise

    async def _set_memory_cache(
        self, key: str, value: Any, ttl: float, priority: Priority
    ):
        """Set value in memory cache with eviction"""
        # Calculate size
        try:
            size_bytes = len(str(value).encode("utf-8"))
        except:
            size_bytes = 1024  # Default estimate

        with self._memory_lock:
            # Remove existing entry if present
            if key in self.memory_cache:
                self._remove_from_memory(key)

            # Check if we need to evict
            while (
                len(self.memory_cache) >= self.max_memory_entries
                or self.memory_size_bytes + size_bytes > self.max_memory_bytes
            ):
                if not self._evict_one():
                    break  # No more entries to evict

            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                created_at=time.time(),
                last_accessed=time.time(),
                priority=priority,
                size_bytes=size_bytes,
            )

            self.memory_cache[key] = entry
            self.memory_size_bytes += size_bytes
            self._update_access_order(key)

    def _evict_one(self) -> bool:
        """Evict one entry based on strategy"""
        if not self.memory_cache:
            return False

        if self.eviction_strategy == CacheStrategy.LRU:
            key_to_evict = self._find_lru_key()
        elif self.eviction_strategy == CacheStrategy.LFU:
            key_to_evict = self._find_lfu_key()
        elif self.eviction_strategy == CacheStrategy.PRIORITY:
            key_to_evict = self._find_lowest_priority_key()
        else:  # TTL
            key_to_evict = self._find_shortest_ttl_key()

        if key_to_evict:
            self._remove_from_memory(key_to_evict)
            self.stats["evictions"] += 1
            return True

        return False

    def _find_lru_key(self) -> str | None:
        """Find least recently used key"""
        while self.access_order:
            key = self.access_order.popleft()
            if key in self.memory_cache:
                self.access_order.appendleft(key)  # Put it back
                return key
        return None

    def _find_lfu_key(self) -> str | None:
        """Find least frequently used key"""
        if not self.memory_cache:
            return None

        return min(
            self.memory_cache.keys(), key=lambda k: self.memory_cache[k].access_count
        )

    def _find_lowest_priority_key(self) -> str | None:
        """Find lowest priority key"""
        if not self.memory_cache:
            return None

        return max(
            self.memory_cache.keys(), key=lambda k: self.memory_cache[k].priority.value
        )

    def _find_shortest_ttl_key(self) -> str | None:
        """Find key with shortest remaining TTL"""
        if not self.memory_cache:
            return None

        current_time = time.time()
        return min(
            self.memory_cache.keys(),
            key=lambda k: (self.memory_cache[k].created_at + self.memory_cache[k].ttl)
            - current_time,
        )

    def _remove_from_memory(self, key: str):
        """Remove entry from memory cache"""
        if key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.memory_size_bytes -= entry.size_bytes
            self.access_set.discard(key)

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self.access_set:
            # Remove from current position
            temp_deque = deque()
            while self.access_order:
                item = self.access_order.popleft()
                if item != key:
                    temp_deque.append(item)
            self.access_order = temp_deque

        # Add to end (most recent)
        self.access_order.append(key)
        self.access_set.add(key)

    def invalidate(self, pattern: str | None = None, key: str | None = None) -> int:
        """Invalidate cache entries"""
        count = 0

        if key:
            # Invalidate specific key
            with self._memory_lock:
                if key in self.memory_cache:
                    self._remove_from_memory(key)
                    count += 1

            # Use existing Redis cache infrastructure
            if redis_cache.is_available() and redis_cache.delete(key):
                count += 1

        elif pattern:
            # Invalidate by pattern in memory cache
            with self._memory_lock:
                keys_to_remove = [k for k in self.memory_cache if pattern in k]
                for k in keys_to_remove:
                    self._remove_from_memory(k)
                    count += 1

            # Use existing Redis cache pattern deletion
            if redis_cache.is_available():
                count += redis_cache.delete_pattern(pattern)

        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        memory_hit_rate = 0.0
        redis_hit_rate = 0.0
        overall_hit_rate = 0.0

        total_memory_requests = self.stats["memory_hits"] + self.stats["memory_misses"]
        total_redis_requests = self.stats["redis_hits"] + self.stats["redis_misses"]

        if total_memory_requests > 0:
            memory_hit_rate = self.stats["memory_hits"] / total_memory_requests

        if total_redis_requests > 0:
            redis_hit_rate = self.stats["redis_hits"] / total_redis_requests

        if self.stats["total_requests"] > 0:
            overall_hit_rate = (
                self.stats["memory_hits"] + self.stats["redis_hits"]
            ) / self.stats["total_requests"]

        return {
            "memory_hit_rate": memory_hit_rate,
            "redis_hit_rate": redis_hit_rate,
            "overall_hit_rate": overall_hit_rate,
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": self.memory_size_bytes / (1024 * 1024),
            "evictions": self.stats["evictions"],
            "total_requests": self.stats["total_requests"],
            **self.stats,
        }


class RequestQueue:
    """Priority queue for API requests with intelligent scheduling"""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.queue: list[QueuedRequest] = []
        self.active_requests: dict[str, QueuedRequest] = {}
        self.completed_requests: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._worker_task: asyncio.Task | None = None
        self._shutdown = False

        logger.info(
            "RequestQueue initialized with %s concurrent requests", max_concurrent
        )

    async def start(self):
        """Start the request processing worker"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("RequestQueue worker started")

    async def stop(self):
        """Stop the request processing worker"""
        self._shutdown = True
        if self._worker_task:
            await self._worker_task
        logger.info("RequestQueue worker stopped")

    async def enqueue(
        self,
        provider: str,
        func: Callable,
        priority: Priority = Priority.MEDIUM,
        timeout: float = 30.0,
        max_retries: int = 3,
        *args,
        **kwargs,
    ) -> Any:
        """Enqueue an API request"""
        request_id = hashlib.md5(
            f"{provider}:{func.__name__}:{time.time()}".encode()
        ).hexdigest()

        future = asyncio.Future()

        request = QueuedRequest(
            id=request_id,
            provider=provider,
            priority=priority,
            created_at=time.time(),
            future=future,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )

        async with self._lock:
            heappush(self.queue, request)

        logger.debug("Enqueued request %s with priority %s", request_id, priority.name)

        # Ensure worker is running
        await self.start()

        # Wait for result
        try:
            return await asyncio.wait_for(future, timeout=timeout + 5.0)
        except asyncio.TimeoutError:
            logger.error("Request %s timed out", request_id)
            future.cancel()
            raise

    async def _worker(self):
        """Process requests from the queue"""
        logger.info("RequestQueue worker started")

        while not self._shutdown:
            try:
                # Get next request
                request = await self._get_next_request()
                if not request:
                    await asyncio.sleep(0.1)
                    continue

                # Process request with semaphore
                async with self._semaphore:
                    await self._process_request(request)

            except Exception as e:
                logger.error("RequestQueue worker error: %s", e)
                await asyncio.sleep(1.0)

    async def _get_next_request(self) -> QueuedRequest | None:
        """Get the next request from priority queue"""
        async with self._lock:
            if self.queue:
                return heappop(self.queue)
        return None

    async def _process_request(self, request: QueuedRequest):
        """Process a single request"""
        self.active_requests[request.id] = request

        try:
            start_time = time.time()

            # Execute the function
            if asyncio.iscoroutinefunction(request.func):
                result = await request.func(*request.args, **request.kwargs)
            else:
                result = request.func(*request.args, **request.kwargs)

            # Set result
            if not request.future.done():
                request.future.set_result(result)

            # Record completion
            completion_time = time.time() - start_time
            self._record_completion(request, completion_time, True)

            logger.debug("Completed request %s in %.2fs", request.id, completion_time)

        except Exception as e:
            # Handle retry logic
            if request.retry_count < request.max_retries:
                request.retry_count += 1

                # Exponential backoff
                delay = min(2**request.retry_count, 30)
                await asyncio.sleep(delay)

                # Re-enqueue
                async with self._lock:
                    heappush(self.queue, request)

                logger.warning(
                    "Retrying request %s (attempt %s)", request.id, request.retry_count
                )
            else:
                # Set exception
                if not request.future.done():
                    request.future.set_exception(e)

                # Record completion
                completion_time = time.time() - request.created_at
                self._record_completion(request, completion_time, False)

                logger.error(
                    "Request %s failed after %s retries: %s",
                    request.id,
                    request.max_retries,
                    e,
                )

        finally:
            self.active_requests.pop(request.id, None)

    def _record_completion(
        self, request: QueuedRequest, duration: float, success: bool
    ):
        """Record request completion for metrics"""
        completion_record = {
            "id": request.id,
            "provider": request.provider,
            "priority": request.priority.name,
            "duration": duration,
            "success": success,
            "retry_count": request.retry_count,
            "completed_at": time.time(),
        }

        self.completed_requests.append(completion_record)

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics"""
        total_requests = len(self.completed_requests)
        successful_requests = sum(1 for r in self.completed_requests if r["success"])

        success_rate = (
            successful_requests / total_requests if total_requests > 0 else 0.0
        )

        avg_duration = (
            np.mean([r["duration"] for r in self.completed_requests])
            if self.completed_requests
            else 0.0
        )

        # Priority distribution
        priority_counts = defaultdict(int)
        for request in self.completed_requests:
            priority_counts[request["priority"]] += 1

        return {
            "queue_length": len(self.queue),
            "active_requests": len(self.active_requests),
            "completed_requests": total_requests,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "priority_distribution": dict(priority_counts),
            "max_concurrent": self.max_concurrent,
        }


class APIUsageMonitor:
    """Monitor API usage, costs, and performance metrics"""

    def __init__(self):
        self.providers: dict[str, dict[str, Any]] = {}
        self.hourly_stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=24))
        self.daily_stats: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self._lock = threading.Lock()

        logger.info("APIUsageMonitor initialized")

    def register_provider(
        self,
        provider: str,
        cost_per_request: float = 0.0,
        quota_limits: dict[str, int] | None = None,
    ):
        """Register a provider for monitoring"""
        with self._lock:
            self.providers[provider] = {
                "cost_per_request": cost_per_request,
                "quota_limits": quota_limits or {},
                "total_requests": 0,
                "total_cost": 0.0,
                "success_count": 0,
                "error_count": 0,
                "total_response_time": 0.0,
                "last_request_time": None,
            }

        logger.info("Registered provider %s for monitoring", provider)

    def record_request(
        self,
        provider: str,
        success: bool,
        response_time: float,
        cost: float | None = None,
        error_type: str | None = None,
    ):
        """Record an API request"""
        if provider not in self.providers:
            self.register_provider(provider)

        with self._lock:
            stats = self.providers[provider]

            # Update totals
            stats["total_requests"] += 1
            stats["total_response_time"] += response_time
            stats["last_request_time"] = datetime.now()

            if success:
                stats["success_count"] += 1
            else:
                stats["error_count"] += 1

            # Calculate cost
            actual_cost = cost if cost is not None else stats["cost_per_request"]

            stats["total_cost"] += actual_cost

            # Record hourly stats
            now = datetime.now()
            hour_key = now.strftime("%Y-%m-%d-%H")

            if (
                not self.hourly_stats[provider]
                or self.hourly_stats[provider][-1]["hour"] != hour_key
            ):
                self.hourly_stats[provider].append(
                    {
                        "hour": hour_key,
                        "requests": 0,
                        "cost": 0.0,
                        "errors": 0,
                        "avg_response_time": 0.0,
                        "response_times": [],
                    }
                )

            hour_stat = self.hourly_stats[provider][-1]
            hour_stat["requests"] += 1
            hour_stat["cost"] += actual_cost
            if not success:
                hour_stat["errors"] += 1
            hour_stat["response_times"].append(response_time)
            hour_stat["avg_response_time"] = np.mean(hour_stat["response_times"])

            # Record daily stats
            day_key = now.strftime("%Y-%m-%d")

            if (
                not self.daily_stats[provider]
                or self.daily_stats[provider][-1]["day"] != day_key
            ):
                self.daily_stats[provider].append(
                    {
                        "day": day_key,
                        "requests": 0,
                        "cost": 0.0,
                        "errors": 0,
                        "avg_response_time": 0.0,
                        "response_times": [],
                    }
                )

            day_stat = self.daily_stats[provider][-1]
            day_stat["requests"] += 1
            day_stat["cost"] += actual_cost
            if not success:
                day_stat["errors"] += 1
            day_stat["response_times"].append(response_time)
            day_stat["avg_response_time"] = np.mean(day_stat["response_times"])

    def get_provider_stats(self, provider: str) -> dict[str, Any]:
        """Get statistics for a specific provider"""
        if provider not in self.providers:
            return {"error": f"Provider {provider} not found"}

        with self._lock:
            stats = self.providers[provider].copy()

            # Calculate derived metrics
            if stats["total_requests"] > 0:
                stats["success_rate"] = stats["success_count"] / stats["total_requests"]
                stats["error_rate"] = stats["error_count"] / stats["total_requests"]
                stats["avg_response_time"] = (
                    stats["total_response_time"] / stats["total_requests"]
                )
            else:
                stats["success_rate"] = 0.0
                stats["error_rate"] = 0.0
                stats["avg_response_time"] = 0.0

            # Add recent hourly data
            stats["hourly_data"] = list(self.hourly_stats[provider])[
                -24:
            ]  # Last 24 hours
            stats["daily_data"] = list(self.daily_stats[provider])[-30:]  # Last 30 days

            return stats

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all providers"""
        all_stats = {}
        total_cost = 0.0
        total_requests = 0

        for provider in self.providers:
            stats = self.get_provider_stats(provider)
            all_stats[provider] = stats
            total_cost += stats["total_cost"]
            total_requests += stats["total_requests"]

        return {
            "providers": all_stats,
            "totals": {
                "cost": total_cost,
                "requests": total_requests,
                "avg_cost_per_request": total_cost / total_requests
                if total_requests > 0
                else 0.0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_cost_alerts(self, daily_threshold: float = 100.0) -> list[dict[str, Any]]:
        """Get cost alerts for providers exceeding thresholds"""
        alerts = []

        for provider in self.providers:
            daily_data = list(self.daily_stats[provider])
            if daily_data:
                today_cost = daily_data[-1]["cost"]
                if today_cost > daily_threshold:
                    alerts.append(
                        {
                            "provider": provider,
                            "type": "daily_cost_exceeded",
                            "current_cost": today_cost,
                            "threshold": daily_threshold,
                            "message": f"Daily cost for {provider} (${today_cost:.2f}) exceeded threshold (${daily_threshold:.2f})",
                        }
                    )

        return alerts


class FallbackManager:
    """Manage graceful fallbacks when APIs are unavailable"""

    def __init__(self, cache_manager: SmartCacheManager):
        self.cache_manager = cache_manager
        self.fallback_strategies: dict[str, list[Callable]] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        logger.info("FallbackManager initialized")

    def register_fallback(self, provider: str, strategies: list[Callable]):
        """Register fallback strategies for a provider"""
        self.fallback_strategies[provider] = strategies
        logger.info(
            "Registered %s fallback strategies for %s", len(strategies), provider
        )

    def register_circuit_breaker(
        self, provider: str, failure_threshold: int = 5, recovery_timeout: int = 60
    ):
        """Register circuit breaker for a provider"""
        from airsenal.framework.xg_data_provider import CircuitBreaker

        self.circuit_breakers[provider] = CircuitBreaker(
            failure_threshold, recovery_timeout
        )
        logger.info("Registered circuit breaker for %s", provider)

    async def execute_with_fallback(
        self,
        provider: str,
        primary_func: Callable,
        cache_key: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with fallback strategies"""
        # Try circuit breaker if available
        circuit_breaker = self.circuit_breakers.get(provider)

        try:
            if circuit_breaker:
                if circuit_breaker.state == CircuitBreakerState.OPEN:
                    logger.warning(
                        "Circuit breaker OPEN for %s, using fallback", provider
                    )
                    return await self._execute_fallback(
                        provider, cache_key, *args, **kwargs
                    )

                # Execute with circuit breaker
                result = circuit_breaker.call(primary_func, *args, **kwargs)

                # Handle async functions
                if asyncio.iscoroutine(result):
                    result = await result

                return result
            # Execute without circuit breaker
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            return primary_func(*args, **kwargs)

        except Exception as e:
            logger.error("Primary function failed for %s: %s", provider, e)
            return await self._execute_fallback(provider, cache_key, *args, **kwargs)

    async def _execute_fallback(
        self, provider: str, cache_key: str | None = None, *args, **kwargs
    ) -> Any:
        """Execute fallback strategies"""
        # First, try cached data
        if cache_key:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.info("Using cached data for %s fallback", provider)
                return cached_data

        # Try registered fallback strategies
        strategies = self.fallback_strategies.get(provider, [])

        for i, strategy in enumerate(strategies):
            try:
                if asyncio.iscoroutinefunction(strategy):
                    result = await strategy(*args, **kwargs)
                else:
                    result = strategy(*args, **kwargs)

                if result is not None:
                    logger.info(
                        "Fallback strategy %s succeeded for %s", i + 1, provider
                    )

                    # Cache the fallback result
                    if cache_key:
                        await self.cache_manager.set(
                            cache_key,
                            result,
                            ttl=1800,  # 30 minutes for fallback data
                            priority=Priority.LOW,
                        )

                    return result

            except Exception as e:
                logger.warning(
                    "Fallback strategy %s failed for %s: %s", i + 1, provider, e
                )
                continue

        # All fallbacks failed
        logger.error("All fallback strategies failed for %s", provider)
        msg = f"All fallback strategies failed for {provider}"
        raise Exception(msg)


class APIRequestContext:
    """Context manager for making rate-limited API requests"""

    def __init__(
        self,
        provider: str,
        rate_limiter: APIRateLimiter,
        cache_manager: SmartCacheManager,
        usage_monitor: APIUsageMonitor,
        fallback_manager: FallbackManager,
        priority: Priority = Priority.MEDIUM,
    ):
        self.provider = provider
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager
        self.usage_monitor = usage_monitor
        self.fallback_manager = fallback_manager
        self.priority = priority
        self.start_time: float | None = None

    async def __aenter__(self):
        """Acquire rate limit permit"""
        success = await self.rate_limiter.acquire_permit(self.provider)
        if not success:
            msg = f"Rate limit exceeded for {self.provider}"
            raise Exception(msg)

        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Record request completion"""
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None

            self.usage_monitor.record_request(
                self.provider,
                success,
                duration,
                error_type=str(exc_type) if exc_type else None,
            )

    async def make_request(
        self,
        func: Callable,
        cache_key: str | None = None,
        ttl: float = 3600,
        use_fallback: bool = True,
        *args,
        **kwargs,
    ) -> Any:
        """Make a rate-limited API request with caching"""
        # Try cache first
        if cache_key:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for %s", cache_key)
                return cached_result

        # Make API request
        if use_fallback:
            result = await self.fallback_manager.execute_with_fallback(
                self.provider, func, cache_key, *args, **kwargs
            )
        else:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

        # Cache result
        if cache_key and result is not None:
            await self.cache_manager.set(cache_key, result, ttl, self.priority)

        return result


class APIManager:
    """Central API management system coordinating all components"""

    def __init__(
        self,
        max_memory_cache_entries: int = 1000,
        max_concurrent_requests: int = 10,
        default_request_timeout: float = 30.0,
    ):
        self.rate_limiter = APIRateLimiter()
        self.cache_manager = SmartCacheManager(
            max_memory_entries=max_memory_cache_entries
        )
        self.usage_monitor = APIUsageMonitor()
        self.fallback_manager = FallbackManager(self.cache_manager)
        self.request_queue = RequestQueue(max_concurrent=max_concurrent_requests)

        self.default_timeout = default_request_timeout
        self._initialized = False

        logger.info("APIManager initialized")

    async def initialize(self):
        """Initialize the API manager"""
        if not self._initialized:
            await self.request_queue.start()
            self._initialized = True
            logger.info("APIManager initialization complete")

    async def shutdown(self):
        """Shutdown the API manager"""
        if self._initialized:
            await self.request_queue.stop()
            self._initialized = False
            logger.info("APIManager shutdown complete")

    def configure_provider(
        self,
        provider: str,
        requests_per_second: float = 1.0,
        burst_capacity: int = 5,
        daily_limit: int = 10000,
        hourly_limit: int = 1000,
        cost_per_request: float = 0.0,
        fallback_strategies: list[Callable] | None = None,
    ):
        """Configure a provider with rate limiting and monitoring"""
        # Configure rate limiter
        self.rate_limiter.configure_provider(
            provider,
            requests_per_second,
            burst_capacity,
            daily_limit,
            hourly_limit,
            cost_per_request,
        )

        # Register with usage monitor
        self.usage_monitor.register_provider(
            provider, cost_per_request, {"daily": daily_limit, "hourly": hourly_limit}
        )

        # Register circuit breaker
        self.fallback_manager.register_circuit_breaker(provider)

        # Register fallback strategies
        if fallback_strategies:
            self.fallback_manager.register_fallback(provider, fallback_strategies)

        logger.info("Configured provider %s", provider)

    @asynccontextmanager
    async def request_context(
        self, provider: str, priority: Priority = Priority.MEDIUM
    ):
        """Create context for making API requests"""
        if not self._initialized:
            await self.initialize()

        context = APIRequestContext(
            provider,
            self.rate_limiter,
            self.cache_manager,
            self.usage_monitor,
            self.fallback_manager,
            priority,
        )

        async with context:
            yield context

    async def queue_request(
        self,
        provider: str,
        func: Callable,
        priority: Priority = Priority.MEDIUM,
        cache_key: str | None = None,
        ttl: float = 3600,
        timeout: float | None = None,
        *args,
        **kwargs,
    ) -> Any:
        """Queue an API request for processing"""
        if not self._initialized:
            await self.initialize()

        timeout = timeout or self.default_timeout

        async def _wrapped_request():
            async with self.request_context(provider, priority) as ctx:
                return await ctx.make_request(
                    func, cache_key, ttl, True, *args, **kwargs
                )

        return await self.request_queue.enqueue(
            provider,
            _wrapped_request,
            priority,
            timeout,
            3,  # max_retries
        )

    def get_cache_manager(self) -> SmartCacheManager:
        """Get the cache manager instance"""
        return self.cache_manager

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics from all components"""
        return {
            "rate_limiter": self.rate_limiter.get_all_status(),
            "cache": self.cache_manager.get_stats(),
            "queue": self.request_queue.get_stats(),
            "usage": self.usage_monitor.get_all_stats(),
            "cost_alerts": self.usage_monitor.get_cost_alerts(),
            "timestamp": datetime.now().isoformat(),
        }


# Decorator for easy API management integration
def api_managed(
    provider: str,
    cache_key_func: Callable | None = None,
    ttl: float = 3600,
    priority: Priority = Priority.MEDIUM,
    use_queue: bool = False,
):
    """Decorator to automatically manage API calls"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create global API manager
            if not hasattr(wrapper, "_api_manager"):
                wrapper._api_manager = APIManager()
                await wrapper._api_manager.initialize()

            api_manager = wrapper._api_manager

            # Generate cache key
            cache_key = None
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            elif hasattr(func, "__name__"):
                # Simple default cache key
                key_parts = [func.__name__] + [str(arg) for arg in args]
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Execute request
            if use_queue:
                return await api_manager.queue_request(
                    provider, func, priority, cache_key, ttl, None, *args, **kwargs
                )
            async with api_manager.request_context(provider, priority) as ctx:
                return await ctx.make_request(
                    func, cache_key, ttl, True, *args, **kwargs
                )

        return wrapper

    return decorator


# Global API manager instance
_global_api_manager: APIManager | None = None


async def get_api_manager() -> APIManager:
    """Get or create global API manager instance"""
    global _global_api_manager

    if _global_api_manager is None:
        _global_api_manager = APIManager()
        await _global_api_manager.initialize()

    return _global_api_manager


# Export main classes
__all__ = [
    "APIManager",
    "APIRateLimiter",
    "APIUsageMonitor",
    "FallbackManager",
    "Priority",
    "RequestQueue",
    "SmartCacheManager",
    "api_managed",
    "get_api_manager",
]
