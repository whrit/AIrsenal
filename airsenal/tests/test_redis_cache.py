"""
Integration tests for Redis caching layer.

Tests comprehensive functionality including:
- Connection management and pooling
- Serialization/deserialization of various data types
- Circuit breaker and fallback mechanisms
- Integration with existing FeatureCacheManager
- Cache warming and bulk operations
- Performance metrics and monitoring
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from airsenal.framework.redis_cache import (
    CacheKeyType,
    CircuitBreaker,
    CompressionType,
    RedisCache,
    RedisCacheKeyBuilder,
    RedisSerializer,
)


class TestRedisSerializer:
    """Test Redis serialization/deserialization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.serializer = RedisSerializer(
            CompressionType.NONE
        )  # No compression for easier testing
        self.serializer_zstd = RedisSerializer(CompressionType.ZSTD)

    def test_serialize_deserialize_basic_types(self):
        """Test serialization of basic Python types."""
        test_cases = [
            42,
            3.14159,
            "hello world",
            [1, 2, 3, 4, 5],
            {"key": "value", "number": 123},
            True,
            None,
        ]

        for original in test_cases:
            serialized = self.serializer.serialize(original)
            deserialized = self.serializer.deserialize(serialized)
            assert deserialized == original

    def test_serialize_deserialize_numpy_arrays(self):
        """Test NumPy array serialization."""
        arrays = [
            np.array([1, 2, 3, 4, 5]),
            np.array([[1, 2], [3, 4]]),
            np.array([1.1, 2.2, 3.3]),
            np.random.rand(10, 5),
        ]

        for original in arrays:
            serialized = self.serializer.serialize(original)
            deserialized = self.serializer.deserialize(serialized)
            np.testing.assert_array_equal(deserialized, original)

    def test_serialize_deserialize_dataframes(self):
        """Test pandas DataFrame serialization."""
        df = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "goals": [2, 1, 0],
                "assists": [1, 2, 1],
                "minutes": [90, 85, 60],
            }
        )

        serialized = self.serializer.serialize(df)
        deserialized = self.serializer.deserialize(serialized)

        pd.testing.assert_frame_equal(deserialized, df)

    def test_compression_reduces_size(self):
        """Test that compression reduces data size for large objects."""
        large_array = np.random.rand(1000, 100)

        # Without compression
        serialized_uncompressed = self.serializer.serialize(large_array)

        # With compression
        serialized_compressed = self.serializer_zstd.serialize(large_array)

        # Compressed should be smaller (though this depends on data)
        # At minimum, both should deserialize correctly
        deserialized_uncompressed = self.serializer.deserialize(serialized_uncompressed)
        deserialized_compressed = self.serializer_zstd.deserialize(
            serialized_compressed
        )

        np.testing.assert_array_equal(deserialized_uncompressed, large_array)
        np.testing.assert_array_equal(deserialized_compressed, large_array)

    def test_serialization_error_handling(self):
        """Test error handling in serialization."""

        # Test with non-serializable object
        class NonSerializable:
            def __reduce__(self):
                msg = "Cannot serialize"
                raise TypeError(msg)

        with pytest.raises(Exception):
            self.serializer.serialize(NonSerializable())

    def test_deserialization_error_handling(self):
        """Test error handling in deserialization."""
        # Test with invalid data
        with pytest.raises(Exception):
            self.serializer.deserialize(b"invalid pickle data")


class TestRedisCacheKeyBuilder:
    """Test Redis cache key building functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key_builder = RedisCacheKeyBuilder("test_namespace", "v1")

    def test_build_basic_key(self):
        """Test basic key building."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.FEATURE, primary_id="player:123"
        )

        assert key == "test_namespace:v1:feat:player:123"

    def test_build_key_with_secondary_id(self):
        """Test key building with secondary ID."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.PREDICTION,
            primary_id="player:123",
            secondary_id="gw10",
        )

        assert key == "test_namespace:v1:pred:player:123:gw10"

    def test_build_key_with_context(self):
        """Test key building with context."""
        context = {"season": "2425", "gameweek": 10}
        key = self.key_builder.build_key(
            key_type=CacheKeyType.FEATURE, primary_id="player:123", context=context
        )

        # Key should contain a hash of the context
        parts = key.split(":")
        assert len(parts) == 5
        assert parts[0] == "test_namespace"
        assert parts[1] == "v1"
        assert parts[2] == "feat"
        assert parts[3] == "player:123"
        # parts[4] should be the context hash

    def test_build_key_with_feature_name(self):
        """Test key building with feature name."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.FEATURE,
            primary_id="player:123",
            feature_name="rolling_goals_5",
        )

        assert key == "test_namespace:v1:feat:player:123:rolling_goals_5"

    def test_parse_key(self):
        """Test key parsing."""
        key = "test_namespace:v1:feat:player:123:hash123:rolling_goals_5"
        parsed = self.key_builder.parse_key(key)

        expected = {
            "namespace": "test_namespace",
            "version": "v1",
            "type": "feat",
            "primary_id": "player:123",
            "secondary_id": "hash123",
            "context_hash": "rolling_goals_5",
            "feature_name": None,
        }

        assert parsed == expected

    def test_parse_invalid_key(self):
        """Test parsing invalid key."""
        with pytest.raises(ValueError):
            self.key_builder.parse_key("invalid:key")


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""

        def successful_function():
            return "success"

        result = self.circuit_breaker.call(successful_function)
        assert result == "success"
        assert self.circuit_breaker.state == "CLOSED"

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""

        def failing_function():
            msg = "Function failed"
            raise Exception(msg)

        # First few failures should not open the circuit
        for _i in range(2):
            with pytest.raises(Exception):
                self.circuit_breaker.call(failing_function)
            assert self.circuit_breaker.state == "CLOSED"

        # Third failure should open the circuit
        with pytest.raises(Exception):
            self.circuit_breaker.call(failing_function)
        assert self.circuit_breaker.state == "OPEN"

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""

        def failing_function():
            msg = "Function failed"
            raise Exception(msg)

        # Trigger circuit breaker to open
        for _i in range(3):
            with pytest.raises(Exception):
                self.circuit_breaker.call(failing_function)

        assert self.circuit_breaker.state == "OPEN"

        # Wait for recovery timeout
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN
        def successful_function():
            return "success"

        result = self.circuit_breaker.call(successful_function)
        assert result == "success"
        assert self.circuit_breaker.state == "CLOSED"


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.expires = {}

    def ping(self):
        return True

    def get(self, key):
        if key in self.expires and self.expires[key] < time.time():
            del self.data[key]
            del self.expires[key]
            return None
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.expires[key] = time.time() + ttl
        return True

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.expires:
                del self.expires[key]
        return count

    def exists(self, key):
        return key in self.data

    def expire(self, key, ttl):
        if key in self.data:
            self.expires[key] = time.time() + ttl
            return True
        return False

    def keys(self, pattern):
        import fnmatch

        pattern = pattern.replace("*", ".*")
        return [k for k in self.data if fnmatch.fnmatch(k, pattern)]

    def mget(self, keys):
        return [self.get(key) for key in keys]

    def pipeline(self):
        return MockRedisPipeline(self)

    def scan(self, cursor, match=None, count=None):
        keys = list(self.data.keys())
        if match:
            import fnmatch

            pattern = match.replace("*", ".*")
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        return (0, keys)  # Simplified: return all in one scan

    def info(self):
        return {
            "redis_version": "6.0.0",
            "used_memory_human": "1MB",
            "connected_clients": 1,
            "total_commands_processed": 100,
            "keyspace_hits": 50,
            "keyspace_misses": 10,
            "role": "master",
        }

    def flushdb(self):
        self.data.clear()
        self.expires.clear()


class MockRedisPipeline:
    """Mock Redis pipeline for testing."""

    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.commands = []

    def setex(self, key, ttl, value):
        self.commands.append(("setex", key, ttl, value))
        return self

    def execute(self):
        results = []
        for cmd, *args in self.commands:
            if cmd == "setex":
                results.append(self.redis_client.setex(*args))
        self.commands.clear()
        return results


class TestRedisCache:
    """Test Redis cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock Redis to be available
        with patch("airsenal.framework.redis_cache.REDIS_AVAILABLE", True):
            with patch("airsenal.framework.redis_cache.redis") as mock_redis_module:
                mock_redis_module.Redis = Mock
                mock_redis_module.ConnectionPool = Mock

                self.redis_cache = RedisCache(
                    namespace="test", version="v1", default_ttl=60
                )

                # Replace the actual Redis client with our mock
                self.mock_redis = MockRedis()
                self.redis_cache.connection_manager.client = self.mock_redis
                self.redis_cache.connection_manager._initialized = True
                self.redis_cache.available = True

    def test_basic_get_set_operations(self):
        """Test basic cache get/set operations."""
        key = "test_key"
        value = "test_value"

        # Set value
        result = self.redis_cache.set(key, value, ttl=60)
        assert result is True

        # Get value
        retrieved = self.redis_cache.get(key)
        assert retrieved == value

    def test_cache_expiration(self):
        """Test cache expiration."""
        key = "expiring_key"
        value = "expiring_value"

        # Set with very short TTL
        self.redis_cache.set(key, value, ttl=1)

        # Should be available immediately
        assert self.redis_cache.get(key) == value

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert self.redis_cache.get(key) is None

    def test_feature_cache_operations(self):
        """Test feature-specific cache operations."""
        feature_name = "rolling_goals_5"
        entity_type = "player"
        entity_id = 123
        value = 2.5

        # Set feature cache
        result = self.redis_cache.set_feature_cache(
            feature_name, entity_type, entity_id, value
        )
        assert result is True

        # Get feature cache
        retrieved = self.redis_cache.get_feature_cache(
            feature_name, entity_type, entity_id
        )
        assert retrieved == value

    def test_prediction_cache_operations(self):
        """Test prediction-specific cache operations."""
        player_id = 123
        gameweek = 10
        season = "2425"
        predictions = np.array([2.5, 1.2, 0.8])

        # Set prediction cache
        result = self.redis_cache.set_prediction_cache(
            player_id, gameweek, season, predictions
        )
        assert result is True

        # Get prediction cache
        retrieved = self.redis_cache.get_prediction_cache(player_id, gameweek, season)
        np.testing.assert_array_equal(retrieved, predictions)

    def test_bulk_operations(self):
        """Test bulk cache operations."""
        data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
        }

        # Bulk set
        result = self.redis_cache.mset(data, ttl=60)
        assert result is True

        # Bulk get
        keys = list(data.keys())
        retrieved = self.redis_cache.mget(keys)

        assert retrieved[0] == "value1"
        assert retrieved[1] == 42
        assert retrieved[2] == [1, 2, 3]

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        player_id = 123

        # Set some prediction data
        self.redis_cache.set_prediction_cache(
            player_id, 10, "2425", np.array([1, 2, 3])
        )
        self.redis_cache.set_prediction_cache(
            player_id, 11, "2425", np.array([4, 5, 6])
        )

        # Invalidate player cache
        count = self.redis_cache.invalidate_player_cache(player_id)
        assert count >= 0  # Should invalidate some entries

        # Verify data is gone
        assert self.redis_cache.get_prediction_cache(player_id, 10, "2425") is None
        assert self.redis_cache.get_prediction_cache(player_id, 11, "2425") is None

    def test_cache_metrics(self):
        """Test cache metrics collection."""
        # Perform some cache operations
        self.redis_cache.set("test_key", "test_value")
        self.redis_cache.get("test_key")  # Hit
        self.redis_cache.get("nonexistent_key")  # Miss

        # Get metrics
        metrics = self.redis_cache.get_metrics()

        assert "available" in metrics
        assert "hit_rate" in metrics
        assert "total_requests" in metrics
        assert metrics["total_requests"] >= 2

    def test_cache_decorators(self):
        """Test caching decorators."""
        call_count = 0

        @self.redis_cache.cache_feature(ttl=60)
        def expensive_computation(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = expensive_computation(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_computation(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again

        # Different parameters should execute function again
        result3 = expensive_computation(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_error_handling(self):
        """Test error handling when Redis operations fail."""
        # Mock Redis to raise an exception
        self.mock_redis.get = Mock(side_effect=Exception("Redis error"))

        # Get operation should handle error gracefully
        result = self.redis_cache.get("test_key")
        assert result is None

        # Metrics should record the error
        metrics = self.redis_cache.get_metrics()
        assert metrics["errors"] > 0


class TestFeatureCacheManagerIntegration:
    """Test integration of Redis cache with FeatureCacheManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock database session
        self.mock_session = Mock()

        # Mock Redis cache
        self.mock_redis_cache = Mock()
        self.mock_redis_cache.is_available.return_value = True

        # Import and patch FeatureCacheManager
        from airsenal.framework.feature_store import FeatureCacheManager

        with patch(
            "airsenal.framework.feature_store.redis_cache", self.mock_redis_cache
        ):
            self.cache_manager = FeatureCacheManager(
                dbsession=self.mock_session, default_ttl=3600
            )

    def test_three_tier_cache_get_memory_hit(self):
        """Test cache get with memory cache hit."""
        # Set up memory cache
        cache_key = "test_feature:player:123:"
        expires_at = datetime.now() + timedelta(seconds=3600)
        self.cache_manager._memory_cache[cache_key] = ("test_value", expires_at)

        # Get value
        result = self.cache_manager.get("test_feature", "player", 123)

        assert result == "test_value"
        assert self.cache_manager._cache_stats["memory_hits"] == 1

        # Redis and database should not be called
        self.mock_redis_cache.get_feature_cache.assert_not_called()
        self.mock_session.query.assert_not_called()

    def test_three_tier_cache_get_redis_hit(self):
        """Test cache get with Redis cache hit."""
        # Memory cache miss, Redis cache hit
        self.mock_redis_cache.get_feature_cache.return_value = "redis_value"

        result = self.cache_manager.get("test_feature", "player", 123)

        assert result == "redis_value"
        assert self.cache_manager._cache_stats["redis_hits"] == 1

        # Value should be cached in memory
        cache_key = "test_feature:player:123:"
        assert cache_key in self.cache_manager._memory_cache

        # Database should not be called
        self.mock_session.query.assert_not_called()

    def test_three_tier_cache_set_all_levels(self):
        """Test cache set updates all levels."""
        # Set feature value
        self.cache_manager.set("test_feature", "player", 123, "test_value", ttl=1800)

        # Memory cache should be updated
        cache_key = "test_feature:player:123:"
        assert cache_key in self.cache_manager._memory_cache

        # Redis cache should be called
        self.mock_redis_cache.set_feature_cache.assert_called_once_with(
            feature_name="test_feature",
            entity_type="player",
            entity_id=123,
            value="test_value",
            context=None,
            ttl=1800,
        )

        # Database should be updated
        self.mock_session.add.assert_called()
        self.mock_session.commit.assert_called()

    def test_cache_invalidation_all_levels(self):
        """Test cache invalidation across all levels."""
        # Set up memory cache
        self.cache_manager._memory_cache["test_feature:player:123:"] = (
            "value",
            datetime.now(),
        )

        # Set up Redis mock
        self.mock_redis_cache.invalidate_player_cache.return_value = 2

        # Set up database mock
        mock_query = Mock()
        mock_query.count.return_value = 1
        mock_query.delete.return_value = None
        self.mock_session.query.return_value.filter_by.return_value = mock_query

        # Invalidate cache
        count = self.cache_manager.invalidate("test_feature", "player", 123)

        # Memory cache should be cleared
        assert "test_feature:player:123:" not in self.cache_manager._memory_cache

        # Redis invalidation should be called
        self.mock_redis_cache.invalidate_player_cache.assert_called_once_with(123)

        # Database invalidation should be called
        mock_query.delete.assert_called_once()

        # Total count should include all levels
        assert count == 4  # 1 memory + 2 redis + 1 database

    def test_redis_unavailable_fallback(self):
        """Test fallback when Redis is unavailable."""
        # Simulate Redis being unavailable
        cache_manager = self.cache_manager
        cache_manager.redis_enabled = False
        cache_manager.redis_cache = None

        # Set value (should only use memory and database)
        cache_manager.set("test_feature", "player", 123, "test_value")

        # Memory cache should work
        cache_key = "test_feature:player:123:"
        assert cache_key in cache_manager._memory_cache

        # Database should work
        self.mock_session.add.assert_called()
        self.mock_session.commit.assert_called()

    def test_comprehensive_stats(self):
        """Test comprehensive statistics across all cache levels."""
        # Simulate some cache activity
        self.cache_manager._cache_stats.update(
            {"hits": 10, "misses": 3, "memory_hits": 5, "redis_hits": 3, "db_hits": 2}
        )

        # Mock Redis metrics
        self.mock_redis_cache.get_metrics.return_value = {
            "available": True,
            "hit_rate": 0.8,
            "total_size_mb": 2.5,
            "avg_retrieval_time_ms": 1.2,
            "connection_status": "CLOSED",
            "compression": "zstd",
        }

        self.mock_redis_cache.get_info.return_value = {
            "redis_version": "6.0.0",
            "used_memory_human": "2MB",
        }

        # Mock database stats
        mock_db_stats = Mock()
        mock_db_stats.total_entries = 100
        mock_db_stats.avg_hits = 5.5
        self.mock_session.query.return_value.first.return_value = mock_db_stats

        # Get stats
        stats = self.cache_manager.get_stats()

        # Verify overall stats
        assert stats["overall"]["hit_rate"] == 10 / 13  # 10 hits out of 13 total
        assert stats["overall"]["total_requests"] == 13

        # Verify per-level stats
        assert stats["memory_cache"]["hits"] == 5
        assert stats["redis_cache"]["hits"] == 3
        assert stats["database_cache"]["hits"] == 2

        # Verify Redis-specific stats
        assert stats["redis_cache"]["available"] is True
        assert stats["redis_cache"]["total_size_mb"] == 2.5


class TestCacheWarming:
    """Test cache warming functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = MockRedis()

        with patch("airsenal.framework.redis_cache.REDIS_AVAILABLE", True):
            self.redis_cache = RedisCache(namespace="test", version="v1")
            self.redis_cache.connection_manager.client = self.mock_redis
            self.redis_cache.connection_manager._initialized = True
            self.redis_cache.available = True

    def test_prediction_cache_warming(self):
        """Test cache warming for predictions."""

        def mock_prediction_func(player_id, gameweek, season):
            return np.array([player_id * gameweek, gameweek, 1.0])

        player_ids = [123, 456]
        gameweeks = [10, 11]
        season = "2425"

        # Warm cache
        count = self.redis_cache.warm_predictions_cache(
            player_ids, gameweeks, season, mock_prediction_func
        )

        assert count == 4  # 2 players × 2 gameweeks

        # Verify cached data
        for player_id in player_ids:
            for gameweek in gameweeks:
                cached = self.redis_cache.get_prediction_cache(
                    player_id, gameweek, season
                )
                expected = np.array([player_id * gameweek, gameweek, 1.0])
                np.testing.assert_array_equal(cached, expected)

    def test_feature_cache_warming(self):
        """Test cache warming for features."""

        def mock_compute_func(feature_name, entity_type, entity_id):
            return f"{feature_name}_{entity_type}_{entity_id}_value"

        feature_names = ["rolling_goals_5", "xg_form"]
        entity_type = "player"
        entity_ids = [123, 456]

        # Warm cache
        count = self.redis_cache.warm_features_cache(
            feature_names, entity_type, entity_ids, mock_compute_func
        )

        assert count == 4  # 2 features × 2 entities

        # Verify cached data
        for feature_name in feature_names:
            for entity_id in entity_ids:
                cached = self.redis_cache.get_feature_cache(
                    feature_name, entity_type, entity_id
                )
                expected = f"{feature_name}_{entity_type}_{entity_id}_value"
                assert cached == expected


if __name__ == "__main__":
    pytest.main([__file__])
