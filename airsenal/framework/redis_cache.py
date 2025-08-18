"""
Redis Caching Layer for AIrsenal

High-performance Redis-based caching system for frequently accessed predictions,
computed features, and expensive database queries. Supports both single Redis
instances and Redis clusters with comprehensive fallback mechanisms.

Key Features:
- Connection pooling and circuit breaker patterns
- Hierarchical cache keys with versioning
- Efficient serialization for NumPy arrays and pandas DataFrames
- Compression with zstandard for large objects
- Comprehensive monitoring and metrics
- Graceful fallback when Redis is unavailable
- Cache warming strategies for hot data

Usage:
    # Initialize Redis cache
    redis_cache = RedisCache()

    # Cache predictions
    redis_cache.set_prediction_cache(
        player_id=123,
        gameweek=10,
        season="2425",
        predictions=predictions_array
    )

    # Retrieve cached data
    cached_data = redis_cache.get_prediction_cache(
        player_id=123,
        gameweek=10,
        season="2425"
    )

    # Use caching decorators
    @redis_cache.cache_feature(ttl=1800)
    def compute_rolling_average(player_id, window=5):
        # Expensive computation here
        return result
"""

import json
import logging
import pickle
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

try:
    import redis
    from redis.cluster import RedisCluster
    from redis.connection import ConnectionPool
    from redis.sentinel import Sentinel

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    Sentinel = None  # type: ignore
    RedisCluster = None  # type: ignore

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None  # type: ignore

try:
    import orjson

    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    orjson = None  # type: ignore

from airsenal.framework.env import (
    AIRSENAL_REDIS_CLUSTER,
    AIRSENAL_REDIS_CLUSTER_NODES,
    AIRSENAL_REDIS_COMPRESSION,
    AIRSENAL_REDIS_CONNECTION_TIMEOUT,
    AIRSENAL_REDIS_DB,
    AIRSENAL_REDIS_DEFAULT_TTL,
    AIRSENAL_REDIS_ENABLE,
    AIRSENAL_REDIS_HOST,
    AIRSENAL_REDIS_MAX_CONNECTIONS,
    AIRSENAL_REDIS_PASSWORD,
    AIRSENAL_REDIS_PORT,
    AIRSENAL_REDIS_SOCKET_TIMEOUT,
)

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Available compression algorithms."""

    NONE = "none"
    ZSTD = "zstd"
    GZIP = "gzip"


class CacheKeyType(Enum):
    """Cache key type prefixes for hierarchical organization."""

    PREDICTION = "pred"
    FEATURE = "feat"
    QUERY = "query"
    METADATA = "meta"
    WARMING = "warm"


@dataclass
class CacheMetrics:
    """Redis cache performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    avg_retrieval_time_ms: float = 0.0
    connection_failures: int = 0


class CircuitBreaker:
    """Circuit breaker for Redis connection failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if (
                    self.last_failure_time is not None
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF_OPEN"
                else:
                    msg = "Circuit breaker is OPEN"
                    raise Exception(msg)

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

                raise e


class RedisSerializer:
    """Handles serialization/deserialization of various data types for Redis storage."""

    def __init__(self, compression: CompressionType = CompressionType.ZSTD):
        self.compression = compression
        self._compressor = None
        self._decompressor = None

        if compression == CompressionType.ZSTD and ZSTD_AVAILABLE:
            self._compressor = zstd.ZstdCompressor(level=3)
            self._decompressor = zstd.ZstdDecompressor()
        elif compression == CompressionType.GZIP:
            import gzip

            self._compress_func = gzip.compress
            self._decompress_func = gzip.decompress

    def serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage with optional compression."""
        try:
            # Handle different data types efficiently
            if isinstance(data, np.ndarray):
                serialized = self._serialize_numpy(data)
            elif isinstance(data, pd.DataFrame):
                serialized = self._serialize_dataframe(data)
            elif isinstance(data, dict | list) and ORJSON_AVAILABLE:
                serialized = orjson.dumps(data)
            else:
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            # Apply compression if enabled
            if self.compression == CompressionType.ZSTD and self._compressor:
                return self._compressor.compress(serialized)
            if self.compression == CompressionType.GZIP:
                return self._compress_func(serialized)
            return serialized

        except Exception as e:
            logger.error("Serialization error: %s", e)
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis storage with optional decompression."""
        try:
            # Apply decompression if enabled
            if self.compression == CompressionType.ZSTD and self._decompressor:
                decompressed = self._decompressor.decompress(data)
            elif self.compression == CompressionType.GZIP:
                decompressed = self._decompress_func(data)
            else:
                decompressed = data

            # Try different deserialization methods
            try:
                # Try orjson first for JSON data
                if ORJSON_AVAILABLE:
                    return orjson.loads(decompressed)
            except:
                pass

            try:
                # Try pickle for Python objects
                return pickle.loads(decompressed)
            except:
                pass

            # Try numpy-specific deserialization
            try:
                return self._deserialize_numpy(decompressed)
            except:
                pass

            # Try DataFrame-specific deserialization
            try:
                return self._deserialize_dataframe(decompressed)
            except:
                pass

            # Fallback to raw bytes
            return decompressed

        except Exception as e:
            logger.error("Deserialization error: %s", e)
            raise

    def _serialize_numpy(self, array: np.ndarray) -> bytes:
        """Efficient NumPy array serialization."""
        return pickle.dumps(
            {
                "type": "numpy",
                "data": array.tobytes(),
                "dtype": str(array.dtype),
                "shape": array.shape,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _deserialize_numpy(self, data: bytes) -> np.ndarray:
        """Efficient NumPy array deserialization."""
        obj = pickle.loads(data)
        if obj.get("type") == "numpy":
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        msg = "Not a numpy array"
        raise ValueError(msg)

    def _serialize_dataframe(self, df: pd.DataFrame) -> bytes:
        """Efficient DataFrame serialization using parquet format."""
        return pickle.dumps(
            {"type": "dataframe", "data": df.to_parquet()},
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _deserialize_dataframe(self, data: bytes) -> pd.DataFrame:
        """Efficient DataFrame deserialization from parquet format."""
        obj = pickle.loads(data)
        if obj.get("type") == "dataframe":
            return pd.read_parquet(obj["data"])
        msg = "Not a DataFrame"
        raise ValueError(msg)


class RedisCacheKeyBuilder:
    """Builds hierarchical cache keys with versioning and namespacing."""

    def __init__(self, namespace: str = "airsenal", version: str = "v1"):
        self.namespace = namespace
        self.version = version

    def build_key(
        self,
        key_type: CacheKeyType,
        primary_id: str | int,
        secondary_id: str | int | None = None,
        context: dict[str, Any] | None = None,
        feature_name: str | None = None,
    ) -> str:
        """
        Build hierarchical cache key.

        Format: namespace:version:type:primary[:secondary][:context_hash][:feature]

        Examples:
        - airsenal:v1:pred:player:123:gw10:season2425
        - airsenal:v1:feat:rolling_goals_5:123:gw10
        - airsenal:v1:query:fixtures:2425:upcoming
        """
        parts = [self.namespace, self.version, key_type.value, str(primary_id)]

        if secondary_id is not None:
            parts.append(str(secondary_id))

        if context:
            # Create stable hash of context for consistent keys
            context_str = json.dumps(context, sort_keys=True)
            context_hash = str(hash(context_str))
            parts.append(context_hash)

        if feature_name:
            parts.append(feature_name)

        return ":".join(parts)

    def parse_key(self, key: str) -> dict[str, str | None]:
        """Parse cache key back into components."""
        parts = key.split(":")
        if len(parts) < 4:
            msg = f"Invalid cache key format: {key}"
            raise ValueError(msg)

        parsed: dict[str, str | None] = {
            "namespace": parts[0],
            "version": parts[1],
            "type": parts[2],
            "primary_id": parts[3],
            "secondary_id": parts[4] if len(parts) > 4 else None,
            "context_hash": parts[5] if len(parts) > 5 else None,
            "feature_name": parts[6] if len(parts) > 6 else None,
        }
        return parsed


class RedisConnectionManager:
    """Manages Redis connections with pooling, clustering, and circuit breaker."""

    def __init__(self):
        self.pool: ConnectionPool | None = None
        self.cluster: RedisCluster | None = None
        self.client: redis.Redis | None = None
        self.circuit_breaker = CircuitBreaker()
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize Redis connection with configuration."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
            return False

        if not AIRSENAL_REDIS_ENABLE:
            logger.info("Redis caching disabled by configuration")
            return False

        with self._lock:
            if self._initialized:
                return True

            try:
                if AIRSENAL_REDIS_CLUSTER:
                    self._init_cluster()
                else:
                    self._init_single()

                # Test connection
                self.circuit_breaker.call(self._test_connection)
                self._initialized = True
                logger.info("Redis connection initialized successfully")
                return True

            except Exception as e:
                logger.error("Failed to initialize Redis connection: %s", e)
                return False

    def _init_single(self):
        """Initialize single Redis instance."""
        self.pool = ConnectionPool(
            host=AIRSENAL_REDIS_HOST,
            port=AIRSENAL_REDIS_PORT,
            password=AIRSENAL_REDIS_PASSWORD,
            db=AIRSENAL_REDIS_DB,
            max_connections=AIRSENAL_REDIS_MAX_CONNECTIONS,
            socket_timeout=AIRSENAL_REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=AIRSENAL_REDIS_CONNECTION_TIMEOUT,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def _init_cluster(self):
        """Initialize Redis cluster."""
        if not AIRSENAL_REDIS_CLUSTER_NODES:
            msg = "Redis cluster nodes not specified"
            raise ValueError(msg)

        # Parse cluster nodes
        from redis.cluster import ClusterNode

        nodes = []
        for node in AIRSENAL_REDIS_CLUSTER_NODES.split(","):
            host, port = node.strip().split(":")
            nodes.append(ClusterNode(host, int(port)))

        self.cluster = RedisCluster(
            startup_nodes=nodes,
            password=AIRSENAL_REDIS_PASSWORD,
            socket_timeout=AIRSENAL_REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=AIRSENAL_REDIS_CONNECTION_TIMEOUT,
            max_connections_per_node=AIRSENAL_REDIS_MAX_CONNECTIONS // len(nodes),
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self.client = self.cluster  # type: ignore

    def _test_connection(self):
        """Test Redis connection."""
        if self.client:
            self.client.ping()

    @contextmanager
    def get_connection(self):
        """Get Redis connection with circuit breaker protection."""
        if not self._initialized:
            msg = "Redis connection not initialized"
            raise Exception(msg)

        try:
            yield self.circuit_breaker.call(lambda: self.client)
        except Exception as e:
            logger.error("Redis connection error: %s", e)
            raise

    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._initialized and self.circuit_breaker.state != "OPEN"


class RedisCache:
    """
    High-performance Redis caching layer for AIrsenal.

    Provides comprehensive caching functionality with:
    - Hierarchical key management
    - Efficient serialization/compression
    - Circuit breaker and fallback mechanisms
    - Performance monitoring and metrics
    - Cache warming strategies
    """

    def __init__(
        self,
        namespace: str = "airsenal",
        version: str = "v1",
        default_ttl: int | None = None,
    ):
        self.namespace = namespace
        self.version = version
        self.default_ttl = default_ttl or AIRSENAL_REDIS_DEFAULT_TTL

        # Initialize components
        self.connection_manager = RedisConnectionManager()
        self.key_builder = RedisCacheKeyBuilder(namespace, version)

        # Choose compression based on configuration
        compression_map = {
            "zstd": CompressionType.ZSTD,
            "gzip": CompressionType.GZIP,
            "none": CompressionType.NONE,
        }
        compression = compression_map.get(
            AIRSENAL_REDIS_COMPRESSION, CompressionType.ZSTD
        )
        self.serializer = RedisSerializer(compression)

        # Metrics tracking
        self.metrics = CacheMetrics()
        self._metrics_lock = threading.Lock()

        # Initialize connection
        self.available = self.connection_manager.initialize()

    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        return self.available and self.connection_manager.is_available()

    # Core cache operations

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.is_available():
            return None

        start_time = time.time()
        try:
            with self.connection_manager.get_connection() as conn:
                data = conn.get(key)

            if data is None:
                self._record_miss()
                return None

            value = self.serializer.deserialize(data)
            self._record_hit(time.time() - start_time)
            return value

        except Exception as e:
            self._record_error()
            logger.error("Cache get error for key %s: %s", key, e)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self.is_available():
            return False

        ttl = ttl or self.default_ttl

        try:
            serialized_data = self.serializer.serialize(value)

            with self.connection_manager.get_connection() as conn:
                result = conn.setex(key, ttl, serialized_data)

            self._record_set(len(serialized_data))
            return bool(result)

        except Exception as e:
            self._record_error()
            logger.error("Cache set error for key %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_available():
            return False

        try:
            with self.connection_manager.get_connection() as conn:
                result = conn.delete(key)

            self._record_delete()
            return bool(result)

        except Exception as e:
            self._record_error()
            logger.error("Cache delete error for key %s: %s", key, e)
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.is_available():
            return 0

        try:
            with self.connection_manager.get_connection() as conn:
                keys = conn.keys(pattern)
                count = conn.delete(*keys) if keys else 0

            self._record_delete(count)
            return count

        except Exception as e:
            self._record_error()
            logger.error("Cache delete pattern error for pattern %s: %s", pattern, e)
            return 0

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_available():
            return False

        try:
            with self.connection_manager.get_connection() as conn:
                return bool(conn.exists(key))

        except Exception as e:
            self._record_error()
            logger.error("Cache exists error for key %s: %s", key, e)
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        if not self.is_available():
            return False

        try:
            with self.connection_manager.get_connection() as conn:
                return bool(conn.expire(key, ttl))

        except Exception as e:
            self._record_error()
            logger.error("Cache expire error for key %s: %s", key, e)
            return False

    # High-level cache operations for AIrsenal

    def get_prediction_cache(
        self, player_id: int, gameweek: int, season: str, model_version: str = "latest"
    ) -> np.ndarray | None:
        """Get cached player predictions."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.PREDICTION,
            primary_id=f"player:{player_id}",
            context={
                "gameweek": gameweek,
                "season": season,
                "model_version": model_version,
            },
        )
        return self.get(key)

    def set_prediction_cache(
        self,
        player_id: int,
        gameweek: int,
        season: str,
        predictions: np.ndarray,
        model_version: str = "latest",
        ttl: int | None = None,
    ) -> bool:
        """Cache player predictions."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.PREDICTION,
            primary_id=f"player:{player_id}",
            context={
                "gameweek": gameweek,
                "season": season,
                "model_version": model_version,
            },
        )
        # Predictions are typically valid until next gameweek
        cache_ttl = ttl or (7 * 24 * 3600)  # 7 days
        return self.set(key, predictions, cache_ttl)

    def get_feature_cache(
        self,
        feature_name: str,
        entity_type: str,
        entity_id: int,
        context: dict[str, Any] | None = None,
    ) -> Any | None:
        """Get cached feature value."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.FEATURE,
            primary_id=f"{entity_type}:{entity_id}",
            context=context,
            feature_name=feature_name,
        )
        return self.get(key)

    def set_feature_cache(
        self,
        feature_name: str,
        entity_type: str,
        entity_id: int,
        value: Any,
        context: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache feature value."""
        key = self.key_builder.build_key(
            key_type=CacheKeyType.FEATURE,
            primary_id=f"{entity_type}:{entity_id}",
            context=context,
            feature_name=feature_name,
        )
        return self.set(key, value, ttl)

    def invalidate_player_cache(self, player_id: int) -> int:
        """Invalidate all cache entries for a player."""
        pattern = (
            self.key_builder.build_key(
                key_type=CacheKeyType.PREDICTION, primary_id=f"player:{player_id}"
            )
            + "*"
        )
        pred_count = self.delete_pattern(pattern)

        pattern = (
            self.key_builder.build_key(
                key_type=CacheKeyType.FEATURE, primary_id=f"player:{player_id}"
            )
            + "*"
        )
        feat_count = self.delete_pattern(pattern)

        return pred_count + feat_count

    def invalidate_gameweek_cache(self, season: str, gameweek: int) -> int:
        """Invalidate all cache entries for a specific gameweek."""
        # This is more complex due to hierarchical keys
        # We'll need to scan and filter
        pattern = f"{self.namespace}:{self.version}:*"
        deleted = 0

        if not self.is_available():
            return 0

        try:
            with self.connection_manager.get_connection() as conn:
                cursor = 0
                while True:
                    cursor, keys = conn.scan(cursor, match=pattern, count=1000)

                    keys_to_delete = []
                    for key in keys:
                        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                        if f"season{season}" in key_str and f"gw{gameweek}" in key_str:
                            keys_to_delete.append(key)

                    if keys_to_delete:
                        deleted += conn.delete(*keys_to_delete)

                    if cursor == 0:
                        break

        except Exception as e:
            logger.error("Error invalidating gameweek cache: %s", e)

        return deleted

    # Bulk operations for efficiency

    def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values from cache."""
        if not self.is_available() or not keys:
            return [None] * len(keys)

        try:
            with self.connection_manager.get_connection() as conn:
                raw_values = conn.mget(keys)

            results: list[Any | None] = []
            for raw_value in raw_values:
                if raw_value is None:
                    results.append(None)
                    self._record_miss()
                else:
                    try:
                        value = self.serializer.deserialize(raw_value)
                        results.append(value)
                        self._record_hit()
                    except Exception as e:
                        logger.error("Deserialization error in mget: %s", e)
                        results.append(None)
                        self._record_error()

            return results

        except Exception as e:
            self._record_error()
            logger.error("Cache mget error: %s", e)
            return [None] * len(keys)

    def mset(self, key_value_pairs: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple key-value pairs."""
        if not self.is_available() or not key_value_pairs:
            return False

        ttl = ttl or self.default_ttl

        try:
            # Serialize all values first
            serialized_pairs = {}
            total_size = 0

            for key, value in key_value_pairs.items():
                serialized_data = self.serializer.serialize(value)
                serialized_pairs[key] = serialized_data
                total_size += len(serialized_data)

            with self.connection_manager.get_connection() as conn:
                # Use pipeline for efficiency
                pipe = conn.pipeline()

                for key, serialized_data in serialized_pairs.items():
                    pipe.setex(key, ttl, serialized_data)

                pipe.execute()

            self._record_set(total_size, len(key_value_pairs))
            return True

        except Exception as e:
            self._record_error()
            logger.error("Cache mset error: %s", e)
            return False

    # Cache warming strategies

    def warm_predictions_cache(
        self,
        player_ids: list[int],
        gameweeks: list[int],
        season: str,
        prediction_func: Callable,
    ) -> int:
        """Warm cache with predictions for specified players and gameweeks."""
        if not self.is_available():
            return 0

        warmed_count = 0

        for player_id in player_ids:
            for gameweek in gameweeks:
                # Check if already cached
                if self.get_prediction_cache(player_id, gameweek, season):
                    continue

                try:
                    # Generate predictions
                    predictions = prediction_func(player_id, gameweek, season)

                    if predictions is not None:
                        # Cache with extended TTL for warming
                        if self.set_prediction_cache(
                            player_id,
                            gameweek,
                            season,
                            predictions,
                            ttl=self.default_ttl * 2,
                        ):
                            warmed_count += 1

                except Exception as e:
                    logger.error(
                        "Error warming prediction cache for player %s, GW %s: %s",
                        player_id,
                        gameweek,
                        e,
                    )

        logger.info("Warmed %s prediction cache entries", warmed_count)
        return warmed_count

    def warm_features_cache(
        self,
        feature_names: list[str],
        entity_type: str,
        entity_ids: list[int],
        compute_func: Callable,
    ) -> int:
        """Warm cache with features for specified entities."""
        if not self.is_available():
            return 0

        warmed_count = 0

        for feature_name in feature_names:
            for entity_id in entity_ids:
                # Check if already cached
                if self.get_feature_cache(feature_name, entity_type, entity_id):
                    continue

                try:
                    # Compute feature
                    feature_value = compute_func(feature_name, entity_type, entity_id)

                    if feature_value is not None:
                        # Cache with extended TTL for warming
                        if self.set_feature_cache(
                            feature_name,
                            entity_type,
                            entity_id,
                            feature_value,
                            ttl=self.default_ttl * 2,
                        ):
                            warmed_count += 1

                except Exception as e:
                    logger.error(
                        "Error warming feature cache for %s, %s:%s: %s",
                        feature_name,
                        entity_type,
                        entity_id,
                        e,
                    )

        logger.info("Warmed %s feature cache entries", warmed_count)
        return warmed_count

    # Caching decorators

    def cache_feature(self, ttl: int | None = None, key_func: Callable | None = None):
        """Decorator for caching feature computation functions."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation from function name and args
                    args_str = "_".join(str(arg) for arg in args)
                    kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = f"func:{func.__name__}:{args_str}:{kwargs_str}"

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Compute and cache result
                result = func(*args, **kwargs)
                if result is not None:
                    self.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def cache_query(self, ttl: int | None = None, key_prefix: str = "query"):
        """Decorator for caching database query results."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key for query
                args_str = "_".join(str(arg) for arg in args)
                kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"

                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute query and cache result
                result = func(*args, **kwargs)
                if result is not None:
                    # Queries typically have shorter TTL
                    query_ttl = ttl or (self.default_ttl // 2)
                    self.set(cache_key, result, query_ttl)

                return result

            return wrapper

        return decorator

    # Monitoring and metrics

    def _record_hit(self, retrieval_time: float = 0.0):
        """Record cache hit."""
        with self._metrics_lock:
            self.metrics.hits += 1
            if retrieval_time > 0:
                # Update running average
                total_requests = self.metrics.hits + self.metrics.misses
                self.metrics.avg_retrieval_time_ms = (
                    self.metrics.avg_retrieval_time_ms * (total_requests - 1)
                    + retrieval_time * 1000
                ) / total_requests

    def _record_miss(self):
        """Record cache miss."""
        with self._metrics_lock:
            self.metrics.misses += 1

    def _record_set(self, size_bytes: int = 0, count: int = 1):
        """Record cache set operation."""
        with self._metrics_lock:
            self.metrics.sets += count
            self.metrics.total_size_bytes += size_bytes

    def _record_delete(self, count: int = 1):
        """Record cache delete operation."""
        with self._metrics_lock:
            self.metrics.deletes += count

    def _record_error(self):
        """Record cache error."""
        with self._metrics_lock:
            self.metrics.errors += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache metrics."""
        with self._metrics_lock:
            total_requests = self.metrics.hits + self.metrics.misses
            hit_rate = self.metrics.hits / total_requests if total_requests > 0 else 0.0

            return {
                "available": self.is_available(),
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "sets": self.metrics.sets,
                "deletes": self.metrics.deletes,
                "errors": self.metrics.errors,
                "total_size_mb": self.metrics.total_size_bytes / (1024 * 1024),
                "avg_retrieval_time_ms": self.metrics.avg_retrieval_time_ms,
                "connection_status": self.connection_manager.circuit_breaker.state,
                "compression": self.serializer.compression.value,
                "namespace": self.namespace,
                "version": self.version,
            }

    def get_info(self) -> dict[str, Any]:
        """Get Redis server information."""
        if not self.is_available():
            return {"error": "Redis not available"}

        try:
            with self.connection_manager.get_connection() as conn:
                info = conn.info()

                # Extract relevant information
                return {
                    "redis_version": info.get("redis_version"),
                    "used_memory_human": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "role": info.get("role"),
                }

        except Exception as e:
            logger.error("Error getting Redis info: %s", e)
            return {"error": str(e)}

    def flush_all(self) -> bool:
        """Flush all cache entries (use with caution)."""
        if not self.is_available():
            return False

        try:
            with self.connection_manager.get_connection() as conn:
                conn.flushdb()

            logger.warning("All cache entries flushed")
            return True

        except Exception as e:
            logger.error("Error flushing cache: %s", e)
            return False


# Global cache instance
redis_cache = RedisCache()
