"""
Redis Cache Performance Benchmarks

Tests performance of the Redis caching layer including:
- Set/get operation latency
- Batch operations performance
- Cache hit rates and miss penalties
- Memory usage and eviction behavior
- Connection pooling efficiency
"""

import pytest
import json
import time
import numpy as np
from typing import List, Dict, Any

from airsenal.framework.redis_cache import RedisCache
from benchmarks.utils import BenchmarkRunner, generate_test_data


class RedisCacheBenchmarks:
    """Redis cache performance benchmark suite."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.redis_cache = runner.redis_cache
        self.config = runner.config
        
        # Test data
        self.test_keys = [f"benchmark:key:{i}" for i in range(1000)]
        self.test_values = [
            {
                "data": np.random.uniform(0, 100, 50).tolist(),
                "metadata": {"source": "benchmark", "id": i}
            }
            for i in range(1000)
        ]
    
    def run_all(self):
        """Run all Redis cache benchmarks."""
        if not self.redis_cache:
            print("Redis cache not available - skipping benchmarks")
            return
        
        self.benchmark_single_operations()
        self.benchmark_batch_operations()
        self.benchmark_cache_hit_rates()
        self.benchmark_serialization()
        self.benchmark_connection_pooling()
        self.benchmark_ttl_performance()
        self.benchmark_memory_usage()
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_single_operations(self):
        """Benchmark single set/get operations."""
        # Benchmark SET operations
        with self.runner.benchmark_context("redis_set_single"):
            for i in range(100):
                key = self.test_keys[i]
                value = self.test_values[i]
                self.redis_cache.set(key, value, ttl=3600)
        
        # Benchmark GET operations
        with self.runner.benchmark_context("redis_get_single"):
            for i in range(100):
                key = self.test_keys[i]
                result = self.redis_cache.get(key)
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_batch_operations(self):
        """Benchmark batch set/get operations."""
        for batch_size in self.config.sample_sizes:
            if batch_size > len(self.test_keys):
                continue
            
            # Benchmark batch SET
            batch_data = {
                self.test_keys[i]: self.test_values[i]
                for i in range(batch_size)
            }
            
            with self.runner.benchmark_context(f"redis_set_batch_{batch_size}"):
                self.redis_cache.set_many(batch_data, ttl=3600)
            
            # Benchmark batch GET
            with self.runner.benchmark_context(f"redis_get_batch_{batch_size}"):
                results = self.redis_cache.get_many(self.test_keys[:batch_size])
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_cache_hit_rates(self):
        """Benchmark cache hit rates and miss penalties."""
        # Populate cache
        for i in range(50):
            self.redis_cache.set(self.test_keys[i], self.test_values[i], ttl=3600)
        
        # Test cache hits
        hit_times = []
        with self.runner.benchmark_context("redis_cache_hits"):
            for i in range(50):
                start = time.perf_counter()
                result = self.redis_cache.get(self.test_keys[i])
                hit_times.append(time.perf_counter() - start)
        
        # Test cache misses
        miss_times = []
        with self.runner.benchmark_context("redis_cache_misses"):
            for i in range(50, 100):
                start = time.perf_counter()
                result = self.redis_cache.get(self.test_keys[i])  # Not in cache
                miss_times.append(time.perf_counter() - start)
        
        # Calculate hit rate and penalties
        avg_hit_time = np.mean(hit_times)
        avg_miss_time = np.mean(miss_times)
        miss_penalty = avg_miss_time / avg_hit_time if avg_hit_time > 0 else 0
        
        # Log metrics
        self.runner.results[-1]["avg_hit_time"] = avg_hit_time
        self.runner.results[-2]["avg_miss_time"] = avg_miss_time
        self.runner.results[-2]["miss_penalty_ratio"] = miss_penalty
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_serialization(self):
        """Benchmark serialization/deserialization performance."""
        # Test different data types and sizes
        test_data = {
            "small_dict": {"id": 123, "value": 45.6},
            "large_dict": {f"key_{i}": np.random.uniform(0, 100) for i in range(1000)},
            "numpy_array": np.random.uniform(0, 100, 1000),
            "list_of_dicts": [{"id": i, "value": np.random.uniform(0, 100)} for i in range(100)]
        }
        
        for data_type, data in test_data.items():
            # Benchmark serialization (SET)
            with self.runner.benchmark_context(f"redis_serialize_{data_type}"):
                self.redis_cache.set(f"serialize_test:{data_type}", data, ttl=3600)
            
            # Benchmark deserialization (GET)
            with self.runner.benchmark_context(f"redis_deserialize_{data_type}"):
                result = self.redis_cache.get(f"serialize_test:{data_type}")
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_connection_pooling(self):
        """Benchmark connection pooling efficiency."""
        # Test multiple rapid connections
        with self.runner.benchmark_context("redis_connection_pool"):
            for i in range(100):
                # Simulate rapid operations that require connections
                self.redis_cache.set(f"pool_test:{i}", {"value": i}, ttl=60)
                result = self.redis_cache.get(f"pool_test:{i}")
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_ttl_performance(self):
        """Benchmark TTL-related operations."""
        # Test setting keys with TTL
        with self.runner.benchmark_context("redis_set_with_ttl"):
            for i in range(100):
                self.redis_cache.set(
                    f"ttl_test:{i}",
                    self.test_values[i],
                    ttl=3600
                )
        
        # Test TTL queries
        with self.runner.benchmark_context("redis_ttl_check"):
            for i in range(100):
                ttl = self.redis_cache.ttl(f"ttl_test:{i}")
        
        # Test expiration behavior
        with self.runner.benchmark_context("redis_short_ttl"):
            for i in range(50):
                self.redis_cache.set(f"expire_test:{i}", {"value": i}, ttl=1)
        
        # Wait for expiration and test access
        time.sleep(2)
        with self.runner.benchmark_context("redis_expired_access"):
            for i in range(50):
                result = self.redis_cache.get(f"expire_test:{i}")
    
    @pytest.mark.benchmark(group="redis_cache")
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        import psutil
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with self.runner.benchmark_context("redis_memory_usage"):
            # Store large amounts of data
            large_data = {
                f"memory_test:{i}": {
                    "large_array": np.random.uniform(0, 100, 1000).tolist(),
                    "metadata": {"id": i, "timestamp": time.time()}
                }
                for i in range(100)
            }
            
            self.redis_cache.set_many(large_data, ttl=3600)
            
            # Retrieve all data
            keys = list(large_data.keys())
            results = self.redis_cache.get_many(keys)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - start_memory
        
        # Log memory usage
        self.runner.results[-1]["memory_used_mb"] = memory_used
        self.runner.results[-1]["peak_memory_mb"] = peak_memory
    
    def benchmark_prediction_cache_pattern(self):
        """Benchmark typical prediction caching patterns."""
        # Simulate player prediction caching
        player_predictions = {
            f"predictions:player:{i}:gw:10": {
                "points": np.random.uniform(0, 15),
                "minutes": np.random.uniform(0, 90),
                "timestamp": time.time()
            }
            for i in range(100)
        }
        
        with self.runner.benchmark_context("redis_prediction_cache"):
            # Cache predictions
            self.redis_cache.set_many(player_predictions, ttl=1800)  # 30 minutes
            
            # Simulate batch retrieval for optimization
            prediction_keys = list(player_predictions.keys())
            cached_predictions = self.redis_cache.get_many(prediction_keys)
    
    def stress_test_concurrent_access(self):
        """Stress test concurrent Redis access."""
        import threading
        import concurrent.futures
        
        def redis_operations(thread_id: int):
            """Perform Redis operations in separate thread."""
            try:
                # Mixed read/write operations
                for i in range(10):
                    key = f"stress:{thread_id}:{i}"
                    
                    # Set operation
                    self.redis_cache.set(key, {"thread": thread_id, "op": i}, ttl=60)
                    
                    # Get operation
                    result = self.redis_cache.get(key)
                    
                    if result is None:
                        return 0
                
                return 1
            except Exception:
                return 0
        
        with self.runner.benchmark_context("redis_concurrent_stress"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(redis_operations, i)
                    for i in range(100)
                ]
                
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
        
        # Log success rate
        success_rate = sum(results) / len(results)
        self.runner.results[-1]["success_rate"] = success_rate


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="redis_single")
def test_redis_single_set(benchmark):
    """Pytest-benchmark test for single Redis SET operation."""
    redis_cache = RedisCache()
    test_data = {"id": 123, "value": 45.6, "metadata": {"source": "benchmark"}}
    
    def redis_set():
        return redis_cache.set("benchmark:single:set", test_data, ttl=3600)
    
    result = benchmark(redis_set)


@pytest.mark.benchmark(group="redis_single")
def test_redis_single_get(benchmark):
    """Pytest-benchmark test for single Redis GET operation."""
    redis_cache = RedisCache()
    test_data = {"id": 123, "value": 45.6, "metadata": {"source": "benchmark"}}
    
    # Setup: ensure key exists
    redis_cache.set("benchmark:single:get", test_data, ttl=3600)
    
    def redis_get():
        return redis_cache.get("benchmark:single:get")
    
    result = benchmark(redis_get)


@pytest.mark.benchmark(group="redis_batch")
def test_redis_batch_operations(benchmark):
    """Pytest-benchmark test for Redis batch operations."""
    redis_cache = RedisCache()
    
    test_data = {
        f"benchmark:batch:{i}": {
            "id": i,
            "value": np.random.uniform(0, 100),
            "metadata": {"source": "benchmark"}
        }
        for i in range(100)
    }
    
    def redis_batch():
        # Set batch
        redis_cache.set_many(test_data, ttl=3600)
        
        # Get batch
        keys = list(test_data.keys())
        return redis_cache.get_many(keys)
    
    result = benchmark(redis_batch)