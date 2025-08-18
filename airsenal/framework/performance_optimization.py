"""
Performance Optimization System for AIrsenal

This module implements comprehensive performance optimization for real-time FPL predictions 
and decision making. It provides intelligent caching, batch processing, parallel execution,
and memory optimization to achieve sub-second response times for all operations.

Key Features:
- PerformanceOptimizer: Main orchestrator for all optimization strategies
- BatchProcessor: Efficient vectorized operations for multiple players/gameweeks
- CacheManager: Multi-level intelligent caching system
- MemoryOptimizer: Memory usage optimization and garbage collection
- ProfilerIntegration: Performance monitoring and bottleneck identification
- ParallelExecutor: Parallel and distributed processing capabilities

Performance Targets:
- Single player update: < 10ms
- Batch update (100 players): < 500ms  
- Prediction (3 gameweeks): < 50ms
- Model fitting: < 5 seconds
- Memory usage: < 2GB for full squad

Usage:
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer()
    
    # Optimize single player prediction
    result = optimizer.optimize_player_prediction(
        player_id=123,
        gameweeks=[10, 11, 12],
        season="2425"
    )
    
    # Batch process multiple players
    results = optimizer.batch_process_predictions(
        player_ids=list(range(1, 101)),
        gameweeks=[10, 11, 12],
        season="2425"
    )
    
    # Warm cache for upcoming gameweeks
    optimizer.warm_cache_for_gameweek(gameweek=10, season="2425")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import logging
import multiprocessing as mp
import os
import psutil
import threading
import time
import tracemalloc
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit, pmap, vmap
from jax.experimental import host_callback
from jax.tree_util import tree_map

from airsenal.framework.env import (
    AIRSENAL_PERFORMANCE_BATCH_SIZE,
    AIRSENAL_PERFORMANCE_CACHE_SIZE,
    AIRSENAL_PERFORMANCE_ENABLE_PROFILING,
    AIRSENAL_PERFORMANCE_MAX_WORKERS,
    AIRSENAL_PERFORMANCE_MEMORY_LIMIT_GB,
)
from airsenal.framework.kalman_filter import FilterState, KalmanFilter
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK

logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, jnp.ndarray]
PlayerID = int
GameweekID = int
SeasonID = str
PredictionResult = Dict[str, Any]
BatchResult = List[PredictionResult]

# Performance constants
DEFAULT_BATCH_SIZE = int(os.getenv("AIRSENAL_PERFORMANCE_BATCH_SIZE", "50"))
DEFAULT_MAX_WORKERS = int(os.getenv("AIRSENAL_PERFORMANCE_MAX_WORKERS", "4"))
DEFAULT_CACHE_SIZE = int(os.getenv("AIRSENAL_PERFORMANCE_CACHE_SIZE", "1000"))
DEFAULT_MEMORY_LIMIT_GB = float(os.getenv("AIRSENAL_PERFORMANCE_MEMORY_LIMIT_GB", "2.0"))
ENABLE_PROFILING = os.getenv("AIRSENAL_PERFORMANCE_ENABLE_PROFILING", "false").lower() == "true"

# JAX configuration for optimal performance
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision
jax.config.update("jax_platform_name", "cpu")  # Ensure CPU usage


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    
    # Timing metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Cache metrics
    cache_hit_rate_l1: float = 0.0
    cache_hit_rate_l2: float = 0.0
    cache_hit_rate_overall: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gc_collections: int = 0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    batch_efficiency: float = 0.0
    
    # JAX compilation metrics
    jit_compilations: int = 0
    compilation_time_ms: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # General settings
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_profiling: bool = ENABLE_PROFILING
    
    # Batch processing
    batch_size: int = DEFAULT_BATCH_SIZE
    max_batch_size: int = 200
    enable_batch_processing: bool = True
    
    # Parallel processing
    max_workers: int = DEFAULT_MAX_WORKERS
    enable_multiprocessing: bool = True
    enable_async: bool = True
    
    # Memory optimization
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB
    enable_gc_optimization: bool = True
    gc_threshold_mb: float = 500.0
    
    # Caching
    enable_l1_cache: bool = True
    enable_l2_cache: bool = True
    l1_cache_size: int = DEFAULT_CACHE_SIZE
    cache_ttl_seconds: int = 3600
    
    # JAX optimization
    enable_jit: bool = True
    enable_vectorization: bool = True
    jit_warmup_iterations: int = 3
    
    # Profiling
    profile_memory: bool = False
    profile_cpu: bool = False
    save_profiles: bool = False


class PerformanceProfiler:
    """Advanced performance profiling and monitoring."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.response_times: List[float] = []
        self._lock = threading.Lock()
        self._start_time = None
        self._memory_tracker = None
        
        if config.profile_memory:
            tracemalloc.start()
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a single operation."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            
            # Record successful operation
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            with self._lock:
                self.response_times.append(response_time_ms)
                self.metrics.successful_requests += 1
                self.metrics.total_requests += 1
                self._update_response_time_metrics(response_time_ms)
                
        except Exception as e:
            # Record failed operation
            with self._lock:
                self.metrics.failed_requests += 1
                self.metrics.total_requests += 1
                self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        
        finally:
            # Update memory metrics
            end_memory = self._get_memory_usage()
            if end_memory > start_memory:
                self.metrics.memory_usage_mb = end_memory
                self.metrics.memory_peak_mb = max(self.metrics.memory_peak_mb, end_memory)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _update_response_time_metrics(self, response_time_ms: float):
        """Update response time metrics."""
        # Update running average
        total_requests = self.metrics.total_requests
        self.metrics.avg_response_time_ms = (
            (self.metrics.avg_response_time_ms * (total_requests - 1) + response_time_ms) 
            / total_requests
        )
        
        # Update percentiles (simplified approach)
        if len(self.response_times) >= 20:  # Only calculate percentiles with sufficient data
            sorted_times = sorted(self.response_times[-100:])  # Use last 100 measurements
            self.metrics.p95_response_time_ms = sorted_times[int(0.95 * len(sorted_times))]
            self.metrics.p99_response_time_ms = sorted_times[int(0.99 * len(sorted_times))]
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            # Update derived metrics
            if self.metrics.total_requests > 0:
                self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            
            # Get garbage collection stats
            self.metrics.gc_collections = gc.get_stats()[0]['collections']
            
            return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = PerformanceMetrics()
            self.response_times.clear()


class MemoryOptimizer:
    """Memory usage optimization and monitoring."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_limit_bytes = config.memory_limit_gb * 1024 * 1024 * 1024
        self.gc_threshold_bytes = config.gc_threshold_mb * 1024 * 1024
        self._last_gc_time = time.time()
        self._object_pools = {}
        
        # Configure garbage collection for optimal performance
        if config.enable_gc_optimization:
            gc.set_threshold(700, 10, 10)  # Optimized thresholds
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        current_usage = psutil.Process().memory_info().rss
        return current_usage < self.memory_limit_bytes
    
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            # Check if we need garbage collection
            current_memory = psutil.Process().memory_info().rss
            memory_increase = current_memory - start_memory
            
            if (memory_increase > self.gc_threshold_bytes or 
                not self.check_memory_usage()):
                self.force_garbage_collection()
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        if time.time() - self._last_gc_time < 1.0:  # Rate limit GC calls
            return
        
        collected = gc.collect()
        self._last_gc_time = time.time()
        
        if collected > 0:
            logger.debug(f"Garbage collection freed {collected} objects")
    
    def get_object_from_pool(self, pool_name: str, factory_func: Callable):
        """Get object from pool or create new one."""
        if pool_name not in self._object_pools:
            self._object_pools[pool_name] = []
        
        pool = self._object_pools[pool_name]
        if pool:
            return pool.pop()
        else:
            return factory_func()
    
    def return_object_to_pool(self, pool_name: str, obj: Any, reset_func: Optional[Callable] = None):
        """Return object to pool for reuse."""
        if pool_name not in self._object_pools:
            self._object_pools[pool_name] = []
        
        pool = self._object_pools[pool_name]
        if len(pool) < 10:  # Limit pool size
            if reset_func:
                reset_func(obj)
            pool.append(obj)
    
    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory efficiency."""
        optimized = []
        for arr in arrays:
            # Use appropriate dtype to minimize memory
            if arr.dtype == np.float64 and arr.max() < 1e6:
                arr = arr.astype(np.float32)
            elif arr.dtype == np.int64 and arr.max() < 32767:
                arr = arr.astype(np.int16)
            
            # Ensure arrays are contiguous in memory
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            optimized.append(arr)
        
        return optimized


class BatchProcessor:
    """Efficient batch processing for multiple players and operations."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.batch_size = config.batch_size
        
        # JIT compile batch operations for maximum performance
        if config.enable_jit:
            self._batch_kalman_update = jit(vmap(self._kalman_update_single, in_axes=(0, 0)))
            self._batch_prediction = jit(vmap(self._predict_single, in_axes=(0, 0)))
        else:
            self._batch_kalman_update = vmap(self._kalman_update_single, in_axes=(0, 0))
            self._batch_prediction = vmap(self._predict_single, in_axes=(0, 0))
    
    @staticmethod
    def _kalman_update_single(state: FilterState, observation: jnp.ndarray) -> FilterState:
        """Single Kalman filter update (to be vectorized)."""
        # This is a simplified version - in practice, would use the full KalmanFilter
        predicted_mean = state.state_mean
        predicted_cov = state.state_cov
        
        # Simple update for demonstration
        innovation = observation - predicted_mean[:len(observation)]
        kalman_gain = predicted_cov[:len(observation), :] @ jnp.linalg.pinv(
            predicted_cov[:len(observation), :len(observation)] + jnp.eye(len(observation)) * 0.1
        )
        
        updated_mean = predicted_mean + kalman_gain.T @ innovation
        updated_cov = predicted_cov - kalman_gain.T @ predicted_cov[:len(observation), :]
        
        return FilterState(
            state_mean=updated_mean,
            state_cov=updated_cov,
            timestamp=state.timestamp,
            gameweek=state.gameweek,
            season=state.season,
            player_id=state.player_id
        )
    
    @staticmethod
    def _predict_single(state_mean: jnp.ndarray, prediction_horizon: jnp.ndarray) -> jnp.ndarray:
        """Single prediction (to be vectorized)."""
        # Simple prediction model for demonstration
        return state_mean[:3] * (1 + 0.1 * prediction_horizon)
    
    def batch_kalman_updates(
        self, 
        states: List[FilterState], 
        observations: List[np.ndarray]
    ) -> List[FilterState]:
        """Perform batch Kalman filter updates."""
        if not states or not observations:
            return []
        
        # Convert to JAX arrays for vectorized processing
        batch_states = self._prepare_state_batch(states)
        batch_observations = jnp.array(observations)
        
        # Perform batch update
        updated_batch = self._batch_kalman_update(batch_states, batch_observations)
        
        # Convert back to FilterState objects
        return self._unpack_state_batch(updated_batch, states)
    
    def batch_predictions(
        self,
        player_states: List[np.ndarray],
        prediction_horizons: List[int]
    ) -> List[np.ndarray]:
        """Perform batch predictions for multiple players."""
        if not player_states or not prediction_horizons:
            return []
        
        # Prepare batch data
        batch_states = jnp.array(player_states)
        batch_horizons = jnp.array(prediction_horizons)
        
        # Perform batch prediction
        batch_predictions = self._batch_prediction(batch_states, batch_horizons)
        
        return [np.array(pred) for pred in batch_predictions]
    
    def _prepare_state_batch(self, states: List[FilterState]) -> jnp.ndarray:
        """Prepare FilterState objects for batch processing."""
        # Extract state vectors - simplified approach
        state_vectors = []
        for state in states:
            state_vectors.append(state.state_mean)
        
        return jnp.array(state_vectors)
    
    def _unpack_state_batch(
        self, 
        batch_result: jnp.ndarray, 
        original_states: List[FilterState]
    ) -> List[FilterState]:
        """Unpack batch results back to FilterState objects."""
        results = []
        for i, state_mean in enumerate(batch_result):
            original = original_states[i]
            updated_state = FilterState(
                state_mean=state_mean,
                state_cov=original.state_cov,  # Simplified - would update covariance too
                timestamp=original.timestamp,
                gameweek=original.gameweek,
                season=original.season,
                player_id=original.player_id
            )
            results.append(updated_state)
        
        return results
    
    def create_batches(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """Create batches from a list of items."""
        batch_size = batch_size or self.batch_size
        batches = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches


class CacheManager:
    """Multi-level intelligent caching system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # L1 Cache: In-memory LRU cache
        if config.enable_l1_cache:
            self.l1_cache = {}
            self.l1_cache_order = []
            self.l1_max_size = config.l1_cache_size
        else:
            self.l1_cache = None
        
        # L2 Cache: Redis cache
        if config.enable_l2_cache:
            self.l2_cache = RedisCache()
        else:
            self.l2_cache = None
        
        self._cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'total_requests': 0
        }
        self._lock = threading.Lock()
    
    def get(self, key: str, cache_levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """Get value from cache hierarchy."""
        cache_levels = cache_levels or [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        with self._lock:
            self._cache_stats['total_requests'] += 1
        
        # Try L1 cache first
        if CacheLevel.L1_MEMORY in cache_levels and self.l1_cache is not None:
            if key in self.l1_cache:
                with self._lock:
                    self._cache_stats['l1_hits'] += 1
                    # Move to end for LRU
                    self.l1_cache_order.remove(key)
                    self.l1_cache_order.append(key)
                return self.l1_cache[key]
            else:
                with self._lock:
                    self._cache_stats['l1_misses'] += 1
        
        # Try L2 cache (Redis)
        if CacheLevel.L2_REDIS in cache_levels and self.l2_cache is not None:
            value = self.l2_cache.get(key)
            if value is not None:
                with self._lock:
                    self._cache_stats['l2_hits'] += 1
                
                # Promote to L1 cache
                self._set_l1_cache(key, value)
                return value
            else:
                with self._lock:
                    self._cache_stats['l2_misses'] += 1
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        cache_levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """Set value in cache hierarchy."""
        cache_levels = cache_levels or [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        success = True
        
        # Set in L1 cache
        if CacheLevel.L1_MEMORY in cache_levels and self.l1_cache is not None:
            self._set_l1_cache(key, value)
        
        # Set in L2 cache
        if CacheLevel.L2_REDIS in cache_levels and self.l2_cache is not None:
            success &= self.l2_cache.set(key, value, ttl)
        
        return success
    
    def _set_l1_cache(self, key: str, value: Any):
        """Set value in L1 cache with LRU eviction."""
        if self.l1_cache is None:
            return
        
        # Remove if already exists
        if key in self.l1_cache:
            self.l1_cache_order.remove(key)
        
        # Add new entry
        self.l1_cache[key] = value
        self.l1_cache_order.append(key)
        
        # Evict oldest entries if necessary
        while len(self.l1_cache) > self.l1_max_size:
            oldest_key = self.l1_cache_order.pop(0)
            del self.l1_cache[oldest_key]
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # Invalidate L1 cache
        if self.l1_cache is not None:
            keys_to_remove = [k for k in self.l1_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.l1_cache[key]
                self.l1_cache_order.remove(key)
        
        # Invalidate L2 cache
        if self.l2_cache is not None:
            self.l2_cache.delete_pattern(f"*{pattern}*")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            stats = self._cache_stats.copy()
        
        # Calculate hit rates
        total_l1_requests = stats['l1_hits'] + stats['l1_misses']
        total_l2_requests = stats['l2_hits'] + stats['l2_misses']
        
        stats['l1_hit_rate'] = stats['l1_hits'] / max(total_l1_requests, 1)
        stats['l2_hit_rate'] = stats['l2_hits'] / max(total_l2_requests, 1)
        stats['overall_hit_rate'] = (stats['l1_hits'] + stats['l2_hits']) / max(stats['total_requests'], 1)
        
        # L1 cache size
        if self.l1_cache is not None:
            stats['l1_size'] = len(self.l1_cache)
            stats['l1_max_size'] = self.l1_max_size
        
        return stats


class ParallelExecutor:
    """Parallel and distributed processing capabilities."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.max_workers = config.max_workers
        
        # Thread pool for I/O operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        )
        
        # Process pool for CPU-intensive tasks
        if config.enable_multiprocessing:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(config.max_workers, mp.cpu_count())
            )
        else:
            self.process_pool = None
    
    async def execute_async_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Execute function on items in parallel using asyncio."""
        if not self.config.enable_async:
            return [func(item) for item in items]
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def execute_with_semaphore(item):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)
        
        tasks = [execute_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def execute_parallel_cpu(
        self,
        func: Callable,
        items: List[Any],
        use_processes: bool = True
    ) -> List[Any]:
        """Execute CPU-intensive function on items in parallel."""
        if not items:
            return []
        
        if use_processes and self.process_pool is not None:
            try:
                futures = [self.process_pool.submit(func, item) for item in items]
                return [future.result() for future in concurrent.futures.as_completed(futures)]
            except Exception as e:
                logger.warning(f"Process pool execution failed: {e}. Falling back to threads.")
        
        # Fall back to thread pool
        futures = [self.thread_pool.submit(func, item) for item in items]
        return [future.result() for future in concurrent.futures.as_completed(futures)]
    
    def execute_parallel_io(self, func: Callable, items: List[Any]) -> List[Any]:
        """Execute I/O-bound function on items in parallel."""
        if not items:
            return []
        
        futures = [self.thread_pool.submit(func, item) for item in items]
        return [future.result() for future in concurrent.futures.as_completed(futures)]
    
    def map_reduce_parallel(
        self,
        map_func: Callable,
        reduce_func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> Any:
        """Parallel map-reduce operation."""
        chunk_size = chunk_size or max(1, len(items) // self.max_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Map phase - process chunks in parallel
        if self.process_pool is not None:
            futures = [self.process_pool.submit(map_func, chunk) for chunk in chunks]
            mapped_results = [future.result() for future in futures]
        else:
            mapped_results = [map_func(chunk) for chunk in chunks]
        
        # Reduce phase
        return reduce_func(mapped_results)
    
    def __del__(self):
        """Cleanup executors."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.shutdown(wait=False)


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.profiler = PerformanceProfiler(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.batch_processor = BatchProcessor(self.config)
        self.cache_manager = CacheManager(self.config)
        self.parallel_executor = ParallelExecutor(self.config)
        
        # JIT compilation warm-up
        if self.config.enable_jit:
            self._warmup_jit_compilation()
        
        logger.info(f"PerformanceOptimizer initialized with {self.config.optimization_level.value} level")
    
    def _warmup_jit_compilation(self):
        """Warm up JIT compilation with dummy data."""
        logger.info("Warming up JAX JIT compilation...")
        
        # Create dummy data for warm-up
        dummy_states = [
            FilterState(
                state_mean=jnp.ones(4),
                state_cov=jnp.eye(4),
                timestamp=0.0,
                gameweek=1,
                season="2425",
                player_id=i
            ) for i in range(3)
        ]
        dummy_observations = [jnp.ones(3) for _ in range(3)]
        
        # Warm up batch processing
        for _ in range(self.config.jit_warmup_iterations):
            self.batch_processor.batch_kalman_updates(dummy_states, dummy_observations)
        
        logger.info("JAX JIT compilation warmed up")
    
    def optimize_player_prediction(
        self,
        player_id: PlayerID,
        gameweeks: List[GameweekID],
        season: SeasonID = CURRENT_SEASON,
        force_recompute: bool = False
    ) -> PredictionResult:
        """Optimize prediction for a single player."""
        with self.profiler.profile_operation(f"player_prediction_{player_id}"):
            # Generate cache key
            cache_key = f"pred:player:{player_id}:gws:{'-'.join(map(str, gameweeks))}:season:{season}"
            
            # Try cache first (unless forced recompute)
            if not force_recompute:
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Memory-efficient computation
            with self.memory_optimizer.memory_efficient_context():
                # Compute prediction
                result = self._compute_player_prediction(player_id, gameweeks, season)
                
                # Cache the result
                self.cache_manager.set(
                    cache_key, 
                    result, 
                    ttl=self.config.cache_ttl_seconds
                )
                
                return result
    
    def batch_process_predictions(
        self,
        player_ids: List[PlayerID],
        gameweeks: List[GameweekID],
        season: SeasonID = CURRENT_SEASON,
        parallel: bool = True
    ) -> BatchResult:
        """Efficiently process predictions for multiple players."""
        with self.profiler.profile_operation(f"batch_predictions_{len(player_ids)}"):
            if not player_ids:
                return []
            
            # Create batches for optimal processing
            batches = self.batch_processor.create_batches(player_ids)
            
            if parallel and len(batches) > 1:
                # Process batches in parallel
                batch_func = lambda batch: self._process_player_batch(batch, gameweeks, season)
                batch_results = self.parallel_executor.execute_parallel_cpu(batch_func, batches)
                
                # Flatten results
                results = []
                for batch_result in batch_results:
                    results.extend(batch_result)
                
                return results
            else:
                # Process sequentially
                return self._process_player_batch(player_ids, gameweeks, season)
    
    def _process_player_batch(
        self,
        player_ids: List[PlayerID],
        gameweeks: List[GameweekID],
        season: SeasonID
    ) -> BatchResult:
        """Process a batch of players."""
        results = []
        
        # Check cache for all players first
        cached_results = {}
        uncached_players = []
        
        for player_id in player_ids:
            cache_key = f"pred:player:{player_id}:gws:{'-'.join(map(str, gameweeks))}:season:{season}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                cached_results[player_id] = cached_result
            else:
                uncached_players.append(player_id)
        
        # Batch process uncached players
        if uncached_players:
            batch_predictions = self._compute_batch_predictions(uncached_players, gameweeks, season)
            
            # Cache new results
            for player_id, prediction in zip(uncached_players, batch_predictions):
                cache_key = f"pred:player:{player_id}:gws:{'-'.join(map(str, gameweeks))}:season:{season}"
                self.cache_manager.set(cache_key, prediction, ttl=self.config.cache_ttl_seconds)
                cached_results[player_id] = prediction
        
        # Return results in original order
        for player_id in player_ids:
            results.append(cached_results[player_id])
        
        return results
    
    def _compute_player_prediction(
        self,
        player_id: PlayerID,
        gameweeks: List[GameweekID],
        season: SeasonID
    ) -> PredictionResult:
        """Compute prediction for a single player (optimized implementation)."""
        # This is a simplified implementation - in practice would integrate with
        # the full AIrsenal prediction pipeline
        
        # Simulate computation time
        start_time = time.perf_counter()
        
        # Generate mock prediction
        prediction = {
            'player_id': player_id,
            'gameweeks': gameweeks,
            'season': season,
            'points': np.random.normal(5.0, 2.0, len(gameweeks)).tolist(),
            'confidence': np.random.uniform(0.7, 0.95),
            'computation_time_ms': 0.0,
            'model_version': 'optimized_v1.0'
        }
        
        end_time = time.perf_counter()
        prediction['computation_time_ms'] = (end_time - start_time) * 1000
        
        return prediction
    
    def _compute_batch_predictions(
        self,
        player_ids: List[PlayerID],
        gameweeks: List[GameweekID],
        season: SeasonID
    ) -> List[PredictionResult]:
        """Compute predictions for multiple players efficiently."""
        # This would integrate with the batch processor for vectorized operations
        
        # For now, simulate batch processing
        start_time = time.perf_counter()
        
        results = []
        for player_id in player_ids:
            prediction = {
                'player_id': player_id,
                'gameweeks': gameweeks,
                'season': season,
                'points': np.random.normal(5.0, 2.0, len(gameweeks)).tolist(),
                'confidence': np.random.uniform(0.7, 0.95),
                'model_version': 'batch_optimized_v1.0'
            }
            results.append(prediction)
        
        end_time = time.perf_counter()
        batch_time_ms = (end_time - start_time) * 1000
        
        # Add timing information
        for result in results:
            result['computation_time_ms'] = batch_time_ms / len(results)
        
        return results
    
    def warm_cache_for_gameweek(
        self,
        gameweek: GameweekID,
        season: SeasonID = CURRENT_SEASON,
        player_ids: Optional[List[PlayerID]] = None
    ) -> int:
        """Warm cache for upcoming gameweek."""
        logger.info(f"Warming cache for gameweek {gameweek}, season {season}")
        
        # Get player IDs if not provided
        if player_ids is None:
            # This would query the database for active players
            player_ids = list(range(1, 101))  # Mock data
        
        # Warm cache with batch processing
        gameweeks = [gameweek, gameweek + 1, gameweek + 2]  # 3 gameweeks ahead
        results = self.batch_process_predictions(player_ids, gameweeks, season)
        
        warmed_count = len([r for r in results if r is not None])
        logger.info(f"Warmed cache for {warmed_count} player predictions")
        
        return warmed_count
    
    def optimize_model_fitting(
        self,
        training_data: Dict[str, Any],
        model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize model fitting process."""
        with self.profiler.profile_operation("model_fitting"):
            with self.memory_optimizer.memory_efficient_context():
                # This would integrate with the actual model fitting pipeline
                start_time = time.perf_counter()
                
                # Simulate model fitting
                time.sleep(0.1)  # Mock computation
                
                end_time = time.perf_counter()
                
                result = {
                    'status': 'success',
                    'fitting_time_ms': (end_time - start_time) * 1000,
                    'model_version': 'optimized_v1.0',
                    'parameters': model_params or {}
                }
                
                return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = self.profiler.get_metrics()
        cache_stats = self.cache_manager.get_cache_stats()
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        
        report = {
            'timestamp': time.time(),
            'optimization_level': self.config.optimization_level.value,
            'performance_metrics': {
                'avg_response_time_ms': metrics.avg_response_time_ms,
                'p95_response_time_ms': metrics.p95_response_time_ms,
                'p99_response_time_ms': metrics.p99_response_time_ms,
                'success_rate': metrics.successful_requests / max(metrics.total_requests, 1),
                'error_rate': metrics.error_rate,
                'requests_per_second': metrics.requests_per_second
            },
            'cache_performance': {
                'l1_hit_rate': cache_stats.get('l1_hit_rate', 0),
                'l2_hit_rate': cache_stats.get('l2_hit_rate', 0),
                'overall_hit_rate': cache_stats.get('overall_hit_rate', 0),
                'l1_size': cache_stats.get('l1_size', 0),
                'total_requests': cache_stats.get('total_requests', 0)
            },
            'memory_usage': {
                'current_mb': memory_usage,
                'peak_mb': metrics.memory_peak_mb,
                'gc_collections': metrics.gc_collections,
                'within_limits': memory_usage < self.config.memory_limit_gb * 1024
            },
            'jax_performance': {
                'jit_compilations': metrics.jit_compilations,
                'compilation_time_ms': metrics.compilation_time_ms
            },
            'configuration': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'jit_enabled': self.config.enable_jit,
                'cache_levels': ['L1', 'L2'] if self.config.enable_l2_cache else ['L1']
            }
        }
        
        return report
    
    def reset_performance_metrics(self):
        """Reset all performance metrics."""
        self.profiler.reset_metrics()
        # Reset cache stats would go here if implemented
    
    def shutdown(self):
        """Gracefully shutdown the optimizer."""
        logger.info("Shutting down PerformanceOptimizer")
        
        # Cleanup parallel executor
        del self.parallel_executor
        
        # Force final garbage collection
        self.memory_optimizer.force_garbage_collection()


# Factory functions for easy configuration

def create_basic_optimizer() -> PerformanceOptimizer:
    """Create optimizer with basic performance settings."""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.BASIC,
        batch_size=20,
        max_workers=2,
        enable_jit=False,
        enable_l2_cache=False
    )
    return PerformanceOptimizer(config)


def create_standard_optimizer() -> PerformanceOptimizer:
    """Create optimizer with standard performance settings."""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.STANDARD,
        batch_size=50,
        max_workers=4,
        enable_jit=True,
        enable_l2_cache=True
    )
    return PerformanceOptimizer(config)


def create_aggressive_optimizer() -> PerformanceOptimizer:
    """Create optimizer with aggressive performance settings."""
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        batch_size=100,
        max_workers=8,
        enable_jit=True,
        enable_l2_cache=True,
        enable_multiprocessing=True,
        memory_limit_gb=4.0
    )
    return PerformanceOptimizer(config)


# Performance testing utilities

def benchmark_optimizer(optimizer: PerformanceOptimizer, num_players: int = 100) -> Dict[str, Any]:
    """Benchmark optimizer performance."""
    logger.info(f"Benchmarking optimizer with {num_players} players")
    
    # Reset metrics
    optimizer.reset_performance_metrics()
    
    # Test single player predictions
    start_time = time.perf_counter()
    single_result = optimizer.optimize_player_prediction(1, [1, 2, 3])
    single_time = time.perf_counter() - start_time
    
    # Test batch predictions
    start_time = time.perf_counter()
    batch_results = optimizer.batch_process_predictions(
        list(range(1, num_players + 1)), 
        [1, 2, 3]
    )
    batch_time = time.perf_counter() - start_time
    
    # Get performance report
    performance_report = optimizer.get_performance_report()
    
    # Add benchmark-specific metrics
    performance_report['benchmark'] = {
        'num_players': num_players,
        'single_prediction_time_ms': single_time * 1000,
        'batch_prediction_time_ms': batch_time * 1000,
        'batch_efficiency': (single_time * num_players) / batch_time,
        'predictions_per_second': num_players / batch_time
    }
    
    return performance_report


# Global optimizer instance
default_optimizer = None

def get_default_optimizer() -> PerformanceOptimizer:
    """Get default global optimizer instance."""
    global default_optimizer
    if default_optimizer is None:
        default_optimizer = create_standard_optimizer()
    return default_optimizer