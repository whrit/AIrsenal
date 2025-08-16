"""
Feature Store Performance Benchmarks

Tests performance of the FeatureStore component including:
- Feature retrieval latency
- Batch feature computation
- Cache hit rates and miss penalties
- Memory usage patterns
- Rolling window calculations
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any

from airsenal.framework.feature_store import FeatureStore
from benchmarks.utils import BenchmarkRunner, generate_test_data


class FeatureStoreBenchmarks:
    """Feature Store performance benchmark suite."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.feature_store = runner.feature_store
        self.config = runner.config
        
        # Test data
        self.player_ids = list(range(1, 101))  # 100 test players
        self.feature_names = [
            "rolling_goals_5",
            "rolling_assists_5",
            "xg_form",
            "minutes_trend",
            "difficulty_adjusted_points"
        ]
    
    def run_all(self):
        """Run all feature store benchmarks."""
        if not self.feature_store:
            print("Feature store not available - skipping benchmarks")
            return
        
        self.benchmark_single_feature_retrieval()
        self.benchmark_batch_feature_retrieval()
        self.benchmark_feature_computation()
        self.benchmark_rolling_window_calculations()
        self.benchmark_cache_performance()
        self.benchmark_memory_usage()
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_single_feature_retrieval(self):
        """Benchmark single feature retrieval performance."""
        with self.runner.benchmark_context("feature_get_single"):
            for _ in range(100):
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=[self.player_ids[0]],
                    feature_names=[self.feature_names[0]],
                    season="2425",
                    gameweek=10
                )
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_batch_feature_retrieval(self):
        """Benchmark batch feature retrieval performance."""
        for batch_size in self.config.sample_sizes:
            if batch_size > len(self.player_ids):
                continue
                
            with self.runner.benchmark_context(f"feature_get_batch_{batch_size}"):
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=self.player_ids[:batch_size],
                    feature_names=self.feature_names,
                    season="2425",
                    gameweek=10
                )
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_feature_computation(self):
        """Benchmark feature computation performance."""
        with self.runner.benchmark_context("feature_compute_rolling"):
            # Test rolling window computation
            self.feature_store.compute_features_batch(
                feature_names=["rolling_goals_5"],
                season="2425",
                gameweek_range=(1, 10)
            )
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_rolling_window_calculations(self):
        """Benchmark rolling window calculation performance."""
        window_sizes = [3, 5, 10, 20]
        
        for window_size in window_sizes:
            with self.runner.benchmark_context(f"rolling_window_{window_size}"):
                # Simulate rolling calculation
                data = np.random.uniform(0, 10, 100)
                
                # Manual rolling calculation (benchmark target)
                rolling_means = []
                for i in range(window_size, len(data)):
                    window_data = data[i-window_size:i]
                    rolling_means.append(np.mean(window_data))
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_cache_performance(self):
        """Benchmark cache hit rates and miss penalties."""
        # Warm cache
        features = self.feature_store.get_features(
            entity_type="player",
            entity_ids=self.player_ids[:10],
            feature_names=[self.feature_names[0]],
            season="2425",
            gameweek=10
        )
        
        # Test cache hits
        with self.runner.benchmark_context("feature_cache_hit"):
            for _ in range(50):
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=self.player_ids[:10],
                    feature_names=[self.feature_names[0]],
                    season="2425",
                    gameweek=10
                )
        
        # Clear cache and test misses
        if hasattr(self.feature_store, 'clear_cache'):
            self.feature_store.clear_cache()
        
        with self.runner.benchmark_context("feature_cache_miss"):
            for i in range(10):
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=[self.player_ids[i]],
                    feature_names=[self.feature_names[0]],
                    season="2425",
                    gameweek=10 + i  # Different gameweek to avoid cache
                )
    
    @pytest.mark.benchmark(group="feature_store")
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        import psutil
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with self.runner.benchmark_context("feature_memory_usage"):
            # Load large batch of features
            features = self.feature_store.get_features(
                entity_type="player",
                entity_ids=self.player_ids,
                feature_names=self.feature_names,
                season="2425",
                gameweek=10
            )
            
            # Simulate feature processing
            if features:
                processed_features = {}
                for player_id, player_features in features.items():
                    processed_features[player_id] = {
                        name: value * 2 for name, value in player_features.items()
                    }
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - start_memory
        
        # Log memory usage
        self.runner.results[-1]["memory_used_mb"] = memory_used
        self.runner.results[-1]["peak_memory_mb"] = peak_memory
    
    def benchmark_feature_versioning(self):
        """Benchmark feature versioning performance."""
        # Test feature schema registration
        with self.runner.benchmark_context("feature_version_register"):
            for i in range(10):
                try:
                    self.feature_store.register_feature(
                        name=f"test_feature_{i}",
                        feature_type="player",
                        computation_logic={
                            "window": 5,
                            "metric": "test_metric",
                            "agg": "mean"
                        },
                        version=f"v1.{i}.0"
                    )
                except Exception:
                    pass  # Feature might already exist
        
        # Test feature evolution
        with self.runner.benchmark_context("feature_version_evolve"):
            try:
                self.feature_store.evolve_feature(
                    name="test_feature_0",
                    new_computation_logic={
                        "window": 10,
                        "metric": "test_metric",
                        "agg": "mean"
                    }
                )
            except Exception:
                pass
    
    def stress_test_concurrent_access(self):
        """Stress test concurrent feature access."""
        import threading
        import concurrent.futures
        
        def fetch_features(thread_id: int):
            """Fetch features in separate thread."""
            try:
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=[self.player_ids[thread_id % len(self.player_ids)]],
                    feature_names=[self.feature_names[0]],
                    season="2425",
                    gameweek=10 + thread_id
                )
                return len(features) if features else 0
            except Exception as e:
                return 0
        
        with self.runner.benchmark_context("feature_concurrent_access"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(fetch_features, i)
                    for i in range(50)
                ]
                
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
        
        # Log success rate
        success_rate = sum(1 for r in results if r > 0) / len(results)
        self.runner.results[-1]["success_rate"] = success_rate


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="feature_store_single")
def test_feature_store_single_retrieval(benchmark):
    """Pytest-benchmark test for single feature retrieval."""
    feature_store = FeatureStore()
    
    def get_single_feature():
        return feature_store.get_features(
            entity_type="player",
            entity_ids=[123],
            feature_names=["rolling_goals_5"],
            season="2425",
            gameweek=10
        )
    
    result = benchmark(get_single_feature)


@pytest.mark.benchmark(group="feature_store_batch")
def test_feature_store_batch_retrieval(benchmark):
    """Pytest-benchmark test for batch feature retrieval."""
    feature_store = FeatureStore()
    player_ids = list(range(1, 101))
    feature_names = ["rolling_goals_5", "xg_form"]
    
    def get_batch_features():
        return feature_store.get_features(
            entity_type="player",
            entity_ids=player_ids,
            feature_names=feature_names,
            season="2425",
            gameweek=10
        )
    
    result = benchmark(get_batch_features)


@pytest.mark.benchmark(group="feature_store_compute") 
def test_feature_store_computation(benchmark):
    """Pytest-benchmark test for feature computation."""
    feature_store = FeatureStore()
    
    def compute_features():
        return feature_store.compute_features_batch(
            feature_names=["rolling_goals_5"],
            season="2425",
            gameweek_range=(1, 5)
        )
    
    result = benchmark(compute_features)