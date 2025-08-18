"""
Benchmark Utilities for AIrsenal Performance Testing

Common utilities and helper functions for performance benchmarking.
"""

import gc
import json
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from airsenal.framework.feature_store import FeatureStore
from airsenal.framework.model_versioning import ModelVersionManager
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import session

from .configs.benchmark_config import BenchmarkConfig, get_config


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""

    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()

    def stop_monitoring(self) -> dict[str, float]:
        """Stop monitoring and return measurements."""
        end_time = time.perf_counter()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()

        return {
            "duration": end_time - self.start_time,
            "memory_used_mb": end_memory - self.start_memory,
            "peak_memory_mb": end_memory,
            "cpu_percent": end_cpu,
        }

    @contextmanager
    def measure(self):
        """Context manager for measuring performance."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.measurements.append(self.stop_monitoring())


class BenchmarkRunner:
    """Main benchmark runner with common functionality."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or get_config()
        self.results = []
        self.monitor = PerformanceMonitor()

        # Initialize components
        self.session = session
        self.redis_cache = None
        self.feature_store = None
        self.model_manager = None

    def setup(self):
        """Set up benchmark environment."""
        # Initialize Redis cache if available
        try:
            self.redis_cache = RedisCache()
            self.redis_cache.ping()
        except Exception:
            print("Redis not available - cache benchmarks will be skipped")
            self.redis_cache = None

        # Initialize feature store
        try:
            self.feature_store = FeatureStore()
        except Exception as e:
            print(f"Feature store initialization failed: {e}")
            self.feature_store = None

        # Initialize model manager
        try:
            self.model_manager = ModelVersionManager()
        except Exception as e:
            print(f"Model manager initialization failed: {e}")
            self.model_manager = None

    def teardown(self):
        """Clean up after benchmarks."""
        if self.session:
            self.session.close()

        if self.redis_cache and self.config.clear_cache_between_tests:
            with suppress(Exception):
                self.redis_cache.clear_all()

    def warm_cache(self, warm_functions: list[Callable] | None = None):
        """Warm up caches before benchmarking."""
        if not self.config.warm_cache:
            return

        print("Warming up caches...")
        if warm_functions:
            for func in warm_functions:
                try:
                    func()
                except Exception as e:
                    print(f"Cache warming failed for {func.__name__}: {e}")

    @contextmanager
    def benchmark_context(self, test_name: str):
        """Context manager for individual benchmark tests."""
        print(f"Running benchmark: {test_name}")

        # Clear memory before test
        if self.config.clear_cache_between_tests:
            gc.collect()

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            result = {
                "test_name": test_name,
                "duration": end_time - start_time,
                "memory_delta_mb": end_memory - start_memory,
                "timestamp": datetime.now().isoformat(),
            }

            self.results.append(result)
            print(
                f"  Duration: {result['duration']:.4f}s, Memory: {result['memory_delta_mb']:.2f}MB"
            )

    def save_results(self, filename: str | None = None):
        """Save benchmark results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        results_path = Path("benchmarks/reports") / filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "config": asdict(self.config),
            "results": self.results,
            "summary": self.get_summary(),
        }

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {results_path}")
        return results_path

    def get_summary(self) -> dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        return {
            "total_tests": len(self.results),
            "total_duration": df["duration"].sum(),
            "avg_duration": df["duration"].mean(),
            "max_duration": df["duration"].max(),
            "min_duration": df["duration"].min(),
            "avg_memory_mb": df["memory_delta_mb"].mean(),
            "max_memory_mb": df["memory_delta_mb"].max(),
            "failed_tests": len([r for r in self.results if r.get("error")]),
        }


def generate_test_data(
    data_type: str, size: int, season: str = "2425", gameweek: int = 10
) -> list[dict] | dict[str, Any]:
    """Generate test data for benchmarks."""

    if data_type == "player_predictions":
        return [
            {
                "player_id": i,
                "gameweek": gameweek,
                "season": season,
                "predicted_points": np.random.uniform(0, 15),
                "predicted_minutes": np.random.uniform(0, 90),
                "prediction_confidence": np.random.uniform(0.5, 1.0),
            }
            for i in range(1, size + 1)
        ]

    if data_type == "feature_matrix":
        features = ["rolling_goals_5", "xg_form", "minutes_trend"]
        return {
            "player_ids": list(range(1, size + 1)),
            "features": {
                feature: np.random.uniform(0, 10, size) for feature in features
            },
            "metadata": {
                "season": season,
                "gameweek": gameweek,
                "feature_version": "v1.0.0",
            },
        }

    if data_type == "cache_entries":
        return [
            {
                "key": f"test:key:{i}",
                "value": {
                    "data": np.random.uniform(0, 100, 50).tolist(),
                    "timestamp": time.time(),
                    "metadata": {"source": "benchmark"},
                },
                "ttl": 3600,
            }
            for i in range(size)
        ]

    msg = f"Unknown data type: {data_type}"
    raise ValueError(msg)


def compare_to_baseline(
    results: list[dict], baseline_file: str | None = None
) -> dict[str, Any]:
    """Compare benchmark results to baseline."""
    if not baseline_file:
        baseline_file = "benchmarks/configs/baselines/baseline.json"

    baseline_path = Path(baseline_file)
    if not baseline_path.exists():
        print(f"No baseline found at {baseline_path}")
        return {"status": "no_baseline"}

    with open(baseline_path) as f:
        baseline = json.load(f)

    comparison = {
        "status": "compared",
        "regressions": [],
        "improvements": [],
        "summary": {},
    }

    # Compare key metrics
    current_df = pd.DataFrame(results)
    baseline_df = pd.DataFrame(baseline.get("results", []))

    if baseline_df.empty:
        return {"status": "invalid_baseline"}

    # Group by test name and compare
    for test_name in current_df["test_name"].unique():
        current_test = current_df[current_df["test_name"] == test_name]
        baseline_test = baseline_df[baseline_df["test_name"] == test_name]

        if baseline_test.empty:
            continue

        current_avg = current_test["duration"].mean()
        baseline_avg = baseline_test["duration"].mean()

        change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100

        result = {
            "test_name": test_name,
            "current_duration": current_avg,
            "baseline_duration": baseline_avg,
            "change_percent": change_percent,
        }

        if change_percent > 10:  # 10% regression threshold
            comparison["regressions"].append(result)
        elif change_percent < -10:  # 10% improvement threshold
            comparison["improvements"].append(result)

    comparison["summary"] = {
        "total_regressions": len(comparison["regressions"]),
        "total_improvements": len(comparison["improvements"]),
        "avg_change_percent": current_df["duration"].mean()
        / baseline_df["duration"].mean()
        * 100
        - 100,
    }

    return comparison


def check_thresholds(results: list[dict], config: BenchmarkConfig) -> dict[str, Any]:
    """Check if benchmark results exceed performance thresholds."""
    violations = []

    for result in results:
        test_name = result["test_name"]
        duration = result["duration"]

        # Map test names to thresholds
        threshold_map = {
            "feature_get_single": config.thresholds.feature_get_single,
            "feature_get_batch": config.thresholds.feature_get_batch_100,
            "redis_get_single": config.thresholds.redis_get_single,
            "redis_set_single": config.thresholds.redis_set_single,
            "model_load": config.thresholds.model_load_time,
            "prediction_single": config.thresholds.prediction_single_player,
        }

        # Check for threshold violations
        for pattern, threshold in threshold_map.items():
            if pattern in test_name and duration > threshold:
                violations.append(
                    {
                        "test_name": test_name,
                        "actual": duration,
                        "threshold": threshold,
                        "violation_percent": ((duration - threshold) / threshold) * 100,
                    }
                )

    return {
        "violations": violations,
        "total_violations": len(violations),
        "pass": len(violations) == 0,
    }
