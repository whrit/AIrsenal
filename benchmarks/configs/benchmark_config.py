"""
Benchmark Configuration for AIrsenal Performance Testing

This module contains configuration settings for all performance benchmarks
including thresholds, test parameters, and environment settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Base configuration
BENCHMARK_ROOT = Path(__file__).parent.parent
REPORTS_DIR = BENCHMARK_ROOT / "reports"
BASELINE_DIR = BENCHMARK_ROOT / "configs" / "baselines"


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration for regression detection."""

    # Feature Store thresholds (in seconds)
    feature_get_single: float = 0.001  # 1ms
    feature_get_batch_100: float = 0.010  # 10ms
    feature_compute_rolling: float = 0.100  # 100ms
    feature_store_memory_mb: float = 50.0  # 50MB

    # Redis Cache thresholds
    redis_get_single: float = 0.0005  # 0.5ms
    redis_set_single: float = 0.001  # 1ms
    redis_get_batch_100: float = 0.005  # 5ms
    redis_hit_rate_min: float = 0.85  # 85% hit rate minimum

    # Model Versioning thresholds
    model_load_time: float = 2.0  # 2 seconds
    model_serialize_time: float = 1.0  # 1 second
    model_artifact_size_mb: float = 100.0  # 100MB

    # Database Operation thresholds
    db_query_simple: float = 0.010  # 10ms
    db_query_complex: float = 0.100  # 100ms
    db_bulk_insert_1000: float = 1.0  # 1 second

    # Prediction Pipeline thresholds
    prediction_single_player: float = 0.050  # 50ms
    prediction_batch_100: float = 2.0  # 2 seconds
    prediction_memory_mb: float = 200.0  # 200MB

    # Base Models thresholds
    adaptive_model_update: float = 0.100  # 100ms
    form_calculation: float = 0.050  # 50ms
    availability_prediction: float = 0.020  # 20ms


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    # Test data configuration
    sample_sizes: list[int] = None
    test_seasons: list[str] = None
    test_gameweeks: list[int] = None

    # Performance thresholds
    thresholds: PerformanceThresholds = None

    # Benchmark execution settings
    warm_cache: bool = True
    clear_cache_between_tests: bool = True
    measure_memory: bool = True
    profile_cpu: bool = False  # CPU profiling (slower)

    # Load testing configuration
    load_test_users: int = 10
    load_test_spawn_rate: float = 1.0
    load_test_duration: int = 60  # seconds

    # Reporting configuration
    save_detailed_results: bool = True
    generate_plots: bool = True
    compare_to_baseline: bool = True

    def __post_init__(self):
        if self.sample_sizes is None:
            self.sample_sizes = [1, 10, 50, 100, 500]

        if self.test_seasons is None:
            self.test_seasons = ["2324", "2425"]

        if self.test_gameweeks is None:
            self.test_gameweeks = list(range(1, 11))  # First 10 gameweeks

        if self.thresholds is None:
            self.thresholds = PerformanceThresholds()


# Default configuration instance
DEFAULT_CONFIG = BenchmarkConfig()

# Environment-specific configurations
CONFIGS = {
    "ci": BenchmarkConfig(
        sample_sizes=[1, 10, 50],
        test_gameweeks=list(range(1, 6)),  # Fewer gameweeks for CI
        profile_cpu=False,
        generate_plots=False,
        load_test_users=5,
        load_test_duration=30,
    ),
    "local": BenchmarkConfig(
        sample_sizes=[1, 10, 50, 100, 500, 1000],
        profile_cpu=True,
        generate_plots=True,
        load_test_users=20,
        load_test_duration=120,
    ),
    "stress": BenchmarkConfig(
        sample_sizes=[100, 500, 1000, 5000],
        load_test_users=100,
        load_test_spawn_rate=10.0,
        load_test_duration=300,  # 5 minutes
        clear_cache_between_tests=False,  # Test cache effectiveness
    ),
}


def get_config(env: str | None = None) -> BenchmarkConfig:
    """Get benchmark configuration for specified environment."""
    if env is None:
        env = os.getenv("BENCHMARK_ENV", "local")

    return CONFIGS.get(env, DEFAULT_CONFIG)


def get_test_data_config() -> dict[str, Any]:
    """Get test data configuration for benchmarks."""
    return {
        "player_ids": [123, 456, 789, 1011, 1213],  # Sample player IDs
        "team_ids": [1, 2, 3, 4, 5],  # Sample team IDs
        "feature_names": [
            "rolling_goals_5",
            "rolling_assists_5",
            "xg_form",
            "minutes_trend",
            "difficulty_adjusted_points",
        ],
        "model_types": ["adaptive", "conjugate", "numpyro"],
        "cache_keys": [
            "predictions:player:123:gw:10",
            "features:rolling_goals:player:456",
            "model:performance:numpyro:v1.2.0",
        ],
    }
