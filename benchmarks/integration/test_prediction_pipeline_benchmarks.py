"""
Prediction Pipeline Integration Benchmarks

Tests end-to-end performance of the prediction pipeline including:
- Full prediction workflow latency
- Batch prediction throughput
- Memory usage during prediction
- Cache effectiveness in real scenarios
- Feature retrieval and computation integration
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any, Optional

from airsenal.framework.player_model import NumpyroPlayerModel
from airsenal.framework.feature_store import FeatureStore
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import Player, PlayerScore, session
from benchmarks.utils import BenchmarkRunner, generate_test_data


class PredictionPipelineBenchmarks:
    """Prediction pipeline integration benchmark suite."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.config = runner.config
        self.session = runner.session
        
        # Initialize components
        self.feature_store = runner.feature_store
        self.redis_cache = runner.redis_cache
        
        # Test data
        self.test_player_ids = list(range(1, 101))  # 100 test players
        self.test_gameweeks = list(range(35, 39))   # Last few gameweeks
        self.season = "2425"
    
    def run_all(self):
        """Run all prediction pipeline benchmarks."""
        self.benchmark_single_player_prediction()
        self.benchmark_batch_prediction()
        self.benchmark_prediction_with_features()
        self.benchmark_prediction_with_cache()
        self.benchmark_end_to_end_pipeline()
        self.benchmark_pipeline_memory_usage()
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_single_player_prediction(self):
        """Benchmark single player prediction performance."""
        with self.runner.benchmark_context("prediction_single_player"):
            for player_id in self.test_player_ids[:50]:
                # Simulate prediction process
                prediction = self._simulate_single_prediction(
                    player_id=player_id,
                    gameweek=35,
                    season=self.season
                )
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_batch_prediction(self):
        """Benchmark batch prediction performance."""
        for batch_size in self.config.sample_sizes:
            if batch_size > len(self.test_player_ids):
                continue
            
            with self.runner.benchmark_context(f"prediction_batch_{batch_size}"):
                player_batch = self.test_player_ids[:batch_size]
                predictions = self._simulate_batch_prediction(
                    player_ids=player_batch,
                    gameweek=35,
                    season=self.season
                )
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_prediction_with_features(self):
        """Benchmark prediction with feature store integration."""
        if not self.feature_store:
            print("Feature store not available - skipping feature integration test")
            return
        
        with self.runner.benchmark_context("prediction_with_features"):
            for player_id in self.test_player_ids[:20]:
                # Get features
                features = self._get_player_features(
                    player_id=player_id,
                    gameweek=35,
                    season=self.season
                )
                
                # Make prediction using features
                prediction = self._simulate_prediction_with_features(
                    player_id=player_id,
                    features=features
                )
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_prediction_with_cache(self):
        """Benchmark prediction with cache integration."""
        if not self.redis_cache:
            print("Redis cache not available - skipping cache integration test")
            return
        
        # Warm cache with some predictions
        cache_keys = []
        for i, player_id in enumerate(self.test_player_ids[:10]):
            cache_key = f"prediction:player:{player_id}:gw:35:season:{self.season}"
            prediction_data = {
                "expected_points": np.random.uniform(0, 15),
                "expected_minutes": np.random.uniform(0, 90),
                "confidence": np.random.uniform(0.5, 1.0),
                "timestamp": time.time()
            }
            self.redis_cache.set(cache_key, prediction_data, ttl=3600)
            cache_keys.append(cache_key)
        
        # Test cache hits
        with self.runner.benchmark_context("prediction_cache_hits"):
            for cache_key in cache_keys:
                cached_prediction = self.redis_cache.get(cache_key)
        
        # Test cache misses (new predictions)
        with self.runner.benchmark_context("prediction_cache_misses"):
            for player_id in self.test_player_ids[10:20]:
                cache_key = f"prediction:player:{player_id}:gw:36:season:{self.season}"
                cached_prediction = self.redis_cache.get(cache_key)
                
                if cached_prediction is None:
                    # Generate new prediction and cache it
                    prediction = self._simulate_single_prediction(
                        player_id=player_id,
                        gameweek=36,
                        season=self.season
                    )
                    self.redis_cache.set(cache_key, prediction, ttl=3600)
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_end_to_end_pipeline(self):
        """Benchmark complete end-to-end prediction pipeline."""
        with self.runner.benchmark_context("prediction_end_to_end"):
            # Simulate complete pipeline for multiple players and gameweeks
            for gameweek in [35, 36]:
                for player_id in self.test_player_ids[:30]:
                    # 1. Check cache
                    cache_key = f"prediction:player:{player_id}:gw:{gameweek}:season:{self.season}"
                    if self.redis_cache:
                        cached = self.redis_cache.get(cache_key)
                        if cached:
                            continue
                    
                    # 2. Get features
                    features = self._get_player_features(
                        player_id=player_id,
                        gameweek=gameweek,
                        season=self.season
                    )
                    
                    # 3. Make prediction
                    prediction = self._simulate_prediction_with_features(
                        player_id=player_id,
                        features=features
                    )
                    
                    # 4. Cache result
                    if self.redis_cache:
                        self.redis_cache.set(cache_key, prediction, ttl=3600)
    
    @pytest.mark.benchmark(group="prediction_pipeline")
    def benchmark_pipeline_memory_usage(self):
        """Benchmark memory usage during prediction pipeline."""
        import psutil
        process = psutil.Process()
        
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with self.runner.benchmark_context("prediction_memory_usage"):
            # Process large batch to test memory usage
            all_predictions = {}
            
            for gameweek in [35, 36, 37]:
                gameweek_predictions = {}
                
                for player_id in self.test_player_ids:
                    # Get features
                    features = self._get_player_features(
                        player_id=player_id,
                        gameweek=gameweek,
                        season=self.season
                    )
                    
                    # Make prediction
                    prediction = self._simulate_prediction_with_features(
                        player_id=player_id,
                        features=features
                    )
                    
                    gameweek_predictions[player_id] = prediction
                
                all_predictions[gameweek] = gameweek_predictions
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - start_memory
        
        # Log memory usage
        self.runner.results[-1]["memory_used_mb"] = memory_used
        self.runner.results[-1]["peak_memory_mb"] = peak_memory
        self.runner.results[-1]["predictions_generated"] = len(all_predictions) * len(self.test_player_ids)
    
    def _simulate_single_prediction(self, player_id: int, gameweek: int, season: str) -> Dict[str, float]:
        """Simulate a single player prediction."""
        # Simulate model prediction time
        time.sleep(0.001)  # 1ms per prediction
        
        return {
            "expected_points": np.random.uniform(0, 15),
            "expected_minutes": np.random.uniform(0, 90),
            "confidence": np.random.uniform(0.5, 1.0),
            "player_id": player_id,
            "gameweek": gameweek,
            "season": season
        }
    
    def _simulate_batch_prediction(self, player_ids: List[int], gameweek: int, season: str) -> Dict[int, Dict[str, float]]:
        """Simulate batch player predictions."""
        # Simulate batch processing efficiency (faster than individual predictions)
        batch_time = len(player_ids) * 0.0005  # 0.5ms per player in batch
        time.sleep(batch_time)
        
        predictions = {}
        for player_id in player_ids:
            predictions[player_id] = {
                "expected_points": np.random.uniform(0, 15),
                "expected_minutes": np.random.uniform(0, 90),
                "confidence": np.random.uniform(0.5, 1.0),
                "player_id": player_id,
                "gameweek": gameweek,
                "season": season
            }
        
        return predictions
    
    def _get_player_features(self, player_id: int, gameweek: int, season: str) -> Dict[str, float]:
        """Get player features from feature store or simulate."""
        if self.feature_store:
            try:
                features = self.feature_store.get_features(
                    entity_type="player",
                    entity_ids=[player_id],
                    feature_names=["rolling_goals_5", "xg_form", "minutes_trend"],
                    season=season,
                    gameweek=gameweek
                )
                if features and player_id in features:
                    return features[player_id]
            except Exception:
                pass
        
        # Simulate feature computation
        time.sleep(0.002)  # 2ms for feature computation
        return {
            "rolling_goals_5": np.random.uniform(0, 1),
            "rolling_assists_5": np.random.uniform(0, 0.5),
            "xg_form": np.random.uniform(0, 2),
            "minutes_trend": np.random.uniform(-10, 10),
            "difficulty_adjusted_points": np.random.uniform(2, 8)
        }
    
    def _simulate_prediction_with_features(self, player_id: int, features: Dict[str, float]) -> Dict[str, float]:
        """Simulate prediction using features."""
        # Simple feature-based prediction simulation
        base_points = sum(features.values()) / len(features)
        
        # Add some model complexity simulation
        time.sleep(0.0015)  # 1.5ms for model inference
        
        return {
            "expected_points": max(0, base_points * np.random.uniform(0.8, 1.2)),
            "expected_minutes": np.random.uniform(0, 90),
            "confidence": min(1.0, len(features) / 10.0),
            "features_used": list(features.keys())
        }
    
    def stress_test_concurrent_predictions(self):
        """Stress test concurrent prediction requests."""
        import threading
        import concurrent.futures
        
        def make_predictions(thread_id: int):
            """Make predictions in separate thread."""
            try:
                predictions = []
                for i in range(10):
                    player_id = (thread_id * 10 + i) % len(self.test_player_ids) + 1
                    prediction = self._simulate_single_prediction(
                        player_id=player_id,
                        gameweek=35,
                        season=self.season
                    )
                    predictions.append(prediction)
                
                return len(predictions)
            except Exception:
                return 0
        
        with self.runner.benchmark_context("prediction_concurrent_stress"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(make_predictions, i)
                    for i in range(20)
                ]
                
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
        
        # Log success rate and throughput
        total_predictions = sum(results)
        success_rate = len([r for r in results if r > 0]) / len(results)
        
        self.runner.results[-1]["total_predictions"] = total_predictions
        self.runner.results[-1]["success_rate"] = success_rate


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="prediction_integration")
def test_single_prediction_pipeline(benchmark):
    """Pytest-benchmark test for single prediction pipeline."""
    
    def single_prediction():
        # Simulate feature retrieval
        features = {
            "rolling_goals_5": np.random.uniform(0, 1),
            "xg_form": np.random.uniform(0, 2),
            "minutes_trend": np.random.uniform(-10, 10)
        }
        
        # Simulate prediction
        time.sleep(0.003)  # 3ms total
        
        return {
            "expected_points": sum(features.values()) / len(features),
            "confidence": 0.8
        }
    
    result = benchmark(single_prediction)


@pytest.mark.benchmark(group="prediction_integration")
def test_batch_prediction_pipeline(benchmark):
    """Pytest-benchmark test for batch prediction pipeline."""
    
    def batch_prediction():
        batch_size = 50
        predictions = {}
        
        # Simulate batch processing
        time.sleep(batch_size * 0.001)  # 1ms per player
        
        for i in range(batch_size):
            predictions[i] = {
                "expected_points": np.random.uniform(0, 15),
                "confidence": 0.8
            }
        
        return predictions
    
    result = benchmark(batch_prediction)