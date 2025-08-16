"""
Base Models Performance Benchmarks

Tests performance of the base model interfaces including:
- Adaptive model update performance
- Form calculation latency
- Availability prediction speed
- Feature engineering pipeline throughput
- Model state serialization
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional

from airsenal.framework.base_models import (
    AdaptivePlayerModel,
    AvailabilityPredictor, 
    FormCalculator,
    FeatureEngineer
)
from benchmarks.utils import BenchmarkRunner, generate_test_data


class MockAdaptiveModel(AdaptivePlayerModel):
    """Mock implementation for benchmarking."""
    
    def __init__(self):
        self.state = {"weights": np.random.normal(0, 1, 10), "bias": 0.0}
    
    def predict(self, features: np.ndarray, player_data: Dict[str, Any]) -> Dict[str, float]:
        # Simple linear prediction
        if len(features) >= 10:
            prediction = np.dot(self.state["weights"], features[:10]) + self.state["bias"]
        else:
            prediction = np.mean(features) * 2.0
        
        return {
            "expected_points": max(0, prediction),
            "confidence": min(1.0, abs(prediction) / 10.0)
        }
    
    def update_model(self, new_data: List[Dict[str, Any]], learning_rate: float = 0.01) -> None:
        # Simple gradient update simulation
        for data_point in new_data:
            if "actual_points" in data_point and "features" in data_point:
                features = np.array(data_point["features"][:10])
                if len(features) == 10:
                    error = data_point["actual_points"] - np.dot(self.state["weights"], features)
                    self.state["weights"] += learning_rate * error * features
                    self.state["bias"] += learning_rate * error
    
    def get_model_state(self) -> Dict[str, Any]:
        return self.state.copy()
    
    def set_model_state(self, state: Dict[str, Any]) -> None:
        self.state = state.copy()


class MockAvailabilityPredictor(AvailabilityPredictor):
    """Mock implementation for benchmarking."""
    
    def predict_availability(self, player_data: Dict[str, Any], gameweek: int) -> Dict[str, float]:
        # Simulate availability prediction based on injury history
        injury_history = player_data.get("injury_history", [])
        fitness_score = player_data.get("fitness_score", 0.8)
        
        # Simple heuristic
        injury_risk = len(injury_history) * 0.1
        availability_prob = max(0.1, min(0.95, fitness_score - injury_risk))
        
        return {
            "availability_probability": availability_prob,
            "injury_risk": injury_risk,
            "fitness_score": fitness_score
        }
    
    def update_injury_data(self, player_id: int, injury_data: Dict[str, Any]) -> None:
        # Simulate updating injury model
        time.sleep(0.001)  # Simulate processing time


class MockFormCalculator(FormCalculator):
    """Mock implementation for benchmarking."""
    
    def calculate_form(self, recent_performances: List[Dict[str, Any]], 
                      window_size: int = 5) -> Dict[str, float]:
        if not recent_performances:
            return {"form_score": 0.0, "trend": 0.0, "consistency": 0.0}
        
        # Take last window_size performances
        recent = recent_performances[-window_size:]
        
        points = [p.get("points", 0) for p in recent]
        minutes = [p.get("minutes", 0) for p in recent]
        
        # Calculate form metrics
        form_score = np.mean(points) if points else 0.0
        trend = np.polyfit(range(len(points)), points, 1)[0] if len(points) > 1 else 0.0
        consistency = 1.0 / (1.0 + np.std(points)) if len(points) > 1 else 0.0
        
        return {
            "form_score": form_score,
            "trend": trend,
            "consistency": consistency,
            "games_played": len([p for p in recent if p.get("minutes", 0) > 0])
        }
    
    def calculate_weighted_form(self, performances: List[Dict[str, Any]], 
                              weights: Optional[List[float]] = None) -> float:
        if not performances:
            return 0.0
        
        points = [p.get("points", 0) for p in performances]
        
        if weights is None:
            # Exponential decay weights (more recent = higher weight)
            weights = [0.5 ** i for i in range(len(points))][::-1]
        
        if len(weights) != len(points):
            weights = weights[:len(points)]
        
        weighted_sum = sum(p * w for p, w in zip(points, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


class MockFeatureEngineer(FeatureEngineer):
    """Mock implementation for benchmarking."""
    
    def extract_features(self, player_data: Dict[str, Any]) -> np.ndarray:
        # Extract basic features
        features = []
        
        # Basic stats
        features.append(player_data.get("goals", 0))
        features.append(player_data.get("assists", 0))
        features.append(player_data.get("minutes", 0))
        features.append(player_data.get("clean_sheets", 0))
        features.append(player_data.get("yellow_cards", 0))
        
        # Derived features
        goals = player_data.get("goals", 0)
        games = player_data.get("games_played", 1)
        features.append(goals / max(1, games))  # Goals per game
        
        minutes = player_data.get("minutes", 0)
        features.append(minutes / max(1, games))  # Minutes per game
        
        # Form features (simplified)
        recent_points = player_data.get("recent_points", [0, 0, 0, 0, 0])
        features.append(np.mean(recent_points))  # Average recent points
        features.append(np.std(recent_points))   # Consistency
        
        # Position encoding
        position = player_data.get("position", "MID")
        position_encoding = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        features.append(position_encoding.get(position, 2))
        
        return np.array(features, dtype=float)
    
    def create_feature_pipeline(self, transformations: List[str]) -> List:
        # Simple feature pipeline simulation
        pipeline = []
        
        for transform in transformations:
            if transform == "normalize":
                pipeline.append(lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8))
            elif transform == "scale":
                pipeline.append(lambda x: x / (np.max(np.abs(x)) + 1e-8))
            elif transform == "log":
                pipeline.append(lambda x: np.log1p(np.abs(x)))
        
        return pipeline


class BaseModelsBenchmarks:
    """Base models performance benchmark suite."""
    
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.config = runner.config
        
        # Initialize mock models
        self.adaptive_model = MockAdaptiveModel()
        self.availability_predictor = MockAvailabilityPredictor()
        self.form_calculator = MockFormCalculator()
        self.feature_engineer = MockFeatureEngineer()
        
        # Test data
        self.test_player_data = self._generate_test_player_data()
        self.test_performances = self._generate_test_performances()
    
    def _generate_test_player_data(self) -> List[Dict[str, Any]]:
        """Generate test player data."""
        return [
            {
                "player_id": i,
                "position": np.random.choice(["GK", "DEF", "MID", "FWD"]),
                "goals": np.random.poisson(5),
                "assists": np.random.poisson(3),
                "minutes": np.random.randint(500, 3000),
                "clean_sheets": np.random.poisson(2),
                "yellow_cards": np.random.poisson(1),
                "games_played": np.random.randint(10, 38),
                "recent_points": np.random.uniform(0, 15, 5).tolist(),
                "injury_history": [{"type": "minor", "duration": 2}] if np.random.random() < 0.3 else [],
                "fitness_score": np.random.uniform(0.5, 1.0),
                "features": np.random.uniform(0, 10, 15).tolist()
            }
            for i in range(100)
        ]
    
    def _generate_test_performances(self) -> List[Dict[str, Any]]:
        """Generate test performance data."""
        return [
            {
                "gameweek": i,
                "points": np.random.uniform(0, 15),
                "minutes": np.random.randint(0, 90),
                "goals": np.random.poisson(0.2),
                "assists": np.random.poisson(0.1)
            }
            for i in range(1, 21)  # 20 gameweeks
        ]
    
    def run_all(self):
        """Run all base model benchmarks."""
        self.benchmark_adaptive_model_operations()
        self.benchmark_availability_prediction()
        self.benchmark_form_calculation()
        self.benchmark_feature_engineering()
        self.benchmark_model_state_operations()
    
    @pytest.mark.benchmark(group="base_models")
    def benchmark_adaptive_model_operations(self):
        """Benchmark adaptive model operations."""
        # Benchmark single prediction
        with self.runner.benchmark_context("adaptive_model_predict_single"):
            for player_data in self.test_player_data[:50]:
                features = np.array(player_data["features"])
                prediction = self.adaptive_model.predict(features, player_data)
        
        # Benchmark batch prediction
        with self.runner.benchmark_context("adaptive_model_predict_batch"):
            for player_data in self.test_player_data:
                features = np.array(player_data["features"])
                prediction = self.adaptive_model.predict(features, player_data)
        
        # Benchmark model update
        update_data = [
            {
                "features": player["features"],
                "actual_points": np.random.uniform(0, 15)
            }
            for player in self.test_player_data[:20]
        ]
        
        with self.runner.benchmark_context("adaptive_model_update"):
            self.adaptive_model.update_model(update_data, learning_rate=0.01)
    
    @pytest.mark.benchmark(group="base_models")
    def benchmark_availability_prediction(self):
        """Benchmark availability prediction performance."""
        # Single availability prediction
        with self.runner.benchmark_context("availability_predict_single"):
            for player_data in self.test_player_data[:50]:
                availability = self.availability_predictor.predict_availability(
                    player_data, gameweek=10
                )
        
        # Batch availability prediction
        with self.runner.benchmark_context("availability_predict_batch"):
            for player_data in self.test_player_data:
                availability = self.availability_predictor.predict_availability(
                    player_data, gameweek=10
                )
        
        # Injury data update
        with self.runner.benchmark_context("availability_update_injury"):
            for i, player_data in enumerate(self.test_player_data[:20]):
                injury_data = {
                    "type": "minor",
                    "duration": 2,
                    "probability": 0.3
                }
                self.availability_predictor.update_injury_data(i, injury_data)
    
    @pytest.mark.benchmark(group="base_models")
    def benchmark_form_calculation(self):
        """Benchmark form calculation performance."""
        # Single form calculation
        with self.runner.benchmark_context("form_calculate_single"):
            for _ in range(50):
                form = self.form_calculator.calculate_form(
                    self.test_performances[:5],
                    window_size=5
                )
        
        # Batch form calculation with different window sizes
        for window_size in [3, 5, 10]:
            with self.runner.benchmark_context(f"form_calculate_window_{window_size}"):
                for _ in range(20):
                    form = self.form_calculator.calculate_form(
                        self.test_performances,
                        window_size=window_size
                    )
        
        # Weighted form calculation
        with self.runner.benchmark_context("form_calculate_weighted"):
            for _ in range(50):
                weighted_form = self.form_calculator.calculate_weighted_form(
                    self.test_performances[:10]
                )
    
    @pytest.mark.benchmark(group="base_models")
    def benchmark_feature_engineering(self):
        """Benchmark feature engineering performance."""
        # Single feature extraction
        with self.runner.benchmark_context("feature_extract_single"):
            for player_data in self.test_player_data[:50]:
                features = self.feature_engineer.extract_features(player_data)
        
        # Batch feature extraction
        with self.runner.benchmark_context("feature_extract_batch"):
            feature_matrix = []
            for player_data in self.test_player_data:
                features = self.feature_engineer.extract_features(player_data)
                feature_matrix.append(features)
            
            feature_matrix = np.array(feature_matrix)
        
        # Feature pipeline creation
        with self.runner.benchmark_context("feature_pipeline_create"):
            transformations = ["normalize", "scale", "log"]
            pipeline = self.feature_engineer.create_feature_pipeline(transformations)
        
        # Feature pipeline execution
        test_features = np.random.uniform(0, 100, (100, 10))
        with self.runner.benchmark_context("feature_pipeline_execute"):
            for transform in pipeline:
                test_features = transform(test_features)
    
    @pytest.mark.benchmark(group="base_models")
    def benchmark_model_state_operations(self):
        """Benchmark model state serialization/deserialization."""
        # Get model state
        with self.runner.benchmark_context("model_state_get"):
            for _ in range(100):
                state = self.adaptive_model.get_model_state()
        
        # Set model state
        state = self.adaptive_model.get_model_state()
        with self.runner.benchmark_context("model_state_set"):
            for _ in range(100):
                self.adaptive_model.set_model_state(state)
        
        # State serialization
        with self.runner.benchmark_context("model_state_serialize"):
            import json
            for _ in range(100):
                state = self.adaptive_model.get_model_state()
                # Convert numpy arrays to lists for JSON serialization
                serializable_state = {}
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        serializable_state[key] = value.tolist()
                    else:
                        serializable_state[key] = value
                
                json_state = json.dumps(serializable_state)


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="base_models_single")
def test_adaptive_model_prediction(benchmark):
    """Pytest-benchmark test for adaptive model prediction."""
    model = MockAdaptiveModel()
    features = np.random.uniform(0, 10, 15)
    player_data = {"position": "MID", "games_played": 20}
    
    def predict():
        return model.predict(features, player_data)
    
    result = benchmark(predict)


@pytest.mark.benchmark(group="base_models_single")
def test_form_calculation(benchmark):
    """Pytest-benchmark test for form calculation."""
    calculator = MockFormCalculator()
    performances = [
        {"points": np.random.uniform(0, 15), "minutes": np.random.randint(0, 90)}
        for _ in range(10)
    ]
    
    def calculate_form():
        return calculator.calculate_form(performances, window_size=5)
    
    result = benchmark(calculate_form)


@pytest.mark.benchmark(group="base_models_single")
def test_feature_extraction(benchmark):
    """Pytest-benchmark test for feature extraction."""
    engineer = MockFeatureEngineer()
    player_data = {
        "goals": 5, "assists": 3, "minutes": 1500,
        "clean_sheets": 2, "yellow_cards": 1,
        "games_played": 20, "position": "MID",
        "recent_points": [4.5, 6.0, 2.0, 8.5, 3.0]
    }
    
    def extract_features():
        return engineer.extract_features(player_data)
    
    result = benchmark(extract_features)