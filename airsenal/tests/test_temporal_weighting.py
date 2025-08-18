"""
Comprehensive tests for temporal weighting system.

Tests all decay functions, weight optimization, position-specific weighting,
importance weighting, and integration with existing systems.
"""

import pytest
import time
import numpy as np
import pandas as pd
import jax.numpy as jnp
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from airsenal.framework.temporal_weighting import (
    DecayFunctions,
    DecayConfig,
    ImportanceConfig,
    PositionConfig,
    ImportanceWeighting,
    PositionSpecificWeighting,
    WeightOptimizer,
    TemporalWeightingSystem,
    WeightingResult,
    create_default_weighting_system,
    calculate_simple_temporal_weights,
    extend_measurement_processor_with_temporal_weights,
    create_temporal_weighted_kalman_update
)


class TestDecayFunctions:
    """Test all decay function implementations."""
    
    def setup_method(self):
        """Set up test data."""
        self.time_diffs = jnp.array([0, 1, 7, 14, 28, 56, 84])
        self.lambda_ = 0.1
        self.alpha = 0.05
        self.beta = 1.5
        self.gamma = 2.0
        self.sigma = 14.0
        self.min_weight = 0.01
    
    def test_exponential_decay(self):
        """Test exponential decay function."""
        weights = DecayFunctions.exponential_decay(
            self.time_diffs, self.lambda_, self.min_weight
        )
        
        # Check basic properties
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        assert weights[0] == 1.0  # t=0 should have weight 1
        assert jnp.all(weights[1:] <= weights[:-1])  # Monotonically decreasing
        
        # Check exponential formula
        expected = jnp.exp(-self.lambda_ * self.time_diffs)
        expected = jnp.maximum(expected, self.min_weight)
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_linear_decay(self):
        """Test linear decay function."""
        weights = DecayFunctions.linear_decay(
            self.time_diffs, self.alpha, self.min_weight
        )
        
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        assert weights[0] == 1.0
        
        # Check linear formula for valid range
        expected = jnp.maximum(0, 1 - self.alpha * self.time_diffs)
        expected = jnp.maximum(expected, self.min_weight)
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_power_law_decay(self):
        """Test power law decay function."""
        weights = DecayFunctions.power_law_decay(
            self.time_diffs, self.beta, self.min_weight
        )
        
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        assert jnp.all(weights[1:] <= weights[:-1])
        
        # Check power law formula
        expected = jnp.power(1 + self.time_diffs, -self.beta)
        expected = jnp.maximum(expected, self.min_weight)
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_hyperbolic_decay(self):
        """Test hyperbolic decay function."""
        weights = DecayFunctions.hyperbolic_decay(
            self.time_diffs, self.gamma, self.min_weight
        )
        
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        assert jnp.all(weights[1:] <= weights[:-1])
        
        # Check hyperbolic formula
        expected = 1.0 / (1.0 + self.gamma * self.time_diffs)
        expected = jnp.maximum(expected, self.min_weight)
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_step_function_decay(self):
        """Test step function decay."""
        step_windows = [
            (0, 7, 1.0),
            (7, 21, 0.8),
            (21, 42, 0.6),
            (42, float('inf'), 0.4)
        ]
        
        weights = DecayFunctions.step_function_decay(
            self.time_diffs, step_windows, self.min_weight
        )
        
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        
        # Check specific values
        assert weights[0] == 1.0   # t=0
        assert weights[1] == 1.0   # t=1
        assert weights[2] == 0.8   # t=7
        assert weights[3] == 0.8   # t=14
        assert weights[4] == 0.6   # t=28
        assert weights[5] == 0.4   # t=56
        assert weights[6] == 0.4   # t=84
    
    def test_gaussian_decay(self):
        """Test Gaussian decay function."""
        weights = DecayFunctions.gaussian_decay(
            self.time_diffs, self.sigma, self.min_weight
        )
        
        assert len(weights) == len(self.time_diffs)
        assert jnp.all(weights >= self.min_weight)
        assert weights[0] == 1.0  # t=0 should have weight 1
        
        # Check Gaussian formula
        expected = jnp.exp(-0.5 * jnp.square(self.time_diffs / self.sigma))
        expected = jnp.maximum(expected, self.min_weight)
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_edge_cases(self):
        """Test edge cases for all decay functions."""
        # Empty input
        empty_input = jnp.array([])
        
        for func in [DecayFunctions.exponential_decay, DecayFunctions.linear_decay,
                    DecayFunctions.power_law_decay, DecayFunctions.hyperbolic_decay,
                    DecayFunctions.gaussian_decay]:
            result = func(empty_input, 0.1, 0.01)
            assert len(result) == 0
        
        # Single value
        single_input = jnp.array([5.0])
        
        for func in [DecayFunctions.exponential_decay, DecayFunctions.linear_decay,
                    DecayFunctions.power_law_decay, DecayFunctions.hyperbolic_decay,
                    DecayFunctions.gaussian_decay]:
            result = func(single_input, 0.1, 0.01)
            assert len(result) == 1
            assert result[0] >= 0.01
        
        # Large time differences
        large_input = jnp.array([1000.0, 10000.0])
        
        for func in [DecayFunctions.exponential_decay, DecayFunctions.linear_decay,
                    DecayFunctions.power_law_decay, DecayFunctions.hyperbolic_decay,
                    DecayFunctions.gaussian_decay]:
            result = func(large_input, 0.1, 0.01)
            assert jnp.all(result >= 0.01)  # Should clamp to min_weight


class TestImportanceWeighting:
    """Test match importance weighting system."""
    
    def setup_method(self):
        """Set up test data."""
        self.importance_weighting = ImportanceWeighting()
    
    def test_derby_match_boost(self):
        """Test derby match importance boost."""
        # Test known derby
        importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Tottenham", 20
        )
        assert importance > 1.0
        
        # Test non-derby
        importance_normal = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Brighton", 20
        )
        assert importance > importance_normal
    
    def test_big_six_fixture_boost(self):
        """Test big six fixture boost."""
        importance = self.importance_weighting.calculate_match_importance(
            "Manchester City", "Liverpool", 20
        )
        assert importance > 1.0
        
        # Compare with non-big-six fixture
        importance_normal = self.importance_weighting.calculate_match_importance(
            "Manchester City", "Burnley", 20
        )
        assert importance > importance_normal
    
    def test_season_phase_adjustment(self):
        """Test season phase adjustments."""
        # Early season
        early_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 5
        )
        
        # Mid season
        mid_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 20
        )
        
        # Late season
        late_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 35
        )
        
        assert late_importance > mid_importance
        assert early_importance > mid_importance
    
    def test_league_position_importance(self):
        """Test league position-based importance."""
        # Title race
        title_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Manchester City", 30, 
            home_position=2, away_position=1
        )
        
        # Mid-table clash
        mid_importance = self.importance_weighting.calculate_match_importance(
            "Brighton", "Crystal Palace", 30,
            home_position=10, away_position=12
        )
        
        assert title_importance > mid_importance
    
    def test_competition_type_adjustment(self):
        """Test different competition weightings."""
        # Premier League
        pl_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 20, competition="Premier League"
        )
        
        # Champions League
        cl_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 20, competition="Champions League"
        )
        
        # FA Cup
        fa_importance = self.importance_weighting.calculate_match_importance(
            "Arsenal", "Chelsea", 20, competition="FA Cup"
        )
        
        assert cl_importance > pl_importance
        assert pl_importance > fa_importance
    
    def test_opponent_quality_adjustment(self):
        """Test opponent quality adjustments."""
        # Test against strong opponent
        strong_adj = self.importance_weighting.calculate_opponent_quality_adjustment(
            "Manchester City", "2023", "Brighton", is_home=True
        )
        
        # Test against weak opponent
        weak_adj = self.importance_weighting.calculate_opponent_quality_adjustment(
            "Brighton", "2023", "Manchester City", is_home=True
        )
        
        # Strong opponent should make performance more valuable
        assert strong_adj != weak_adj
    
    def test_home_away_adjustment(self):
        """Test home/away adjustments."""
        home_adj = self.importance_weighting.calculate_opponent_quality_adjustment(
            "Arsenal", "2023", "Chelsea", is_home=True
        )
        
        away_adj = self.importance_weighting.calculate_opponent_quality_adjustment(
            "Arsenal", "2023", "Chelsea", is_home=False
        )
        
        assert home_adj != away_adj


class TestPositionSpecificWeighting:
    """Test position-specific weighting adjustments."""
    
    def setup_method(self):
        """Set up test data."""
        self.position_weighting = PositionSpecificWeighting()
        self.base_config = DecayConfig()
    
    def test_position_decay_adjustments(self):
        """Test position-specific decay adjustments."""
        positions = ["GK", "DEF", "MID", "FWD"]
        configs = {}
        
        for position in positions:
            config = self.position_weighting.get_position_adjusted_decay_config(
                position, self.base_config
            )
            configs[position] = config
        
        # Goalkeepers should have slower decay (lower lambda)
        assert configs["GK"].lambda_ < configs["FWD"].lambda_
        
        # All configs should be valid
        for config in configs.values():
            assert config.lambda_ > 0
            assert config.min_weight >= 0
    
    def test_memory_length_differences(self):
        """Test different memory lengths by position."""
        positions = ["GK", "DEF", "MID", "FWD"]
        horizons = {}
        
        for position in positions:
            config = self.position_weighting.get_position_adjusted_decay_config(
                position, self.base_config
            )
            horizons[position] = config.max_time_horizon
        
        # Goalkeepers should have longer memory
        assert horizons["GK"] > horizons["FWD"]
        assert horizons["DEF"] > horizons["FWD"]
    
    def test_form_adjustment_calculation(self):
        """Test form-based adjustments."""
        # Improving form
        improving_performances = jnp.array([0.3, 0.5, 0.7, 0.9])
        time_diffs = jnp.array([21, 14, 7, 0])
        
        adj = self.position_weighting.calculate_form_adjustment(
            "FWD", improving_performances, time_diffs
        )
        assert adj >= 1.0  # Improving form should boost weights
        
        # Declining form
        declining_performances = jnp.array([0.9, 0.7, 0.5, 0.3])
        
        adj = self.position_weighting.calculate_form_adjustment(
            "FWD", declining_performances, time_diffs
        )
        assert adj <= 1.0  # Declining form should reduce weights
    
    def test_position_sensitivity_differences(self):
        """Test different form sensitivities by position."""
        performances = jnp.array([0.3, 0.5, 0.7, 0.9])
        time_diffs = jnp.array([21, 14, 7, 0])
        
        gk_adj = self.position_weighting.calculate_form_adjustment(
            "GK", performances, time_diffs
        )
        
        fwd_adj = self.position_weighting.calculate_form_adjustment(
            "FWD", performances, time_diffs
        )
        
        # Forwards should be more sensitive to form changes
        assert abs(fwd_adj - 1.0) >= abs(gk_adj - 1.0)


class TestWeightOptimizer:
    """Test weight parameter optimization."""
    
    def setup_method(self):
        """Set up test data."""
        self.optimizer = WeightOptimizer(optimization_method="grid_search", n_calls=10)
        
        # Create synthetic historical data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="7D")
        
        self.historical_data = pd.DataFrame({
            "player_id": np.repeat(range(10), 10),
            "date": np.tile(dates[:10], 10),
            "position": np.repeat(["GK", "DEF", "MID", "FWD"], [25, 25, 25, 25]),
            "points": np.random.gamma(2, 2, 100),
            "goals": np.random.poisson(0.3, 100),
            "assists": np.random.poisson(0.2, 100)
        })
    
    def test_parameter_optimization(self):
        """Test basic parameter optimization."""
        config, score = self.optimizer.optimize_decay_parameters(
            self.historical_data, "points", "MID", "exponential"
        )
        
        assert isinstance(config, DecayConfig)
        assert config.decay_type == "exponential"
        assert config.lambda_ > 0
        assert isinstance(score, float)
    
    def test_optimization_all_positions(self):
        """Test optimization for all positions."""
        results = self.optimizer.optimize_all_positions(
            self.historical_data, "points", "exponential"
        )
        
        assert len(results) == 4
        assert all(pos in results for pos in ["GK", "DEF", "MID", "FWD"])
        
        for position, (config, score) in results.items():
            assert isinstance(config, DecayConfig)
            assert isinstance(score, float)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        small_data = self.historical_data.head(5)
        
        config, score = self.optimizer.optimize_decay_parameters(
            small_data, "points", "MID", "exponential"
        )
        
        # Should return default config
        assert isinstance(config, DecayConfig)
        assert score == 0.0
    
    def test_different_decay_types(self):
        """Test optimization for different decay types."""
        decay_types = ["exponential", "linear", "power_law", "hyperbolic", "gaussian"]
        
        for decay_type in decay_types:
            config, score = self.optimizer.optimize_decay_parameters(
                self.historical_data, "points", "MID", decay_type
            )
            
            assert config.decay_type == decay_type
            assert isinstance(score, float)


class TestTemporalWeightingSystem:
    """Test main temporal weighting system."""
    
    def setup_method(self):
        """Set up test data."""
        self.system = TemporalWeightingSystem()
        
        self.observation_dates = [
            "2023-10-01", "2023-10-08", "2023-10-15", "2023-10-22", "2023-10-29"
        ]
        self.current_date = "2023-11-05"
        
        self.match_contexts = [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "gameweek": 10,
                "opponent_team": "Chelsea",
                "player_team": "Arsenal",
                "is_home": True
            },
            {
                "home_team": "Brighton",
                "away_team": "Arsenal", 
                "gameweek": 11,
                "opponent_team": "Brighton",
                "player_team": "Arsenal",
                "is_home": False
            },
            None,  # No context available
            {
                "home_team": "Arsenal",
                "away_team": "Tottenham",
                "gameweek": 13,
                "opponent_team": "Tottenham",
                "player_team": "Arsenal",
                "is_home": True
            },
            {
                "home_team": "Liverpool",
                "away_team": "Arsenal",
                "gameweek": 14,
                "opponent_team": "Liverpool", 
                "player_team": "Arsenal",
                "is_home": False
            }
        ]
    
    def test_basic_weight_calculation(self):
        """Test basic weight calculation."""
        result = self.system.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID"
        )
        
        assert isinstance(result, WeightingResult)
        assert len(result.weights) == len(self.observation_dates)
        assert len(result.normalized_weights) == len(self.observation_dates)
        assert jnp.sum(result.normalized_weights) == pytest.approx(1.0, rel=1e-6)
        assert result.total_weight > 0
        assert result.effective_sample_size > 0
        assert result.confidence_score >= 0
    
    def test_position_specific_weighting(self):
        """Test position-specific weight differences."""
        positions = ["GK", "DEF", "MID", "FWD"]
        results = {}
        
        for position in positions:
            result = self.system.calculate_weights(
                observation_dates=self.observation_dates,
                current_date=self.current_date,
                position=position
            )
            results[position] = result
        
        # All should have valid results
        for result in results.values():
            assert jnp.sum(result.normalized_weights) == pytest.approx(1.0, rel=1e-6)
            assert result.total_weight > 0
        
        # Different positions should have different weight patterns
        gk_weights = results["GK"].normalized_weights
        fwd_weights = results["FWD"].normalized_weights
        
        # Not exactly equal (allowing for different decay patterns)
        assert not jnp.allclose(gk_weights, fwd_weights, rtol=1e-3)
    
    def test_importance_weighting(self):
        """Test importance-based weight adjustments."""
        # Without importance context
        result_basic = self.system.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID"
        )
        
        # With importance context
        result_importance = self.system.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID",
            match_contexts=self.match_contexts
        )
        
        assert result_basic.importance_adjustment_applied == False
        assert result_importance.importance_adjustment_applied == True
        
        # Derby match (Arsenal vs Tottenham) should have higher weight
        derby_index = 3  # Arsenal vs Tottenham
        assert result_importance.normalized_weights[derby_index] >= result_basic.normalized_weights[derby_index]
    
    def test_different_decay_functions(self):
        """Test different decay function types."""
        decay_types = ["exponential", "linear", "power_law", "hyperbolic", "gaussian"]
        
        for decay_type in decay_types:
            custom_config = DecayConfig(decay_type=decay_type)
            result = self.system.calculate_weights(
                observation_dates=self.observation_dates,
                current_date=self.current_date,
                position="MID",
                custom_decay_config=custom_config
            )
            
            assert result.decay_function_used == decay_type
            assert jnp.sum(result.normalized_weights) == pytest.approx(1.0, rel=1e-6)
            assert result.total_weight > 0
    
    def test_weight_application(self):
        """Test applying weights to measurements."""
        # Create sample measurements
        measurements = jnp.array([
            [1.0, 2.0, 3.0],  # observation 1
            [1.5, 2.5, 3.5],  # observation 2
            [2.0, 3.0, 4.0],  # observation 3
            [1.2, 2.2, 3.2],  # observation 4
            [1.8, 2.8, 3.8]   # observation 5
        ])
        
        result = self.system.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID"
        )
        
        # Test different application methods
        weighted_multiply = self.system.apply_weights(
            measurements, result.normalized_weights, method="multiply"
        )
        
        weighted_avg = self.system.apply_weights(
            measurements, result.normalized_weights, method="weighted_average"
        )
        
        weighted_sum = self.system.apply_weights(
            measurements, result.normalized_weights, method="weighted_sum"
        )
        
        # Check shapes
        assert weighted_multiply.shape == measurements.shape
        assert weighted_avg.shape == (3,)  # Features dimension
        assert weighted_sum.shape == (3,)  # Features dimension
        
        # Check that weights are applied correctly
        expected_avg = jnp.sum(measurements * result.normalized_weights[:, np.newaxis], axis=0)
        np.testing.assert_array_almost_equal(weighted_avg, expected_avg, decimal=5)
    
    def test_batch_weight_calculation(self):
        """Test batch weight calculation."""
        batch_data = [
            {
                "observation_dates": self.observation_dates,
                "current_date": self.current_date,
                "position": "MID"
            },
            {
                "observation_dates": self.observation_dates,
                "current_date": self.current_date,
                "position": "FWD"
            },
            {
                "observation_dates": self.observation_dates[:3],
                "current_date": self.current_date,
                "position": "DEF"
            }
        ]
        
        results = self.system.batch_calculate_weights(batch_data)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, WeightingResult)
            assert result.total_weight > 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty observations
        result = self.system.calculate_weights(
            observation_dates=[],
            current_date=self.current_date,
            position="MID"
        )
        assert len(result.weights) == 0
        assert result.total_weight == 0
        
        # Future dates (should be filtered out)
        future_dates = ["2023-12-01", "2023-12-08"]
        result = self.system.calculate_weights(
            observation_dates=future_dates,
            current_date=self.current_date,
            position="MID"
        )
        assert result.total_weight == 0
        
        # Single observation
        result = self.system.calculate_weights(
            observation_dates=["2023-10-29"],
            current_date=self.current_date,
            position="MID"
        )
        assert len(result.weights) == 1
        assert result.normalized_weights[0] == 1.0
    
    def test_performance_metrics(self):
        """Test performance metric tracking."""
        # Perform some calculations
        for _ in range(5):
            self.system.calculate_weights(
                observation_dates=self.observation_dates,
                current_date=self.current_date,
                position="MID"
            )
        
        metrics = self.system.get_performance_metrics()
        
        assert "total_calculations" in metrics
        assert "average_calculation_time_ms" in metrics
        assert metrics["total_calculations"] >= 5
        assert metrics["average_calculation_time_ms"] >= 0
    
    def test_caching_functionality(self):
        """Test weight caching for performance."""
        system_with_cache = TemporalWeightingSystem(enable_caching=True)
        
        # Make same calculation twice
        result1 = system_with_cache.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID"
        )
        
        result2 = system_with_cache.calculate_weights(
            observation_dates=self.observation_dates,
            current_date=self.current_date,
            position="MID"
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result1.weights, result2.weights)
        np.testing.assert_array_equal(result1.normalized_weights, result2.normalized_weights)


class TestUtilityFunctions:
    """Test utility functions and integrations."""
    
    def test_create_default_weighting_system(self):
        """Test default system creation."""
        system = create_default_weighting_system(decay_type="exponential")
        
        assert isinstance(system, TemporalWeightingSystem)
        assert system.decay_config.decay_type == "exponential"
    
    def test_calculate_simple_temporal_weights(self):
        """Test simple weight calculation utility."""
        observation_dates = ["2023-10-01", "2023-10-08", "2023-10-15"]
        current_date = "2023-10-22"
        
        weights = calculate_simple_temporal_weights(
            observation_dates, current_date, position="MID"
        )
        
        assert len(weights) == len(observation_dates)
        assert jnp.sum(weights) == pytest.approx(1.0, rel=1e-6)
        assert jnp.all(weights >= 0)
    
    def test_measurement_processor_extension(self):
        """Test extending MeasurementProcessor with temporal weights."""
        # Create mock measurement processor
        mock_processor = Mock()
        mock_processor.normalize_features = Mock(return_value=jnp.array([1.0, 2.0, 3.0]))
        
        # Create weighting system
        weighting_system = TemporalWeightingSystem()
        
        # Extend processor
        extend_measurement_processor_with_temporal_weights(
            mock_processor, weighting_system
        )
        
        # Test extended functionality
        result = mock_processor.normalize_features(
            features={"goals": 1, "assists": 2},
            position="MID",
            observation_dates=["2023-10-01", "2023-10-08"],
            current_date="2023-10-15"
        )
        
        assert isinstance(result, jnp.ndarray)
        assert len(result) == 3
    
    def test_kalman_filter_integration(self):
        """Test Kalman filter integration."""
        # Create mock Kalman filter
        mock_filter = Mock()
        mock_filter.update = Mock(return_value="updated_state")
        
        # Create weighting system
        weighting_system = TemporalWeightingSystem()
        
        # Create temporal weighted update function
        temporal_update = create_temporal_weighted_kalman_update(
            mock_filter, weighting_system
        )
        
        # Test update with temporal weighting
        result = temporal_update(
            state="test_state",
            observation=jnp.array([1.0, 2.0]),
            noise_matrix=jnp.eye(2),
            observation_dates=["2023-10-01", "2023-10-08"],
            current_date="2023-10-15",
            position="MID"
        )
        
        assert result == "updated_state"
        
        # Verify that update was called with modified noise matrix
        mock_filter.update.assert_called_once()
        call_args = mock_filter.update.call_args
        
        # Noise matrix should be modified (not identity)
        noise_matrix_used = call_args[0][2]  # Third positional argument
        assert not jnp.allclose(noise_matrix_used, jnp.eye(2))


class TestPerformanceOptimization:
    """Test performance optimizations and vectorization."""
    
    def test_vectorized_decay_functions(self):
        """Test that decay functions work with vectorized inputs."""
        # Large time difference array
        large_time_diffs = jnp.linspace(0, 100, 1000)
        
        # Test all decay functions with large inputs
        decay_functions = [
            (DecayFunctions.exponential_decay, {"lambda_": 0.1}),
            (DecayFunctions.linear_decay, {"alpha": 0.05}),
            (DecayFunctions.power_law_decay, {"beta": 1.5}),
            (DecayFunctions.hyperbolic_decay, {"gamma": 2.0}),
            (DecayFunctions.gaussian_decay, {"sigma": 14.0})
        ]
        
        for func, params in decay_functions:
            start_time = time.time()
            weights = func(large_time_diffs, **params, min_weight=0.01)
            end_time = time.time()
            
            # Should complete quickly (vectorized)
            assert (end_time - start_time) < 0.1  # Less than 100ms
            assert len(weights) == len(large_time_diffs)
            assert jnp.all(weights >= 0.01)
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        import time
        
        system = TemporalWeightingSystem()
        
        # Create large batch
        observation_dates = [f"2023-{month:02d}-01" for month in range(1, 13)]
        current_date = "2023-12-15"
        
        batch_data = [
            {
                "observation_dates": observation_dates,
                "current_date": current_date,
                "position": position
            }
            for position in ["GK", "DEF", "MID", "FWD"] * 25  # 100 calculations
        ]
        
        start_time = time.time()
        results = system.batch_calculate_weights(batch_data)
        end_time = time.time()
        
        # Should complete reasonably quickly
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Less than 5 seconds for 100 calculations
        assert len(results) == 100
        
        # All results should be valid
        for result in results:
            assert isinstance(result, WeightingResult)
            assert result.total_weight >= 0
    
    def test_large_dataset_optimization(self):
        """Test optimization with large datasets."""
        # Create large synthetic dataset
        np.random.seed(42)
        n_players = 50
        n_gameweeks = 38
        
        dates = pd.date_range("2023-08-01", periods=n_gameweeks, freq="7D")
        
        large_data = []
        for player_id in range(n_players):
            for gw, date in enumerate(dates):
                large_data.append({
                    "player_id": player_id,
                    "date": date,
                    "position": ["GK", "DEF", "MID", "FWD"][player_id % 4],
                    "points": np.random.gamma(2, 2),
                    "goals": np.random.poisson(0.3),
                    "assists": np.random.poisson(0.2)
                })
        
        large_df = pd.DataFrame(large_data)
        
        # Test optimization with large dataset
        optimizer = WeightOptimizer(optimization_method="grid_search", n_calls=5)
        
        start_time = time.time()
        config, score = optimizer.optimize_decay_parameters(
            large_df, "points", "MID", "exponential"
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 30.0  # Less than 30 seconds
        assert isinstance(config, DecayConfig)
        assert isinstance(score, float)


class TestIntegrationWithExistingSystems:
    """Test integration with existing AIrsenal systems."""
    
    def setup_method(self):
        """Set up integration test data."""
        self.weighting_system = TemporalWeightingSystem()
    
    @patch('airsenal.framework.temporal_weighting.logger')
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and logging."""
        # Test with invalid date format
        result = self.weighting_system.calculate_weights(
            observation_dates=["invalid-date"],
            current_date="2023-10-15",
            position="MID"
        )
        
        # Should return empty result gracefully
        assert isinstance(result, WeightingResult)
        assert len(result.weights) == 0
        
        # Should have logged error
        mock_logger.error.assert_called()
    
    def test_configuration_flexibility(self):
        """Test system configuration flexibility."""
        # Test custom configurations
        custom_decay = DecayConfig(
            decay_type="exponential",
            lambda_=0.2,
            min_weight=0.05
        )
        
        custom_importance = ImportanceConfig(
            derby_boost=1.5,
            big_six_boost=1.3
        )
        
        custom_position = PositionConfig(
            memory_lengths={"GK": 15.0, "DEF": 12.0, "MID": 10.0, "FWD": 8.0}
        )
        
        system = TemporalWeightingSystem(
            decay_config=custom_decay,
            importance_config=custom_importance,
            position_config=custom_position
        )
        
        result = system.calculate_weights(
            observation_dates=["2023-10-01", "2023-10-08"],
            current_date="2023-10-15",
            position="GK"
        )
        
        assert isinstance(result, WeightingResult)
        assert result.decay_function_used == "exponential"
        assert result.position_adjustment_applied == True
    
    def test_real_world_scenario_simulation(self):
        """Test realistic scenario with typical AIrsenal data patterns."""
        # Simulate player performance over a season
        gameweeks = list(range(1, 20))  # Half season
        observation_dates = [f"2023-{8 + (gw-1)//4:02d}-{((gw-1)%4)*7 + 1:02d}" for gw in gameweeks]
        current_date = "2024-01-01"
        
        # Test for different positions
        positions = ["GK", "DEF", "MID", "FWD"]
        
        for position in positions:
            result = self.weighting_system.calculate_weights(
                observation_dates=observation_dates,
                current_date=current_date,
                position=position
            )
            
            # Validate realistic properties
            assert len(result.weights) == len(gameweeks)
            assert 3.0 <= result.effective_sample_size <= len(gameweeks)
            assert 0.5 <= result.confidence_score <= 1.0
            assert result.oldest_significant_observation is not None
            
            # Recent observations should have higher weights
            recent_weight = result.normalized_weights[-1]  # Most recent
            old_weight = result.normalized_weights[0]      # Oldest
            assert recent_weight >= old_weight


if __name__ == "__main__":
    pytest.main([__file__])