"""
Comprehensive tests for State Prediction Module

This test suite covers all aspects of the state prediction implementation:
- StatePredictor functionality and multi-step predictions
- UncertaintyPropagator and uncertainty growth modeling
- PredictionValidator and calibration testing
- SeasonalityHandler and seasonal adjustments
- FixtureAwarePredictor and fixture context
- RiskMetrics calculation and validation
- PredictivePlayerModel integration
- Performance and optimization integration

Test Structure:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete prediction pipeline
- Performance and validation tests
- Error handling and edge case tests
"""

import logging
import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from airsenal.framework.state_prediction import (
    StatePredictor,
    UncertaintyPropagator,
    PredictionValidator,
    SeasonalityHandler,
    FixtureAwarePredictor,
    RiskMetrics,
    PredictionResult,
    PredictionInterval,
    PredictionConfig,
    StatePredictionError,
    get_predictions,
    get_prediction_intervals,
    get_risk_metrics,
    scenario_analysis
)

from airsenal.framework.predictive_player_model import (
    PredictivePlayerModel,
    ModelEnsemble,
    PredictiveModelError
)

from airsenal.framework.kalman_player_model import (
    KalmanPlayerModel,
    KalmanPlayerModelError
)

from airsenal.framework.kalman_filter import (
    FilterConfig,
    FilterState,
    KalmanFilter,
    create_player_ability_config
)

from airsenal.framework.adaptive_player_model import (
    StateSpaceConfig,
    PlayerState,
    PlayerData
)

# Test configuration
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def prediction_config():
    """Basic prediction configuration for testing."""
    return PredictionConfig(
        max_gameweeks_ahead=5,
        default_confidence_levels=[0.5, 0.8, 0.95],
        uncertainty_inflation_rate=0.05,
        enable_seasonality=True,
        enable_fixture_awareness=True,
        min_validation_samples=10
    )


@pytest.fixture
def filter_config():
    """Basic filter configuration for testing."""
    return FilterConfig(
        state_dim=4,
        obs_dim=4,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
        use_joseph_form=True,
        enable_adaptive_noise=True
    )


@pytest.fixture
def state_space_config():
    """Basic state space configuration for testing."""
    return StateSpaceConfig(
        state_dim=4,
        obs_dim=4,
        state_names=["skill", "form", "consistency", "momentum"],
        obs_names=["goals", "assists", "minutes", "bonus"]
    )


@pytest.fixture
def mock_kalman_model(filter_config, state_space_config):
    """Mock Kalman player model for testing."""
    model = Mock(spec=KalmanPlayerModel)
    model.config = state_space_config
    model.filter_config = filter_config
    model.is_fitted = True
    
    # Mock filters
    mock_filter = Mock(spec=KalmanFilter)
    mock_filter.config = filter_config
    model.filters = {"main": mock_filter}
    
    # Mock player states
    player_state = PlayerState(
        player_id=123,
        state_mean=jnp.array([0.5, 0.6, 0.7, 0.1]),
        state_cov=jnp.eye(4) * 0.1,
        gameweek=15,
        season="2024",
        last_updated=datetime.now(timezone.utc).isoformat()
    )
    model.player_states = {123: player_state}
    
    # Mock methods
    model.predict_state.return_value = player_state
    model.observation_model.return_value = jnp.array([2.0, 1.5, 85.0, 1.0])
    
    return model


@pytest.fixture
def sample_prediction_result():
    """Sample prediction result for testing."""
    return PredictionResult(
        player_id=123,
        gameweeks_ahead=3,
        prediction_date=datetime.now(timezone.utc).isoformat(),
        expected_state=jnp.array([0.6, 0.7, 0.8, 0.2]),
        state_covariance=jnp.eye(4) * 0.15,
        expected_performance=jnp.array([2.5, 1.8, 80.0, 1.2]),
        performance_variance=jnp.array([0.5, 0.3, 100.0, 0.8]),
        prediction_intervals={},
        risk_metrics=None
    )


class TestRiskMetrics:
    """Test risk metrics calculation and validation."""
    
    def test_risk_metrics_from_distribution(self):
        """Test risk metrics calculation from sample distribution."""
        # Generate sample data
        np.random.seed(42)
        samples = np.random.normal(5.0, 2.0, 1000)
        
        # Calculate risk metrics
        risk_metrics = RiskMetrics.from_distribution(samples, var_confidence=0.05)
        
        # Validate results
        assert isinstance(risk_metrics.expected_value, float)
        assert isinstance(risk_metrics.variance, float)
        assert isinstance(risk_metrics.standard_deviation, float)
        assert isinstance(risk_metrics.value_at_risk, float)
        assert isinstance(risk_metrics.conditional_var, float)
        assert isinstance(risk_metrics.sharpe_ratio, float)
        
        # Check reasonable values
        assert abs(risk_metrics.expected_value - 5.0) < 0.2  # Close to true mean
        assert abs(risk_metrics.standard_deviation - 2.0) < 0.2  # Close to true std
        assert risk_metrics.variance > 0
        assert risk_metrics.value_at_risk > 0  # Should be positive (loss)
        assert risk_metrics.conditional_var >= risk_metrics.value_at_risk  # CVaR >= VaR
    
    def test_risk_metrics_edge_cases(self):
        """Test risk metrics with edge cases."""
        # Constant samples (no variance)
        constant_samples = np.full(100, 5.0)
        risk_metrics = RiskMetrics.from_distribution(constant_samples)
        
        assert risk_metrics.variance == 0.0
        assert risk_metrics.standard_deviation == 0.0
        assert risk_metrics.sharpe_ratio == 0.0  # Undefined, should handle gracefully
        
        # Single sample
        single_sample = np.array([5.0])
        risk_metrics = RiskMetrics.from_distribution(single_sample)
        
        assert risk_metrics.expected_value == 5.0
        assert risk_metrics.variance == 0.0


class TestPredictionInterval:
    """Test prediction interval functionality."""
    
    def test_prediction_interval_creation(self):
        """Test prediction interval creation and validation."""
        lower = jnp.array([1.0, 0.5, 60.0, 0.5])
        upper = jnp.array([3.0, 2.5, 90.0, 2.0])
        
        interval = PredictionInterval(
            confidence_level=0.8,
            lower_bound=lower,
            upper_bound=upper,
            width=upper - lower
        )
        
        assert interval.confidence_level == 0.8
        assert jnp.array_equal(interval.lower_bound, lower)
        assert jnp.array_equal(interval.upper_bound, upper)
        assert jnp.array_equal(interval.width, upper - lower)
    
    def test_prediction_interval_validation(self):
        """Test prediction interval validation."""
        with pytest.raises(ValueError):
            PredictionInterval(
                confidence_level=1.5,  # Invalid confidence level
                lower_bound=jnp.array([1.0]),
                upper_bound=jnp.array([2.0]),
                width=jnp.array([1.0])
            )
        
        with pytest.raises(ValueError):
            PredictionInterval(
                confidence_level=0.0,  # Invalid confidence level
                lower_bound=jnp.array([1.0]),
                upper_bound=jnp.array([2.0]),
                width=jnp.array([1.0])
            )


class TestUncertaintyPropagator:
    """Test uncertainty propagation functionality."""
    
    def test_uncertainty_propagator_initialization(self, prediction_config, mock_kalman_model):
        """Test uncertainty propagator initialization."""
        base_filter = mock_kalman_model.filters["main"]
        propagator = UncertaintyPropagator(prediction_config, base_filter)
        
        assert propagator.config == prediction_config
        assert propagator.base_filter == base_filter
        assert propagator.inflation_rate == prediction_config.uncertainty_inflation_rate
    
    def test_propagate_uncertainty_basic(self, prediction_config, mock_kalman_model):
        """Test basic uncertainty propagation."""
        base_filter = mock_kalman_model.filters["main"]
        propagator = UncertaintyPropagator(prediction_config, base_filter)
        
        initial_cov = jnp.eye(4) * 0.1
        gameweeks_ahead = 3
        
        propagated_cov = propagator.propagate_uncertainty(initial_cov, gameweeks_ahead)
        
        # Check that uncertainty increases
        assert jnp.all(jnp.diag(propagated_cov) > jnp.diag(initial_cov))
        
        # Check matrix is still positive definite
        eigenvals = jnp.linalg.eigvals(propagated_cov)
        assert jnp.all(eigenvals > 0)
    
    def test_propagate_uncertainty_with_external_factors(self, prediction_config, mock_kalman_model):
        """Test uncertainty propagation with external factors."""
        base_filter = mock_kalman_model.filters["main"]
        propagator = UncertaintyPropagator(prediction_config, base_filter)
        
        initial_cov = jnp.eye(4) * 0.1
        external_factors = {"fixture_difficulty": 0.8, "injury_risk": 0.3}
        
        propagated_cov = propagator.propagate_uncertainty(
            initial_cov, 2, external_factors
        )
        
        # Should have more uncertainty than without external factors
        baseline_cov = propagator.propagate_uncertainty(initial_cov, 2)
        assert jnp.all(jnp.diag(propagated_cov) >= jnp.diag(baseline_cov))
    
    def test_calculate_prediction_intervals(self, prediction_config, mock_kalman_model):
        """Test prediction interval calculation."""
        base_filter = mock_kalman_model.filters["main"]
        propagator = UncertaintyPropagator(prediction_config, base_filter)
        
        predicted_mean = jnp.array([2.0, 1.5, 80.0, 1.0])
        predicted_cov = jnp.eye(4) * 0.2
        confidence_levels = [0.5, 0.8, 0.95]
        
        intervals = propagator.calculate_prediction_intervals(
            predicted_mean, predicted_cov, confidence_levels
        )
        
        # Check all intervals were calculated
        assert len(intervals) == len(confidence_levels)
        
        for conf_level in confidence_levels:
            assert conf_level in intervals
            interval = intervals[conf_level]
            
            # Check interval properties
            assert interval.confidence_level == conf_level
            assert jnp.all(interval.lower_bound <= predicted_mean)
            assert jnp.all(interval.upper_bound >= predicted_mean)
            assert jnp.all(interval.width > 0)
        
        # Check that higher confidence levels have wider intervals
        assert jnp.all(intervals[0.95].width >= intervals[0.8].width)
        assert jnp.all(intervals[0.8].width >= intervals[0.5].width)


class TestSeasonalityHandler:
    """Test seasonality handling functionality."""
    
    def test_seasonality_handler_initialization(self, prediction_config):
        """Test seasonality handler initialization."""
        handler = SeasonalityHandler(prediction_config)
        
        assert handler.config == prediction_config
        assert "early" in handler.season_phases
        assert "mid" in handler.season_phases
        assert "late" in handler.season_phases
        assert "winter_break" in handler.season_phases
    
    def test_get_seasonal_adjustments_early_season(self, prediction_config):
        """Test seasonal adjustments for early season."""
        handler = SeasonalityHandler(prediction_config)
        
        adjustments = handler.get_seasonal_adjustments(
            current_gameweek=5,
            gameweeks_ahead=3,
            season="2024"
        )
        
        # Should have early season adjustments
        assert "early_season_uncertainty" in adjustments
        assert adjustments["early_season_uncertainty"] > 0
    
    def test_get_seasonal_adjustments_late_season(self, prediction_config):
        """Test seasonal adjustments for late season."""
        handler = SeasonalityHandler(prediction_config)
        
        adjustments = handler.get_seasonal_adjustments(
            current_gameweek=30,
            gameweeks_ahead=3,
            season="2024"
        )
        
        # Should have late season adjustments
        assert "motivation_factor" in adjustments
        assert adjustments["motivation_factor"] > 0
    
    def test_apply_seasonal_adjustments(self, prediction_config):
        """Test application of seasonal adjustments."""
        handler = SeasonalityHandler(prediction_config)
        
        predicted_state = jnp.array([0.5, 0.6, 0.7, 0.1])
        predicted_cov = jnp.eye(4) * 0.1
        adjustments = {"motivation_factor": 0.1, "consistency_boost": 0.05}
        
        adjusted_state, adjusted_cov = handler.apply_seasonal_adjustments(
            predicted_state, predicted_cov, adjustments
        )
        
        # State should be adjusted
        assert not jnp.array_equal(adjusted_state, predicted_state)
        
        # Covariance might be adjusted too
        # (depending on uncertainty-related adjustments)
    
    def test_seasonality_disabled(self, prediction_config):
        """Test behavior when seasonality is disabled."""
        prediction_config.enable_seasonality = False
        handler = SeasonalityHandler(prediction_config)
        
        adjustments = handler.get_seasonal_adjustments(5, 3, "2024")
        
        # Should return empty adjustments
        assert adjustments == {}


class TestFixtureAwarePredictor:
    """Test fixture-aware prediction functionality."""
    
    def test_fixture_aware_predictor_initialization(self, prediction_config):
        """Test fixture-aware predictor initialization."""
        predictor = FixtureAwarePredictor(prediction_config)
        
        assert predictor.config == prediction_config
        assert isinstance(predictor.opponent_strengths, dict)
        assert isinstance(predictor.fixture_cache, dict)
    
    def test_get_fixture_adjustments(self, prediction_config):
        """Test fixture adjustments calculation."""
        predictor = FixtureAwarePredictor(prediction_config)
        
        # Mock the fixture data method
        mock_fixtures = [
            {"gameweek": 16, "opponent_strength": 0.8, "is_home": True, "games_in_week": 1},
            {"gameweek": 17, "opponent_strength": 0.3, "is_home": False, "games_in_week": 1},
            {"gameweek": 18, "opponent_strength": 0.6, "is_home": True, "games_in_week": 2}
        ]
        
        with patch.object(predictor, '_get_upcoming_fixtures', return_value=mock_fixtures):
            adjustments = predictor.get_fixture_adjustments(123, 15, 3, "2024")
        
        # Should have fixture-related adjustments
        assert "opponent_difficulty" in adjustments
        assert "home_advantage" in adjustments or "away_disadvantage" in adjustments
        assert "double_gameweek_bonus" in adjustments
    
    def test_apply_fixture_adjustments(self, prediction_config):
        """Test application of fixture adjustments."""
        predictor = FixtureAwarePredictor(prediction_config)
        
        predicted_performance = jnp.array([2.0, 1.5, 80.0, 1.0])
        performance_cov = jnp.eye(4) * 0.1
        adjustments = {
            "opponent_difficulty": -0.1,
            "home_advantage": 0.05,
            "double_gameweek_bonus": 0.2
        }
        
        adjusted_performance, adjusted_cov = predictor.apply_fixture_adjustments(
            predicted_performance, performance_cov, adjustments
        )
        
        # Performance should be adjusted
        assert not jnp.array_equal(adjusted_performance, predicted_performance)
    
    def test_fixture_awareness_disabled(self, prediction_config):
        """Test behavior when fixture awareness is disabled."""
        prediction_config.enable_fixture_awareness = False
        predictor = FixtureAwarePredictor(prediction_config)
        
        adjustments = predictor.get_fixture_adjustments(123, 15, 3, "2024")
        
        # Should return empty adjustments
        assert adjustments == {}


class TestPredictionValidator:
    """Test prediction validation functionality."""
    
    def test_prediction_validator_initialization(self, prediction_config):
        """Test prediction validator initialization."""
        validator = PredictionValidator(prediction_config)
        
        assert validator.config == prediction_config
        assert isinstance(validator.validation_history, list)
    
    def test_validate_predictions_insufficient_samples(self, prediction_config):
        """Test validation with insufficient samples."""
        validator = PredictionValidator(prediction_config)
        
        # Too few samples
        predictions = [Mock() for _ in range(5)]
        actuals = [np.random.randn(4) for _ in range(5)]
        player_ids = list(range(5))
        
        with pytest.raises(ValueError):
            validator.validate_predictions(predictions, actuals, player_ids)
    
    def test_validate_predictions_basic(self, prediction_config, sample_prediction_result):
        """Test basic prediction validation."""
        validator = PredictionValidator(prediction_config)
        
        # Create mock predictions with intervals
        predictions = []
        actuals = []
        
        for i in range(20):
            pred = Mock(spec=PredictionResult)
            pred.expected_performance = np.array([2.0 + 0.1*i, 1.5, 80.0, 1.0])
            pred.prediction_intervals = {
                0.8: PredictionInterval(
                    confidence_level=0.8,
                    lower_bound=np.array([1.5 + 0.1*i, 1.0, 70.0, 0.5]),
                    upper_bound=np.array([2.5 + 0.1*i, 2.0, 90.0, 1.5]),
                    width=np.array([1.0, 1.0, 20.0, 1.0])
                )
            }
            pred.get_interval = lambda conf: pred.prediction_intervals.get(conf)
            pred.state_covariance = np.eye(4) * 0.1
            
            predictions.append(pred)
            actuals.append(np.array([2.0 + 0.1*i + np.random.normal(0, 0.1), 1.5, 80.0, 1.0]))
        
        player_ids = list(range(20))
        
        results = validator.validate_predictions(predictions, actuals, player_ids)
        
        # Check validation results structure
        assert "num_samples" in results
        assert "coverage_tests" in results
        assert "calibration_metrics" in results
        assert "skill_scores" in results
        assert "error_analysis" in results
        
        assert results["num_samples"] == 20
    
    def test_test_prediction_interval_coverage(self, prediction_config):
        """Test prediction interval coverage testing."""
        validator = PredictionValidator(prediction_config)
        
        # Create mock data where 80% of actuals fall within 80% intervals
        predictions = []
        actuals = []
        
        for i in range(100):
            pred = Mock()
            lower = np.array([1.0, 1.0, 70.0, 0.5])
            upper = np.array([3.0, 2.0, 90.0, 1.5])
            
            pred.prediction_intervals = {
                0.8: PredictionInterval(0.8, lower, upper, upper - lower)
            }
            pred.get_interval = lambda conf: pred.prediction_intervals.get(conf)
            
            predictions.append(pred)
            
            # 80% of the time, generate actuals within the interval
            if i < 80:
                actual = lower + np.random.random(4) * (upper - lower)
            else:
                actual = upper + np.random.random(4)  # Outside interval
            
            actuals.append(actual)
        
        coverage_results = validator._test_prediction_interval_coverage(predictions, actuals)
        
        # Coverage should be close to nominal level
        assert "coverage_0.8" in coverage_results
        assert abs(coverage_results["coverage_0.8"] - 0.8) < 0.1  # Within 10%


class TestStatePredictor:
    """Test main state predictor functionality."""
    
    def test_state_predictor_initialization(self, prediction_config, mock_kalman_model):
        """Test state predictor initialization."""
        predictor = StatePredictor(
            kalman_model=mock_kalman_model,
            config=prediction_config,
            enable_fixture_awareness=True,
            enable_seasonality=True
        )
        
        assert predictor.kalman_model == mock_kalman_model
        assert predictor.config == prediction_config
        assert isinstance(predictor.uncertainty_propagator, UncertaintyPropagator)
        assert isinstance(predictor.seasonality_handler, SeasonalityHandler)
        assert predictor.fixture_predictor is not None
        assert isinstance(predictor.validator, PredictionValidator)
    
    def test_state_predictor_initialization_no_filters(self, prediction_config):
        """Test state predictor initialization with invalid kalman model."""
        mock_model = Mock()
        mock_model.filters = {}  # No filters
        
        with pytest.raises(ValueError):
            StatePredictor(mock_model, prediction_config)
    
    def test_predict_multi_step_basic(self, prediction_config, mock_kalman_model):
        """Test basic multi-step prediction."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        result = predictor.predict_multi_step(
            player_id=123,
            gameweeks_ahead=3,
            confidence_levels=[0.8, 0.95]
        )
        
        # Check result structure
        assert isinstance(result, PredictionResult)
        assert result.player_id == 123
        assert result.gameweeks_ahead == 3
        assert len(result.prediction_intervals) == 2  # Two confidence levels
        assert 0.8 in result.prediction_intervals
        assert 0.95 in result.prediction_intervals
    
    def test_predict_multi_step_invalid_player(self, prediction_config, mock_kalman_model):
        """Test prediction for invalid player ID."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        with pytest.raises(ValueError):
            predictor.predict_multi_step(player_id=999, gameweeks_ahead=3)
    
    def test_predict_multi_step_invalid_gameweeks(self, prediction_config, mock_kalman_model):
        """Test prediction with invalid gameweeks ahead."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        with pytest.raises(ValueError):
            predictor.predict_multi_step(player_id=123, gameweeks_ahead=0)
        
        with pytest.raises(ValueError):
            predictor.predict_multi_step(player_id=123, gameweeks_ahead=20)  # Too many
    
    def test_predict_scenario(self, prediction_config, mock_kalman_model):
        """Test scenario-based prediction."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        scenario = {
            "injury_probability": 0.3,
            "rotation_risk": 0.2,
            "new_signing": True
        }
        
        result = predictor.predict_scenario(123, scenario, gameweeks_ahead=3)
        
        assert isinstance(result, PredictionResult)
        assert result.player_id == 123
    
    def test_get_predictions_multiple_players(self, prediction_config, mock_kalman_model):
        """Test predictions for multiple players."""
        # Add another player to mock model
        player_state_456 = PlayerState(
            player_id=456,
            state_mean=jnp.array([0.4, 0.5, 0.6, 0.0]),
            state_cov=jnp.eye(4) * 0.1,
            gameweek=15,
            season="2024",
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        mock_kalman_model.player_states[456] = player_state_456
        
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        results = predictor.get_predictions([123, 456], gameweeks_ahead=2)
        
        assert len(results) == 2
        assert 123 in results
        assert 456 in results
        assert all(isinstance(result, PredictionResult) for result in results.values())
    
    def test_compare_players(self, prediction_config, mock_kalman_model):
        """Test player comparison functionality."""
        # Add another player
        player_state_456 = PlayerState(
            player_id=456,
            state_mean=jnp.array([0.4, 0.5, 0.6, 0.0]),
            state_cov=jnp.eye(4) * 0.1,
            gameweek=15,
            season="2024",
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        mock_kalman_model.player_states[456] = player_state_456
        
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        comparison = predictor.compare_players([123, 456], metrics=["expected_points"])
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "player_id" in comparison.columns
        assert "expected_points" in comparison.columns
    
    def test_clear_cache(self, prediction_config, mock_kalman_model):
        """Test cache clearing functionality."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Make a prediction to populate cache
        predictor.predict_multi_step(123, 3)
        assert len(predictor.prediction_cache) > 0
        
        # Clear cache
        predictor.clear_cache()
        assert len(predictor.prediction_cache) == 0
    
    def test_get_diagnostics(self, prediction_config, mock_kalman_model):
        """Test diagnostics retrieval."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        diagnostics = predictor.get_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert "config" in diagnostics
        assert "cache_size" in diagnostics
        assert "model_diagnostics" in diagnostics


class TestPredictivePlayerModel:
    """Test PredictivePlayerModel functionality."""
    
    def test_predictive_model_initialization(self, filter_config, prediction_config):
        """Test PredictivePlayerModel initialization."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config,
            enable_fixture_awareness=True,
            enable_seasonality=True
        )
        
        assert model.prediction_config == prediction_config
        assert model.enable_fixture_awareness is True
        assert model.enable_seasonality is True
        assert model.state_predictor is None  # Not fitted yet
    
    def test_predictive_model_fit(self, filter_config, prediction_config):
        """Test PredictivePlayerModel fitting."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock training data
        training_data = {
            "player_ids": [123, 456],
            "features": np.random.randn(2, 20, 4),  # 2 players, 20 gameweeks, 4 features
            "gameweeks": np.arange(1, 21)
        }
        
        # Mock the base fit method
        with patch.object(KalmanPlayerModel, 'fit'):
            with patch.object(model, '_validate_model_performance'):
                model.fit(training_data, season="2024", validate=True)
        
        assert model.is_fitted is True
        assert model.state_predictor is not None
    
    def test_predict_player_performance_not_fitted(self, filter_config, prediction_config):
        """Test prediction before model is fitted."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        with pytest.raises(PredictiveModelError):
            model.predict_player_performance(123, 3)
    
    def test_predict_player_performance_fitted(self, filter_config, prediction_config, mock_kalman_model):
        """Test prediction after model is fitted."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock fitted state
        model.is_fitted = True
        model.state_predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Mock the predict_multi_step method
        mock_result = PredictionResult(
            player_id=123,
            gameweeks_ahead=3,
            prediction_date=datetime.now(timezone.utc).isoformat(),
            expected_state=jnp.array([0.6, 0.7, 0.8, 0.2]),
            state_covariance=jnp.eye(4) * 0.15,
            expected_performance=jnp.array([2.5, 1.8, 80.0, 1.2]),
            performance_variance=jnp.array([0.5, 0.3, 100.0, 0.8]),
            prediction_intervals={},
            risk_metrics=None
        )
        
        with patch.object(model.state_predictor, 'predict_multi_step', return_value=mock_result):
            result = model.predict_player_performance(123, 3)
        
        assert isinstance(result, PredictionResult)
        assert result.player_id == 123
        assert result.gameweeks_ahead == 3
    
    def test_get_expected_points(self, filter_config, prediction_config, mock_kalman_model):
        """Test expected points calculation for optimization."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock fitted state
        model.is_fitted = True
        model.state_predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Mock predict_batch method
        mock_predictions = {
            123: Mock(spec=PredictionResult),
            456: Mock(spec=PredictionResult)
        }
        mock_predictions[123].get_expected_points.return_value = 5.5
        mock_predictions[456].get_expected_points.return_value = 4.2
        
        with patch.object(model, 'predict_batch', return_value=mock_predictions):
            expected_points = model.get_expected_points([123, 456], gameweeks=3)
        
        assert expected_points[123] == 5.5
        assert expected_points[456] == 4.2
    
    def test_get_risk_adjusted_points(self, filter_config, prediction_config, mock_kalman_model):
        """Test risk-adjusted points calculation."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock fitted state
        model.is_fitted = True
        model.state_predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Mock predictions with risk metrics
        mock_risk_metrics = Mock(spec=RiskMetrics)
        mock_risk_metrics.sharpe_ratio = 0.8
        mock_risk_metrics.value_at_risk = 2.0
        mock_risk_metrics.conditional_var = 2.5
        
        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.get_expected_points.return_value = 5.0
        mock_prediction.risk_metrics = mock_risk_metrics
        
        mock_predictions = {123: mock_prediction}
        
        with patch.object(model, 'predict_batch', return_value=mock_predictions):
            risk_adjusted = model.get_risk_adjusted_points([123], risk_aversion=0.5, risk_measure="sharpe")
        
        assert 123 in risk_adjusted
        # Should be adjusted based on Sharpe ratio
        assert risk_adjusted[123] != 5.0
    
    def test_scenario_analysis(self, filter_config, prediction_config, mock_kalman_model):
        """Test scenario analysis functionality."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock fitted state
        model.is_fitted = True
        model.state_predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        scenarios = {
            "base_case": {},
            "injury_risk": {"injury_probability": 0.3},
            "rotation_risk": {"rotation_risk": 0.4}
        }
        
        # Mock scenario_analysis function
        with patch('airsenal.framework.state_prediction.scenario_analysis') as mock_scenario:
            mock_scenario.return_value = {"base_case": Mock(), "injury_risk": Mock()}
            
            results = model.scenario_analysis(123, scenarios)
        
        assert isinstance(results, dict)
        mock_scenario.assert_called_once()
    
    def test_get_top_performers(self, filter_config, prediction_config, mock_kalman_model):
        """Test top performers identification."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Mock fitted state with multiple players
        model.is_fitted = True
        model.player_states = {123: Mock(), 456: Mock(), 789: Mock()}
        model.state_predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Mock predictions
        mock_predictions = {}
        expected_points = [5.5, 4.2, 6.1]
        
        for i, (player_id, points) in enumerate(zip([123, 456, 789], expected_points)):
            mock_pred = Mock(spec=PredictionResult)
            mock_pred.get_expected_points.return_value = points
            mock_pred.expected_performance = np.array([2.0, 1.0, 85.0, 1.0])  # Minutes > 60
            mock_pred.risk_metrics = Mock()
            mock_pred.risk_metrics.sharpe_ratio = 0.8
            mock_predictions[player_id] = mock_pred
        
        with patch.object(model, 'predict_batch', return_value=mock_predictions):
            top_performers = model.get_top_performers(metric="expected_points", top_n=2)
        
        assert len(top_performers) == 2
        # Should be sorted by expected points (descending)
        assert top_performers[0][1] > top_performers[1][1]
    
    def test_save_and_load_model_state(self, filter_config, prediction_config, tmp_path):
        """Test model state persistence."""
        model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        
        # Set some state
        model.is_fitted = True
        model.prediction_performance["total_predictions"] = 10
        
        # Save model
        filepath = tmp_path / "test_model.pkl"
        model.save_model_state(str(filepath))
        
        # Create new model and load state
        new_model = PredictivePlayerModel(
            filter_config=filter_config,
            prediction_config=prediction_config
        )
        new_model.load_model_state(str(filepath))
        
        assert new_model.is_fitted is True
        assert new_model.prediction_performance["total_predictions"] == 10


class TestModelEnsemble:
    """Test ModelEnsemble functionality."""
    
    def test_ensemble_initialization(self, filter_config, prediction_config):
        """Test ensemble initialization."""
        models = [
            Mock(spec=PredictivePlayerModel),
            Mock(spec=PredictivePlayerModel),
            Mock(spec=PredictivePlayerModel)
        ]
        
        ensemble = ModelEnsemble(models, weights=[0.4, 0.3, 0.3])
        
        assert len(ensemble.models) == 3
        assert ensemble.weights == [0.4, 0.3, 0.3]
        assert ensemble.combination_method == "weighted_average"
    
    def test_ensemble_initialization_errors(self):
        """Test ensemble initialization error cases."""
        # No models
        with pytest.raises(ValueError):
            ModelEnsemble([])
        
        # Mismatched weights
        models = [Mock(), Mock()]
        with pytest.raises(ValueError):
            ModelEnsemble(models, weights=[0.5, 0.3, 0.2])  # Too many weights
        
        # Weights don't sum to 1
        with pytest.raises(ValueError):
            ModelEnsemble(models, weights=[0.6, 0.6])
    
    def test_ensemble_prediction(self, filter_config, prediction_config):
        """Test ensemble prediction combination."""
        # Create mock models
        models = [Mock(spec=PredictivePlayerModel) for _ in range(3)]
        
        # Mock individual predictions
        mock_predictions = []
        for i in range(3):
            pred = Mock(spec=PredictionResult)
            pred.player_id = 123
            pred.gameweeks_ahead = 3
            pred.expected_state = jnp.array([0.5 + i*0.1, 0.6, 0.7, 0.1])
            pred.state_covariance = jnp.eye(4) * (0.1 + i*0.02)
            pred.expected_performance = jnp.array([2.0 + i*0.5, 1.5, 80.0, 1.0])
            pred.performance_variance = jnp.array([0.5, 0.3, 100.0, 0.8])
            pred.risk_metrics = Mock()
            pred.risk_metrics.expected_value = 5.0 + i
            pred.risk_metrics.variance = 1.0
            pred.risk_metrics.value_at_risk = 2.0
            pred.risk_metrics.conditional_var = 2.5
            pred.risk_metrics.sharpe_ratio = 0.8
            pred.risk_metrics.max_drawdown = -1.0
            pred.risk_metrics.downside_deviation = 0.5
            pred.fixture_adjustments = {}
            pred.seasonality_factors = {}
            
            mock_predictions.append(pred)
            models[i].predict_player_performance.return_value = pred
        
        ensemble = ModelEnsemble(models, weights=[0.5, 0.3, 0.2])
        
        result = ensemble.predict_player_performance(123, 3)
        
        assert isinstance(result, PredictionResult)
        assert result.player_id == 123
        assert result.gameweeks_ahead == 3
        
        # Check that result is combination of individual predictions
        # (exact values depend on weighted averaging implementation)
        assert result.risk_metrics is not None


class TestAPIFunctions:
    """Test convenience API functions."""
    
    def test_get_predictions_function(self, prediction_config, mock_kalman_model):
        """Test get_predictions convenience function."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        with patch.object(predictor, 'predict_multi_step') as mock_predict:
            mock_predict.return_value = Mock(spec=PredictionResult)
            
            result = get_predictions(123, 3, predictor)
        
        assert mock_predict.called
        mock_predict.assert_called_with(123, 3)
    
    def test_get_prediction_intervals_function(self, prediction_config, mock_kalman_model):
        """Test get_prediction_intervals convenience function."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        mock_result = Mock(spec=PredictionResult)
        mock_intervals = {0.8: Mock(spec=PredictionInterval), 0.95: Mock(spec=PredictionInterval)}
        mock_result.prediction_intervals = mock_intervals
        
        with patch.object(predictor, 'predict_multi_step', return_value=mock_result):
            intervals = get_prediction_intervals(123, [0.8, 0.95], predictor)
        
        assert intervals == mock_intervals
    
    def test_get_risk_metrics_function(self, prediction_config, mock_kalman_model):
        """Test get_risk_metrics convenience function."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        mock_risk_metrics = Mock(spec=RiskMetrics)
        mock_risk_metrics.value_at_risk = 2.0
        mock_risk_metrics.conditional_var = 2.5
        mock_risk_metrics.sharpe_ratio = 0.8
        mock_risk_metrics.max_drawdown = -1.0
        
        mock_result = Mock(spec=PredictionResult)
        mock_result.risk_metrics = mock_risk_metrics
        
        with patch.object(predictor, 'predict_multi_step', return_value=mock_result):
            risk_data = get_risk_metrics(123, ["var", "cvar", "sharpe"], predictor)
        
        assert "value_at_risk" in risk_data
        assert "conditional_var" in risk_data
        assert "sharpe_ratio" in risk_data
        assert risk_data["value_at_risk"] == 2.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_state_prediction_error(self):
        """Test StatePredictionError exception."""
        with pytest.raises(StatePredictionError):
            raise StatePredictionError("Test error")
    
    def test_predictive_model_error(self):
        """Test PredictiveModelError exception."""
        with pytest.raises(PredictiveModelError):
            raise PredictiveModelError("Test error")
    
    def test_prediction_with_nan_observations(self, prediction_config, mock_kalman_model):
        """Test prediction handling with NaN observations."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # This would typically be tested in integration with real Kalman model
        # that handles NaN observations gracefully
        pass
    
    def test_prediction_with_invalid_covariance(self, prediction_config, mock_kalman_model):
        """Test prediction handling with invalid covariance matrices."""
        # This would test numerical stability features
        pass


class TestPerformanceCharacteristics:
    """Test performance and efficiency characteristics."""
    
    def test_prediction_caching(self, prediction_config, mock_kalman_model):
        """Test that predictions are properly cached."""
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # First prediction
        result1 = predictor.predict_multi_step(123, 3, current_gameweek=15)
        assert len(predictor.prediction_cache) == 1
        
        # Second identical prediction should use cache
        result2 = predictor.predict_multi_step(123, 3, current_gameweek=15)
        assert len(predictor.prediction_cache) == 1
        assert result1 is result2  # Same object from cache
    
    def test_batch_prediction_efficiency(self, prediction_config, mock_kalman_model):
        """Test that batch predictions are efficient."""
        # Add multiple players to mock model
        for player_id in [123, 456, 789]:
            player_state = PlayerState(
                player_id=player_id,
                state_mean=jnp.array([0.5, 0.6, 0.7, 0.1]),
                state_cov=jnp.eye(4) * 0.1,
                gameweek=15,
                season="2024",
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            mock_kalman_model.player_states[player_id] = player_state
        
        predictor = StatePredictor(mock_kalman_model, prediction_config)
        
        # Batch prediction should work
        results = predictor.get_predictions([123, 456, 789], gameweeks_ahead=2)
        
        assert len(results) == 3
        assert all(isinstance(result, PredictionResult) for result in results.values())


# Integration tests
class TestIntegration:
    """Integration tests for complete prediction pipeline."""
    
    def test_end_to_end_prediction_pipeline(self, filter_config, prediction_config):
        """Test complete end-to-end prediction pipeline."""
        # This would test the full pipeline from raw data to predictions
        # with real Kalman model integration
        pass
    
    def test_optimization_integration(self, filter_config, prediction_config):
        """Test integration with optimization algorithms."""
        # This would test the API functions used by optimization
        pass
    
    def test_database_integration(self, filter_config, prediction_config):
        """Test integration with database for fixture data."""
        # This would test database queries for fixture information
        pass


if __name__ == "__main__":
    pytest.main([__file__])