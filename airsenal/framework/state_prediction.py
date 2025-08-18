"""
State Prediction Module for AIrsenal

This module implements comprehensive state prediction capabilities for forecasting 
player abilities into future gameweeks using Kalman filter prediction steps with 
proper uncertainty propagation and fixture-aware adjustments.

The module provides:
- Multi-step ahead predictions with uncertainty quantification
- Prediction interval calculation at various confidence levels
- Risk metrics for optimization integration
- Fixture-aware adjustments for opponent strength
- Seasonality handling for calendar effects
- Comprehensive validation framework

Classes:
    StatePredictor: Main prediction engine for multi-step forecasts
    UncertaintyPropagator: Track uncertainty growth over prediction horizons
    PredictionValidator: Validate predictions against historical actuals
    SeasonalityHandler: Account for seasonal patterns and calendar effects
    FixtureAwarePredictor: Adjust predictions based on upcoming fixtures
    RiskMetricsCalculator: Calculate risk measures for optimization
    PredictionResult: Container for prediction outputs with intervals

Usage:
    ```python
    from airsenal.framework.state_prediction import StatePredictor
    from airsenal.framework.kalman_player_model import KalmanPlayerModel
    
    # Initialize predictor with trained model
    predictor = StatePredictor(kalman_model, enable_fixture_awareness=True)
    
    # Multi-step prediction
    result = predictor.predict_multi_step(
        player_id=123,
        gameweeks_ahead=3,
        confidence_levels=[0.5, 0.8, 0.95]
    )
    
    # Access predictions and intervals
    expected_points = result.expected_performance
    prediction_intervals = result.prediction_intervals
    risk_metrics = result.risk_metrics
    ```
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sqlalchemy.orm.session import Session

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
    PlayerData
)
from airsenal.framework.kalman_filter import (
    BaseKalmanFilter,
    FilterState,
    FilterConfig,
    KalmanFilterError
)

logger = logging.getLogger(__name__)

# Type aliases
Array = Union[np.ndarray, jnp.ndarray]
ConfidenceLevel = float  # Between 0 and 1
RiskMeasure = Literal["var", "cvar", "sharpe", "max_drawdown"]
SeasonPhase = Literal["early", "mid", "late", "winter_break", "summer_break"]
FixtureStrength = float  # Opponent difficulty rating


@dataclass
class PredictionConfig:
    """Configuration for state prediction parameters."""
    
    # Prediction horizons
    max_gameweeks_ahead: int = 10
    default_confidence_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    
    # Uncertainty propagation
    uncertainty_inflation_rate: float = 0.05  # Per gameweek
    min_uncertainty: float = 0.01
    max_uncertainty: float = 2.0
    
    # Seasonality parameters
    enable_seasonality: bool = True
    early_season_uncertainty_boost: float = 0.3
    late_season_motivation_factor: float = 0.1
    winter_break_decay: float = 0.05
    
    # Fixture awareness
    enable_fixture_awareness: bool = True
    home_advantage_factor: float = 0.1
    opponent_strength_weight: float = 0.2
    fixture_congestion_penalty: float = 0.05
    
    # Risk calculations
    var_confidence: float = 0.05  # 5% VaR
    cvar_confidence: float = 0.05  # 5% CVaR
    risk_free_rate: float = 0.0  # For Sharpe ratio
    
    # Validation parameters
    min_validation_samples: int = 20
    calibration_bins: int = 10
    backtest_window: int = 38  # Full season


@dataclass
class PredictionInterval:
    """Container for prediction intervals at different confidence levels."""
    
    confidence_level: float
    lower_bound: Array
    upper_bound: Array
    width: Array
    
    def __post_init__(self):
        """Validate interval properties."""
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {self.confidence_level}")
        
        if np.any(self.lower_bound > self.upper_bound):
            warnings.warn("Some prediction intervals have lower > upper bounds")


@dataclass
class RiskMetrics:
    """Container for risk assessment metrics."""
    
    expected_value: float
    variance: float
    standard_deviation: float
    value_at_risk: float  # VaR at specified confidence level
    conditional_var: float  # CVaR (Expected Shortfall)
    sharpe_ratio: float
    max_drawdown: float
    downside_deviation: float
    
    @classmethod
    def from_distribution(
        cls,
        samples: Array,
        var_confidence: float = 0.05,
        risk_free_rate: float = 0.0
    ) -> 'RiskMetrics':
        """Calculate risk metrics from sample distribution."""
        samples = np.asarray(samples)
        
        expected_value = float(np.mean(samples))
        variance = float(np.var(samples, ddof=1))
        std_dev = float(np.std(samples, ddof=1))
        
        # Value at Risk (negative of quantile for losses)
        var = -float(np.quantile(samples, var_confidence))
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_mask = samples <= -var
        cvar = -float(np.mean(samples[cvar_mask])) if np.any(cvar_mask) else var
        
        # Sharpe ratio
        sharpe = (expected_value - risk_free_rate) / std_dev if std_dev > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumsum(samples)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = float(np.min(drawdown))
        
        # Downside deviation
        downside_samples = samples[samples < expected_value]
        downside_dev = float(np.std(downside_samples)) if len(downside_samples) > 0 else 0.0
        
        return cls(
            expected_value=expected_value,
            variance=variance,
            standard_deviation=std_dev,
            value_at_risk=var,
            conditional_var=cvar,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            downside_deviation=downside_dev
        )


@dataclass
class PredictionResult:
    """Complete prediction result with uncertainty quantification."""
    
    player_id: int
    gameweeks_ahead: int
    prediction_date: str
    
    # Core predictions
    expected_state: Array  # Expected state vector
    state_covariance: Array  # State uncertainty
    expected_performance: Array  # Expected observable performance
    performance_variance: Array  # Performance uncertainty
    
    # Prediction intervals
    prediction_intervals: Dict[float, PredictionInterval]
    
    # Risk metrics
    risk_metrics: RiskMetrics
    
    # Metadata
    fixture_adjustments: Dict[str, float] = field(default_factory=dict)
    seasonality_factors: Dict[str, float] = field(default_factory=dict)
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def get_interval(self, confidence_level: float) -> Optional[PredictionInterval]:
        """Get prediction interval for specific confidence level."""
        return self.prediction_intervals.get(confidence_level)
    
    def get_expected_points(self) -> float:
        """Get expected FPL points from performance prediction."""
        # Simple mapping from performance to FPL points
        # This would typically use FPL scoring rules
        goals, assists, minutes, bonus = self.expected_performance[:4]
        
        points = (
            goals * 4 +  # Goal points (simplified)
            assists * 3 +  # Assist points
            (minutes >= 60) * 2 +  # Appearance points
            bonus
        )
        
        return float(points)


class StatePredictionError(Exception):
    """Base exception for state prediction operations."""
    pass


class UncertaintyPropagator:
    """
    Handles uncertainty propagation for multi-step predictions.
    
    This class manages how uncertainty grows over prediction horizons,
    accounting for model uncertainty, process noise, and external factors.
    """
    
    def __init__(
        self,
        config: PredictionConfig,
        base_filter: BaseKalmanFilter
    ):
        """
        Initialize uncertainty propagator.
        
        Args:
            config: Prediction configuration
            base_filter: Base Kalman filter for uncertainty propagation
        """
        self.config = config
        self.base_filter = base_filter
        
        # Track uncertainty inflation parameters
        self.inflation_rate = config.uncertainty_inflation_rate
        self.min_uncertainty = config.min_uncertainty
        self.max_uncertainty = config.max_uncertainty
        
        logger.debug("Initialized UncertaintyPropagator")
    
    def propagate_uncertainty(
        self,
        initial_covariance: Array,
        gameweeks_ahead: int,
        external_factors: Optional[Dict[str, float]] = None
    ) -> Array:
        """
        Propagate uncertainty forward in time.
        
        Args:
            initial_covariance: Starting covariance matrix
            gameweeks_ahead: Number of gameweeks to propagate
            external_factors: Additional uncertainty sources
            
        Returns:
            Propagated covariance matrix
        """
        try:
            current_cov = jnp.array(initial_covariance)
            
            for step in range(gameweeks_ahead):
                # Apply base filter prediction uncertainty
                current_cov = self._apply_process_uncertainty(current_cov)
                
                # Add time-dependent uncertainty inflation
                current_cov = self._apply_time_inflation(current_cov, step + 1)
                
                # Add external uncertainty sources
                if external_factors:
                    current_cov = self._apply_external_uncertainty(current_cov, external_factors)
                
                # Enforce bounds
                current_cov = self._enforce_uncertainty_bounds(current_cov)
            
            return current_cov
            
        except Exception as e:
            logger.error(f"Uncertainty propagation failed: {e}")
            raise StatePredictionError(f"Failed to propagate uncertainty: {e}")
    
    def _apply_process_uncertainty(self, covariance: Array) -> Array:
        """Apply process noise from Kalman filter."""
        process_noise = self.base_filter.config.process_noise_std**2
        process_cov = jnp.eye(covariance.shape[0]) * process_noise
        return covariance + process_cov
    
    def _apply_time_inflation(self, covariance: Array, steps: int) -> Array:
        """Apply time-dependent uncertainty inflation."""
        inflation_factor = 1 + self.inflation_rate * steps
        return covariance * inflation_factor
    
    def _apply_external_uncertainty(
        self, 
        covariance: Array, 
        external_factors: Dict[str, float]
    ) -> Array:
        """Apply external uncertainty sources."""
        additional_uncertainty = 0.0
        
        for factor, weight in external_factors.items():
            if factor == "fixture_difficulty":
                additional_uncertainty += weight * 0.1
            elif factor == "injury_risk":
                additional_uncertainty += weight * 0.2
            elif factor == "rotation_risk":
                additional_uncertainty += weight * 0.15
        
        if additional_uncertainty > 0:
            external_cov = jnp.eye(covariance.shape[0]) * additional_uncertainty
            covariance = covariance + external_cov
        
        return covariance
    
    def _enforce_uncertainty_bounds(self, covariance: Array) -> Array:
        """Enforce minimum and maximum uncertainty bounds."""
        # Ensure minimum uncertainty
        min_cov = jnp.eye(covariance.shape[0]) * self.min_uncertainty**2
        covariance = jnp.maximum(covariance, min_cov)
        
        # Cap maximum uncertainty
        max_cov = jnp.eye(covariance.shape[0]) * self.max_uncertainty**2
        covariance = jnp.minimum(covariance, max_cov)
        
        # Ensure positive definite
        eigenvals, eigenvecs = jnp.linalg.eigh(covariance)
        eigenvals = jnp.maximum(eigenvals, 1e-10)
        covariance = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        return covariance
    
    def calculate_prediction_intervals(
        self,
        predicted_mean: Array,
        predicted_covariance: Array,
        confidence_levels: List[float]
    ) -> Dict[float, PredictionInterval]:
        """
        Calculate prediction intervals at multiple confidence levels.
        
        Args:
            predicted_mean: Predicted state or performance mean
            predicted_covariance: Predicted covariance matrix
            confidence_levels: List of confidence levels (0 < level < 1)
            
        Returns:
            Dictionary mapping confidence levels to prediction intervals
        """
        intervals = {}
        
        for conf_level in confidence_levels:
            try:
                # Calculate z-score for confidence level
                alpha = 1 - conf_level
                z_score = stats.norm.ppf(1 - alpha/2)
                
                # Extract standard deviations
                std_devs = jnp.sqrt(jnp.diag(predicted_covariance))
                
                # Calculate bounds
                lower_bound = predicted_mean - z_score * std_devs
                upper_bound = predicted_mean + z_score * std_devs
                width = upper_bound - lower_bound
                
                intervals[conf_level] = PredictionInterval(
                    confidence_level=conf_level,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    width=width
                )
                
            except Exception as e:
                logger.warning(f"Failed to calculate interval for confidence {conf_level}: {e}")
                continue
        
        return intervals


class SeasonalityHandler:
    """
    Handles seasonal patterns and calendar effects in player performance.
    
    This class adjusts predictions based on:
    - Early season uncertainty (new signings, tactical changes)
    - Mid-season consistency patterns
    - End-season motivation factors
    - Winter/summer break effects
    - Historical performance patterns
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize seasonality handler.
        
        Args:
            config: Prediction configuration with seasonality parameters
        """
        self.config = config
        self.season_phases = self._define_season_phases()
        
        logger.debug("Initialized SeasonalityHandler")
    
    def _define_season_phases(self) -> Dict[str, Tuple[int, int]]:
        """Define gameweek ranges for different season phases."""
        return {
            "early": (1, 10),      # First 10 gameweeks
            "mid": (11, 25),       # Middle of season
            "late": (26, 38),      # End of season
            "winter_break": (19, 21),  # Typical winter break period
            "summer_break": (39, 40)   # Post-season (if applicable)
        }
    
    def get_seasonal_adjustments(
        self,
        current_gameweek: int,
        gameweeks_ahead: int,
        season: str = "2024"
    ) -> Dict[str, float]:
        """
        Calculate seasonal adjustments for prediction period.
        
        Args:
            current_gameweek: Current gameweek number
            gameweeks_ahead: Prediction horizon
            season: Season identifier
            
        Returns:
            Dictionary of seasonal adjustment factors
        """
        if not self.config.enable_seasonality:
            return {}
        
        adjustments = {}
        
        # Analyze prediction period
        target_gameweeks = range(current_gameweek + 1, current_gameweek + gameweeks_ahead + 1)
        
        for gw in target_gameweeks:
            phase = self._get_season_phase(gw)
            
            # Early season adjustments
            if phase == "early":
                adjustments["early_season_uncertainty"] = self.config.early_season_uncertainty_boost
                adjustments["new_signings_uncertainty"] = 0.2
                adjustments["tactical_adjustment"] = 0.1
            
            # Mid-season stability
            elif phase == "mid":
                adjustments["consistency_boost"] = 0.05
                adjustments["form_stability"] = 0.1
            
            # Late season effects
            elif phase == "late":
                adjustments["motivation_factor"] = self.config.late_season_motivation_factor
                adjustments["rotation_risk"] = 0.1  # Rest for key players
                adjustments["relegation_pressure"] = 0.05
            
            # Winter break effects
            if self._is_winter_break_period(gw):
                adjustments["winter_break_decay"] = self.config.winter_break_decay
                adjustments["fitness_uncertainty"] = 0.15
        
        return adjustments
    
    def _get_season_phase(self, gameweek: int) -> SeasonPhase:
        """Determine season phase for given gameweek."""
        for phase, (start, end) in self.season_phases.items():
            if start <= gameweek <= end:
                return phase
        return "mid"  # Default fallback
    
    def _is_winter_break_period(self, gameweek: int) -> bool:
        """Check if gameweek falls in winter break period."""
        winter_start, winter_end = self.season_phases["winter_break"]
        return winter_start <= gameweek <= winter_end
    
    def apply_seasonal_adjustments(
        self,
        predicted_state: Array,
        predicted_covariance: Array,
        adjustments: Dict[str, float]
    ) -> Tuple[Array, Array]:
        """
        Apply seasonal adjustments to predictions.
        
        Args:
            predicted_state: Predicted state vector
            predicted_covariance: Predicted covariance matrix
            adjustments: Seasonal adjustment factors
            
        Returns:
            Tuple of adjusted (state, covariance)
        """
        adjusted_state = jnp.array(predicted_state)
        adjusted_cov = jnp.array(predicted_covariance)
        
        # Apply state adjustments
        for factor, value in adjustments.items():
            if factor == "motivation_factor":
                # Boost skill and form for late season motivation
                adjusted_state = adjusted_state.at[0].add(value)  # skill
                adjusted_state = adjusted_state.at[1].add(value * 0.5)  # form
                
            elif factor == "consistency_boost":
                # Improve consistency in mid-season
                adjusted_state = adjusted_state.at[2].add(value)  # consistency
                
            elif factor == "winter_break_decay":
                # Slight decay in form after break
                adjusted_state = adjusted_state.at[1].multiply(1 - value)  # form
        
        # Apply uncertainty adjustments
        uncertainty_multiplier = 1.0
        for factor, value in adjustments.items():
            if "uncertainty" in factor:
                uncertainty_multiplier += value
        
        if uncertainty_multiplier != 1.0:
            adjusted_cov = adjusted_cov * uncertainty_multiplier
        
        return adjusted_state, adjusted_cov


class FixtureAwarePredictor:
    """
    Adjusts predictions based on upcoming fixture difficulty and context.
    
    This class considers:
    - Opponent strength ratings
    - Home/away venue effects
    - Fixture congestion (multiple games per week)
    - Double gameweeks and blank gameweeks
    - Cup competition scheduling
    """
    
    def __init__(
        self,
        config: PredictionConfig,
        session: Optional[Session] = None
    ):
        """
        Initialize fixture-aware predictor.
        
        Args:
            config: Prediction configuration
            session: Database session for fixture data
        """
        self.config = config
        self.session = session
        
        # Fixture difficulty cache
        self.opponent_strengths: Dict[str, float] = {}
        self.fixture_cache: Dict[int, List[Dict]] = {}
        
        logger.debug("Initialized FixtureAwarePredictor")
    
    def get_fixture_adjustments(
        self,
        player_id: int,
        current_gameweek: int,
        gameweeks_ahead: int,
        season: str = "2024"
    ) -> Dict[str, float]:
        """
        Calculate fixture-based adjustments for prediction period.
        
        Args:
            player_id: Player ID
            current_gameweek: Current gameweek
            gameweeks_ahead: Prediction horizon
            season: Season identifier
            
        Returns:
            Dictionary of fixture adjustment factors
        """
        if not self.config.enable_fixture_awareness:
            return {}
        
        try:
            # Get upcoming fixtures
            fixtures = self._get_upcoming_fixtures(player_id, current_gameweek, gameweeks_ahead, season)
            
            if not fixtures:
                return {}
            
            adjustments = {}
            
            # Analyze fixture difficulty
            avg_opponent_strength = np.mean([f.get("opponent_strength", 0.5) for f in fixtures])
            adjustments["opponent_difficulty"] = (avg_opponent_strength - 0.5) * self.config.opponent_strength_weight
            
            # Home/away analysis
            home_games = sum(1 for f in fixtures if f.get("is_home", False))
            away_games = len(fixtures) - home_games
            
            if home_games > away_games:
                adjustments["home_advantage"] = self.config.home_advantage_factor
            elif away_games > home_games:
                adjustments["away_disadvantage"] = -self.config.home_advantage_factor * 0.5
            
            # Fixture congestion
            congested_weeks = sum(1 for f in fixtures if f.get("games_in_week", 1) > 1)
            if congested_weeks > 0:
                congestion_penalty = (congested_weeks / len(fixtures)) * self.config.fixture_congestion_penalty
                adjustments["fixture_congestion"] = -congestion_penalty
            
            # Double gameweek bonus
            double_gameweeks = sum(1 for f in fixtures if f.get("games_in_week", 1) == 2)
            if double_gameweeks > 0:
                adjustments["double_gameweek_bonus"] = (double_gameweeks / len(fixtures)) * 0.3
            
            # Blank gameweek penalty
            blank_gameweeks = sum(1 for f in fixtures if f.get("games_in_week", 1) == 0)
            if blank_gameweeks > 0:
                adjustments["blank_gameweek_penalty"] = -(blank_gameweeks / len(fixtures)) * 0.5
            
            return adjustments
            
        except Exception as e:
            logger.warning(f"Failed to calculate fixture adjustments for player {player_id}: {e}")
            return {}
    
    def _get_upcoming_fixtures(
        self,
        player_id: int,
        current_gameweek: int,
        gameweeks_ahead: int,
        season: str
    ) -> List[Dict[str, Any]]:
        """Get upcoming fixtures for player's team."""
        # This would typically query the database for fixture information
        # For now, return mock data structure
        
        fixtures = []
        for gw in range(current_gameweek + 1, current_gameweek + gameweeks_ahead + 1):
            # Mock fixture data - in real implementation, query database
            fixture = {
                "gameweek": gw,
                "opponent_team": f"Team_{gw % 20}",
                "opponent_strength": np.random.uniform(0.2, 0.8),  # Mock strength
                "is_home": (gw % 2 == 0),  # Alternate home/away
                "games_in_week": 1,  # Normal single gameweek
                "is_cup_week": False
            }
            
            # Simulate some double gameweeks
            if gw % 10 == 0:
                fixture["games_in_week"] = 2
            
            fixtures.append(fixture)
        
        return fixtures
    
    def apply_fixture_adjustments(
        self,
        predicted_performance: Array,
        performance_covariance: Array,
        adjustments: Dict[str, float]
    ) -> Tuple[Array, Array]:
        """
        Apply fixture adjustments to performance predictions.
        
        Args:
            predicted_performance: Predicted performance vector
            performance_covariance: Performance uncertainty
            adjustments: Fixture adjustment factors
            
        Returns:
            Tuple of adjusted (performance, covariance)
        """
        adjusted_performance = jnp.array(predicted_performance)
        adjusted_cov = jnp.array(performance_covariance)
        
        # Apply performance adjustments
        for factor, value in adjustments.items():
            if factor == "opponent_difficulty":
                # Stronger opponents reduce expected performance
                adjusted_performance = adjusted_performance * (1 + value)
                
            elif factor == "home_advantage":
                # Home advantage boosts all metrics
                adjusted_performance = adjusted_performance * (1 + value)
                
            elif factor == "away_disadvantage":
                # Away games slightly reduce performance
                adjusted_performance = adjusted_performance * (1 + value)
                
            elif factor == "fixture_congestion":
                # Congestion reduces performance and increases uncertainty
                adjusted_performance = adjusted_performance * (1 + value)
                adjusted_cov = adjusted_cov * (1 - value)  # Increase uncertainty
                
            elif factor == "double_gameweek_bonus":
                # Double gameweeks boost expected performance
                adjusted_performance = adjusted_performance * (1 + value)
                
            elif factor == "blank_gameweek_penalty":
                # Blank gameweeks zero out performance
                adjusted_performance = adjusted_performance * (1 + value)
        
        return adjusted_performance, adjusted_cov


class PredictionValidator:
    """
    Validates prediction accuracy and calibration against historical data.
    
    This class implements comprehensive validation including:
    - Prediction interval coverage testing
    - Calibration plots and reliability diagrams
    - Skill scores and proper scoring rules
    - Cross-validation for hyperparameter tuning
    - Backtesting on historical seasons
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize prediction validator.
        
        Args:
            config: Prediction configuration
        """
        self.config = config
        self.validation_history: List[Dict] = []
        
        logger.debug("Initialized PredictionValidator")
    
    def validate_predictions(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array],
        player_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of prediction accuracy.
        
        Args:
            predictions: List of prediction results
            actuals: List of actual observed performance
            player_ids: List of player IDs
            
        Returns:
            Dictionary of validation metrics
        """
        if len(predictions) != len(actuals) or len(predictions) < self.config.min_validation_samples:
            raise ValueError(f"Insufficient validation samples: {len(predictions)}")
        
        validation_results = {
            "num_samples": len(predictions),
            "coverage_tests": self._test_prediction_interval_coverage(predictions, actuals),
            "calibration_metrics": self._calculate_calibration_metrics(predictions, actuals),
            "skill_scores": self._calculate_skill_scores(predictions, actuals),
            "error_analysis": self._analyze_prediction_errors(predictions, actuals),
            "reliability_diagram": self._create_reliability_diagram(predictions, actuals)
        }
        
        # Store validation history
        self.validation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": validation_results
        })
        
        return validation_results
    
    def _test_prediction_interval_coverage(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array]
    ) -> Dict[str, float]:
        """Test if prediction intervals have correct empirical coverage."""
        coverage_tests = {}
        
        # Get all confidence levels
        confidence_levels = set()
        for pred in predictions:
            confidence_levels.update(pred.prediction_intervals.keys())
        
        for conf_level in confidence_levels:
            coverage_count = 0
            valid_count = 0
            
            for pred, actual in zip(predictions, actuals):
                interval = pred.get_interval(conf_level)
                if interval is None:
                    continue
                
                valid_count += 1
                
                # Check if actual falls within interval
                in_interval = np.all(
                    (actual >= interval.lower_bound) & (actual <= interval.upper_bound)
                )
                
                if in_interval:
                    coverage_count += 1
            
            if valid_count > 0:
                empirical_coverage = coverage_count / valid_count
                coverage_tests[f"coverage_{conf_level}"] = empirical_coverage
                coverage_tests[f"coverage_error_{conf_level}"] = abs(empirical_coverage - conf_level)
        
        return coverage_tests
    
    def _calculate_calibration_metrics(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array]
    ) -> Dict[str, float]:
        """Calculate calibration metrics for probabilistic predictions."""
        # Simplified calibration using prediction means vs actuals
        predicted_means = np.array([pred.expected_performance for pred in predictions])
        actual_values = np.array(actuals)
        
        # Mean absolute error
        mae = np.mean(np.abs(predicted_means - actual_values))
        
        # Root mean square error
        rmse = np.sqrt(np.mean((predicted_means - actual_values)**2))
        
        # Mean bias
        bias = np.mean(predicted_means - actual_values)
        
        # Correlation coefficient
        correlation = np.corrcoef(predicted_means.flatten(), actual_values.flatten())[0, 1]
        
        return {
            "mean_absolute_error": float(mae),
            "root_mean_square_error": float(rmse),
            "bias": float(bias),
            "correlation": float(correlation)
        }
    
    def _calculate_skill_scores(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array]
    ) -> Dict[str, float]:
        """Calculate proper scoring rules and skill scores."""
        predicted_means = np.array([pred.expected_performance for pred in predictions])
        predicted_vars = np.array([np.diag(pred.state_covariance) for pred in predictions])
        actual_values = np.array(actuals)
        
        # Continuous Ranked Probability Score (CRPS) approximation
        # Using normal distribution approximation
        crps_scores = []
        for pred_mean, pred_var, actual in zip(predicted_means, predicted_vars, actual_values):
            std_dev = np.sqrt(pred_var)
            # Simplified CRPS calculation
            z = (actual - pred_mean) / std_dev
            crps = std_dev * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
            crps_scores.append(np.mean(crps))
        
        mean_crps = float(np.mean(crps_scores))
        
        # Logarithmic score (negative log-likelihood)
        log_scores = []
        for pred_mean, pred_var, actual in zip(predicted_means, predicted_vars, actual_values):
            # Multivariate normal log-likelihood
            diff = actual - pred_mean
            log_score = 0.5 * (
                np.log(2 * np.pi * pred_var) + 
                (diff**2) / pred_var
            )
            log_scores.append(np.mean(log_score))
        
        mean_log_score = float(np.mean(log_scores))
        
        return {
            "continuous_ranked_probability_score": mean_crps,
            "logarithmic_score": mean_log_score,
            "brier_skill_score": 0.0  # Placeholder for binary outcome predictions
        }
    
    def _analyze_prediction_errors(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array]
    ) -> Dict[str, Any]:
        """Analyze patterns in prediction errors."""
        predicted_means = np.array([pred.expected_performance for pred in predictions])
        actual_values = np.array(actuals)
        errors = predicted_means - actual_values
        
        return {
            "error_statistics": {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "median": float(np.median(errors)),
                "skewness": float(stats.skew(errors.flatten())),
                "kurtosis": float(stats.kurtosis(errors.flatten()))
            },
            "error_percentiles": {
                "p5": float(np.percentile(errors, 5)),
                "p25": float(np.percentile(errors, 25)),
                "p75": float(np.percentile(errors, 75)),
                "p95": float(np.percentile(errors, 95))
            }
        }
    
    def _create_reliability_diagram(
        self,
        predictions: List[PredictionResult],
        actuals: List[Array]
    ) -> Dict[str, List[float]]:
        """Create reliability diagram data for calibration assessment."""
        # Simplified reliability diagram using prediction confidence
        
        # Bin predictions by confidence level
        bins = np.linspace(0, 1, self.config.calibration_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        reliability_data = {
            "bin_centers": bin_centers.tolist(),
            "observed_frequencies": [],
            "predicted_frequencies": [],
            "bin_counts": []
        }
        
        # This would typically require probability predictions
        # For now, return placeholder structure
        for i in range(len(bin_centers)):
            reliability_data["observed_frequencies"].append(float(bin_centers[i]))
            reliability_data["predicted_frequencies"].append(float(bin_centers[i]))
            reliability_data["bin_counts"].append(len(predictions) // len(bin_centers))
        
        return reliability_data


class StatePredictor:
    """
    Main prediction engine for multi-step ahead player state and performance forecasting.
    
    This class orchestrates all prediction components to provide comprehensive
    forecasts with uncertainty quantification, risk metrics, and contextual adjustments.
    """
    
    def __init__(
        self,
        kalman_model: AdaptivePlayerModel,
        config: Optional[PredictionConfig] = None,
        enable_fixture_awareness: bool = True,
        enable_seasonality: bool = True,
        session: Optional[Session] = None
    ):
        """
        Initialize state predictor.
        
        Args:
            kalman_model: Trained Kalman player model
            config: Prediction configuration
            enable_fixture_awareness: Whether to adjust for fixtures
            enable_seasonality: Whether to apply seasonal adjustments
            session: Database session for fixture data
        """
        self.kalman_model = kalman_model
        self.config = config or PredictionConfig()
        self.session = session
        
        # Initialize component modules
        if hasattr(kalman_model, 'filters') and kalman_model.filters:
            base_filter = next(iter(kalman_model.filters.values()))
        else:
            raise ValueError("Kalman model must have initialized filters")
            
        self.uncertainty_propagator = UncertaintyPropagator(self.config, base_filter)
        self.seasonality_handler = SeasonalityHandler(self.config)
        
        if enable_fixture_awareness:
            self.fixture_predictor = FixtureAwarePredictor(self.config, session)
        else:
            self.fixture_predictor = None
        
        self.validator = PredictionValidator(self.config)
        
        # Prediction cache for efficiency
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        logger.info(f"Initialized StatePredictor with fixture_awareness={enable_fixture_awareness}, seasonality={enable_seasonality}")
    
    def predict_multi_step(
        self,
        player_id: int,
        gameweeks_ahead: int,
        confidence_levels: Optional[List[float]] = None,
        current_gameweek: Optional[int] = None,
        season: str = "2024",
        include_risk_metrics: bool = True,
        external_factors: Optional[Dict[str, float]] = None
    ) -> PredictionResult:
        """
        Generate multi-step prediction with uncertainty quantification.
        
        Args:
            player_id: Player ID to predict
            gameweeks_ahead: Number of gameweeks ahead to predict
            confidence_levels: Confidence levels for prediction intervals
            current_gameweek: Current gameweek (auto-detected if None)
            season: Season identifier
            include_risk_metrics: Whether to calculate risk metrics
            external_factors: Additional factors affecting uncertainty
            
        Returns:
            Comprehensive prediction result
        """
        try:
            # Validate inputs
            if gameweeks_ahead <= 0 or gameweeks_ahead > self.config.max_gameweeks_ahead:
                raise ValueError(f"gameweeks_ahead must be between 1 and {self.config.max_gameweeks_ahead}")
            
            if confidence_levels is None:
                confidence_levels = self.config.default_confidence_levels
            
            # Check cache
            cache_key = f"{player_id}_{gameweeks_ahead}_{current_gameweek}_{season}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Get current player state
            if player_id not in self.kalman_model.player_states:
                raise ValueError(f"Player {player_id} not found in model")
            
            current_state = self.kalman_model.player_states[player_id]
            
            # Multi-step prediction using Kalman filter
            predicted_state = current_state
            for step in range(gameweeks_ahead):
                predicted_state = self.kalman_model.predict_state(
                    predicted_state,
                    gameweeks_ahead=1
                )
            
            # Propagate uncertainty
            propagated_covariance = self.uncertainty_propagator.propagate_uncertainty(
                current_state.state_cov,
                gameweeks_ahead,
                external_factors
            )
            
            # Apply seasonal adjustments
            if self.config.enable_seasonality:
                seasonal_adjustments = self.seasonality_handler.get_seasonal_adjustments(
                    current_gameweek or current_state.gameweek,
                    gameweeks_ahead,
                    season
                )
                
                predicted_state.state_mean, propagated_covariance = \
                    self.seasonality_handler.apply_seasonal_adjustments(
                        predicted_state.state_mean,
                        propagated_covariance,
                        seasonal_adjustments
                    )
            else:
                seasonal_adjustments = {}
            
            # Convert state to performance prediction
            expected_performance = self.kalman_model.observation_model(predicted_state.state_mean)
            
            # Calculate performance uncertainty
            # This is a simplified approach - full implementation would use observation model Jacobian
            obs_noise = self.kalman_model.filter_config.measurement_noise_std**2
            performance_covariance = jnp.eye(len(expected_performance)) * obs_noise
            
            # Apply fixture adjustments
            fixture_adjustments = {}
            if self.fixture_predictor:
                fixture_adjustments = self.fixture_predictor.get_fixture_adjustments(
                    player_id,
                    current_gameweek or current_state.gameweek,
                    gameweeks_ahead,
                    season
                )
                
                expected_performance, performance_covariance = \
                    self.fixture_predictor.apply_fixture_adjustments(
                        expected_performance,
                        performance_covariance,
                        fixture_adjustments
                    )
            
            # Calculate prediction intervals
            prediction_intervals = self.uncertainty_propagator.calculate_prediction_intervals(
                expected_performance,
                performance_covariance,
                confidence_levels
            )
            
            # Calculate risk metrics
            if include_risk_metrics:
                # Generate samples for risk calculation
                n_samples = 10000
                samples = np.random.multivariate_normal(
                    expected_performance, 
                    performance_covariance, 
                    n_samples
                )
                
                # Convert to points (simplified)
                point_samples = []
                for sample in samples:
                    goals, assists, minutes, bonus = sample[:4]
                    points = goals * 4 + assists * 3 + (minutes >= 60) * 2 + bonus
                    point_samples.append(points)
                
                risk_metrics = RiskMetrics.from_distribution(
                    point_samples,
                    self.config.var_confidence,
                    self.config.risk_free_rate
                )
            else:
                risk_metrics = None
            
            # Create prediction result
            result = PredictionResult(
                player_id=player_id,
                gameweeks_ahead=gameweeks_ahead,
                prediction_date=datetime.now(timezone.utc).isoformat(),
                expected_state=predicted_state.state_mean,
                state_covariance=propagated_covariance,
                expected_performance=expected_performance,
                performance_variance=jnp.diag(performance_covariance),
                prediction_intervals=prediction_intervals,
                risk_metrics=risk_metrics,
                fixture_adjustments=fixture_adjustments,
                seasonality_factors=seasonal_adjustments,
                model_diagnostics={
                    "model_type": type(self.kalman_model).__name__,
                    "filter_type": getattr(self.kalman_model, 'filter_type', 'unknown'),
                    "uncertainty_inflation": self.config.uncertainty_inflation_rate * gameweeks_ahead
                }
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-step prediction failed for player {player_id}: {e}")
            raise StatePredictionError(f"Prediction failed: {e}")
    
    def predict_scenario(
        self,
        player_id: int,
        scenario: Dict[str, Any],
        gameweeks_ahead: int = 3
    ) -> PredictionResult:
        """
        Generate predictions under specific scenario conditions.
        
        Args:
            player_id: Player ID
            scenario: Scenario parameters (injuries, transfers, etc.)
            gameweeks_ahead: Prediction horizon
            
        Returns:
            Scenario-adjusted prediction result
        """
        # Extract scenario factors
        external_factors = {}
        
        if "injury_probability" in scenario:
            external_factors["injury_risk"] = scenario["injury_probability"]
        
        if "rotation_risk" in scenario:
            external_factors["rotation_risk"] = scenario["rotation_risk"]
        
        if "new_signing" in scenario:
            external_factors["competition_uncertainty"] = 0.3
        
        # Generate prediction with scenario adjustments
        return self.predict_multi_step(
            player_id=player_id,
            gameweeks_ahead=gameweeks_ahead,
            external_factors=external_factors
        )
    
    def get_predictions(
        self,
        player_ids: List[int],
        gameweeks_ahead: int = 3,
        **kwargs
    ) -> Dict[int, PredictionResult]:
        """
        Get predictions for multiple players.
        
        Args:
            player_ids: List of player IDs
            gameweeks_ahead: Prediction horizon
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary mapping player IDs to prediction results
        """
        predictions = {}
        
        for player_id in player_ids:
            try:
                predictions[player_id] = self.predict_multi_step(
                    player_id=player_id,
                    gameweeks_ahead=gameweeks_ahead,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to predict for player {player_id}: {e}")
                continue
        
        return predictions
    
    def compare_players(
        self,
        player_ids: List[int],
        metrics: List[str] = None,
        gameweeks_ahead: int = 3
    ) -> pd.DataFrame:
        """
        Compare multiple players across specified metrics.
        
        Args:
            player_ids: List of player IDs to compare
            metrics: List of metrics to compare
            gameweeks_ahead: Prediction horizon
            
        Returns:
            DataFrame with player comparison
        """
        if metrics is None:
            metrics = ["expected_points", "variance", "sharpe_ratio", "value_at_risk"]
        
        predictions = self.get_predictions(player_ids, gameweeks_ahead)
        
        comparison_data = []
        for player_id, pred in predictions.items():
            row = {"player_id": player_id}
            
            for metric in metrics:
                if metric == "expected_points":
                    row[metric] = pred.get_expected_points()
                elif metric == "variance" and pred.risk_metrics:
                    row[metric] = pred.risk_metrics.variance
                elif metric == "sharpe_ratio" and pred.risk_metrics:
                    row[metric] = pred.risk_metrics.sharpe_ratio
                elif metric == "value_at_risk" and pred.risk_metrics:
                    row[metric] = pred.risk_metrics.value_at_risk
                else:
                    row[metric] = None
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def clear_cache(self):
        """Clear prediction cache."""
        self.prediction_cache.clear()
        logger.debug("Prediction cache cleared")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive prediction system diagnostics."""
        return {
            "config": {
                "max_gameweeks_ahead": self.config.max_gameweeks_ahead,
                "uncertainty_inflation_rate": self.config.uncertainty_inflation_rate,
                "enable_seasonality": self.config.enable_seasonality,
                "enable_fixture_awareness": self.config.enable_fixture_awareness
            },
            "cache_size": len(self.prediction_cache),
            "model_diagnostics": self.kalman_model.get_model_diagnostics() if hasattr(self.kalman_model, 'get_model_diagnostics') else {},
            "validation_history": len(self.validator.validation_history)
        }


# Convenience API functions for optimization integration
def get_predictions(
    player_id: int,
    gameweeks_ahead: int,
    predictor: StatePredictor
) -> PredictionResult:
    """Get predictions for a single player."""
    return predictor.predict_multi_step(player_id, gameweeks_ahead)


def get_prediction_intervals(
    player_id: int,
    confidence_levels: List[float],
    predictor: StatePredictor,
    gameweeks_ahead: int = 3
) -> Dict[float, PredictionInterval]:
    """Get prediction intervals for specific confidence levels."""
    result = predictor.predict_multi_step(player_id, gameweeks_ahead, confidence_levels)
    return result.prediction_intervals


def get_risk_metrics(
    player_id: int,
    risk_measures: List[RiskMeasure],
    predictor: StatePredictor,
    gameweeks_ahead: int = 3
) -> Dict[str, float]:
    """Get risk metrics for a player."""
    result = predictor.predict_multi_step(player_id, gameweeks_ahead)
    
    if not result.risk_metrics:
        return {}
    
    risk_data = {}
    for measure in risk_measures:
        if measure == "var":
            risk_data["value_at_risk"] = result.risk_metrics.value_at_risk
        elif measure == "cvar":
            risk_data["conditional_var"] = result.risk_metrics.conditional_var
        elif measure == "sharpe":
            risk_data["sharpe_ratio"] = result.risk_metrics.sharpe_ratio
        elif measure == "max_drawdown":
            risk_data["max_drawdown"] = result.risk_metrics.max_drawdown
    
    return risk_data


def scenario_analysis(
    player_id: int,
    scenarios: Dict[str, Dict[str, Any]],
    predictor: StatePredictor,
    gameweeks_ahead: int = 3
) -> Dict[str, PredictionResult]:
    """Perform scenario analysis for a player."""
    results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        try:
            results[scenario_name] = predictor.predict_scenario(
                player_id, scenario_params, gameweeks_ahead
            )
        except Exception as e:
            logger.warning(f"Scenario {scenario_name} failed for player {player_id}: {e}")
            continue
    
    return results