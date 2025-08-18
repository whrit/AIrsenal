"""
Temporal Weighting System for AIrsenal's Machine Learning Models

This module implements a comprehensive temporal weighting system that assigns different
importance to observations based on recency, with configurable decay functions.
The system is designed to improve model predictions by appropriately emphasizing
recent performances while maintaining enough historical context for stability.

Key Features:
- Multiple decay functions (exponential, linear, power law, hyperbolic, step, Gaussian)
- Position-specific weighting parameters
- Match importance weighting (derby, title race, relegation battles)
- Opponent quality adjustment
- Competition type weighting (league vs cup)
- Injury recovery and transfer adaptation weighting
- Performance-optimized vectorized operations
- Parameter optimization from historical data
- Integration with existing Kalman filter and measurement systems

Classes:
    DecayFunctions: Collection of temporal decay function implementations
    ImportanceWeighting: Match importance and context-based weighting
    PositionSpecificWeighting: Position-aware weight configurations
    WeightOptimizer: Parameter optimization from historical data
    TemporalWeightingSystem: Main weighting engine
    
Usage:
    ```python
    from airsenal.framework.temporal_weighting import TemporalWeightingSystem
    
    # Create system with default configuration
    weighting_system = TemporalWeightingSystem()
    
    # Calculate weights for player observations
    weights = weighting_system.calculate_weights(
        player_id=123,
        observation_dates=["2023-10-01", "2023-10-08", "2023-10-15"],
        current_date="2023-10-22",
        position="MID"
    )
    
    # Apply weights to measurements
    weighted_measurements = weighting_system.apply_weights(
        measurements=measurement_matrix,
        weights=weights
    )
    ```
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from jax import vmap
from jax.scipy import optimize as jax_optimize
from jax.scipy import stats as jax_stats

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from scipy import optimize, stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available, some optimization features will be limited")

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("Scikit-learn not available, using simplified cross-validation")

# Type aliases
Array = Union[np.ndarray, jnp.ndarray]
DecayType = Literal["exponential", "linear", "power_law", "hyperbolic", "step", "gaussian"]
Position = Literal["GK", "DEF", "MID", "FWD"]
WeightVector = Array
TimeVector = Array


@dataclass
class DecayConfig:
    """Configuration for decay function parameters."""
    
    # Decay function type
    decay_type: DecayType = "exponential"
    
    # Exponential decay parameters
    lambda_: float = 0.1  # Decay rate
    
    # Linear decay parameters
    alpha: float = 0.05  # Linear decay rate
    
    # Power law decay parameters
    beta: float = 1.5  # Power law exponent
    
    # Hyperbolic decay parameters
    gamma: float = 2.0  # Hyperbolic decay rate
    
    # Step function parameters
    step_windows: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 7, 1.0),      # Last week: full weight
        (7, 21, 0.8),     # 1-3 weeks: 80% weight
        (21, 42, 0.6),    # 3-6 weeks: 60% weight
        (42, 84, 0.4),    # 6-12 weeks: 40% weight
        (84, float('inf'), 0.2)  # >12 weeks: 20% weight
    ])
    
    # Gaussian decay parameters
    sigma: float = 14.0  # Standard deviation in days
    
    # Minimum weight threshold
    min_weight: float = 0.01
    
    # Maximum time consideration (in days)
    max_time_horizon: float = 365.0


@dataclass
class ImportanceConfig:
    """Configuration for match importance weighting."""
    
    # Derby match boost
    derby_boost: float = 1.3
    
    # Big six fixture boost
    big_six_boost: float = 1.2
    
    # Title race implications
    title_race_boost: float = 1.25
    
    # Relegation battle boost
    relegation_boost: float = 1.15
    
    # European competition qualification boost
    european_boost: float = 1.1
    
    # Cup competition weights
    cup_weights: Dict[str, float] = field(default_factory=lambda: {
        "FA Cup": 0.9,
        "League Cup": 0.8,
        "Champions League": 1.4,
        "Europa League": 1.2,
        "Europa Conference League": 1.1
    })
    
    # Season phase adjustments
    early_season_factor: float = 1.1  # GW 1-8
    mid_season_factor: float = 1.0    # GW 9-30
    late_season_factor: float = 1.2   # GW 31-38
    
    # Fixture congestion penalty
    congestion_penalty: float = 0.95  # Multiple games in short period


@dataclass
class PositionConfig:
    """Position-specific temporal weighting configuration."""
    
    # Memory length by position (in gameweeks)
    memory_lengths: Dict[Position, float] = field(default_factory=lambda: {
        "GK": 12.0,   # Goalkeepers: longer memory (consistency matters)
        "DEF": 10.0,  # Defenders: medium-long memory
        "MID": 8.0,   # Midfielders: balanced
        "FWD": 6.0    # Forwards: shorter memory (recent form crucial)
    })
    
    # Decay rate adjustments by position
    decay_adjustments: Dict[Position, float] = field(default_factory=lambda: {
        "GK": 0.8,    # Slower decay (more stable)
        "DEF": 0.9,   # Moderately slow decay
        "MID": 1.0,   # Baseline decay
        "FWD": 1.2    # Faster decay (form-sensitive)
    })
    
    # Form sensitivity by position
    form_sensitivity: Dict[Position, float] = field(default_factory=lambda: {
        "GK": 0.7,    # Less sensitive to short-term form
        "DEF": 0.8,   # Moderately sensitive
        "MID": 1.0,   # Baseline sensitivity
        "FWD": 1.3    # Highly sensitive to recent form
    })


@dataclass
class WeightingResult:
    """Result of temporal weight calculation."""
    
    weights: Array
    normalized_weights: Array
    total_weight: float
    effective_sample_size: float
    oldest_significant_observation: Optional[datetime]
    decay_function_used: DecayType
    position_adjustment_applied: bool
    importance_adjustment_applied: bool
    confidence_score: float


class DecayFunctions:
    """
    Collection of temporal decay function implementations.
    
    All functions are JAX-compiled for performance and accept vectorized inputs.
    Time differences are expected in days.
    """
    
    @staticmethod
    @jax.jit
    def exponential_decay(time_diff: Array, lambda_: float, min_weight: float = 0.01) -> Array:
        """
        Exponential decay: w(t) = exp(-λt)
        
        Args:
            time_diff: Time differences in days
            lambda_: Decay rate parameter
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        weights = jnp.exp(-lambda_ * time_diff)
        return jnp.maximum(weights, min_weight)
    
    @staticmethod
    @jax.jit
    def linear_decay(time_diff: Array, alpha: float, min_weight: float = 0.01) -> Array:
        """
        Linear decay: w(t) = max(0, 1 - αt)
        
        Args:
            time_diff: Time differences in days
            alpha: Linear decay rate
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        weights = jnp.maximum(0, 1 - alpha * time_diff)
        return jnp.maximum(weights, min_weight)
    
    @staticmethod
    @jax.jit
    def power_law_decay(time_diff: Array, beta: float, min_weight: float = 0.01) -> Array:
        """
        Power law decay: w(t) = (1 + t)^(-β)
        
        Args:
            time_diff: Time differences in days
            beta: Power law exponent
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        weights = jnp.power(1 + time_diff, -beta)
        return jnp.maximum(weights, min_weight)
    
    @staticmethod
    @jax.jit
    def hyperbolic_decay(time_diff: Array, gamma: float, min_weight: float = 0.01) -> Array:
        """
        Hyperbolic decay: w(t) = 1 / (1 + γt)
        
        Args:
            time_diff: Time differences in days
            gamma: Hyperbolic decay rate
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        weights = 1.0 / (1.0 + gamma * time_diff)
        return jnp.maximum(weights, min_weight)
    
    @staticmethod
    def step_function_decay(
        time_diff: Array, 
        step_windows: List[Tuple[float, float, float]], 
        min_weight: float = 0.01
    ) -> Array:
        """
        Step function: Fixed windows with discrete weights
        
        Args:
            time_diff: Time differences in days
            step_windows: List of (start, end, weight) tuples
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        time_diff = np.asarray(time_diff)
        weights = np.full_like(time_diff, min_weight, dtype=float)
        
        for start, end, weight in step_windows:
            mask = (time_diff >= start) & (time_diff < end)
            weights[mask] = weight
        
        return jnp.array(weights)
    
    @staticmethod
    @jax.jit
    def gaussian_decay(time_diff: Array, sigma: float, min_weight: float = 0.01) -> Array:
        """
        Gaussian decay: w(t) = exp(-t²/2σ²)
        
        Args:
            time_diff: Time differences in days
            sigma: Standard deviation parameter
            min_weight: Minimum weight threshold
            
        Returns:
            Decay weights
        """
        weights = jnp.exp(-0.5 * jnp.square(time_diff / sigma))
        return jnp.maximum(weights, min_weight)


class ImportanceWeighting:
    """
    Match importance and context-based weighting adjustments.
    
    Accounts for factors like match importance, opponent quality,
    competition type, and situational context.
    """
    
    def __init__(self, config: Optional[ImportanceConfig] = None):
        """Initialize importance weighting system."""
        self.config = config or ImportanceConfig()
        
        # Cache for team strength ratings
        self._team_strength_cache: Dict[Tuple[str, str], float] = {}
        
        # Cache for derby classifications
        self._derby_cache: Dict[Tuple[str, str], bool] = {}
        
        logger.info("Initialized ImportanceWeighting system")
    
    def calculate_match_importance(
        self,
        home_team: str,
        away_team: str,
        gameweek: int,
        competition: str = "Premier League",
        season: str = "2023",
        home_position: Optional[int] = None,
        away_position: Optional[int] = None
    ) -> float:
        """
        Calculate match importance multiplier.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            gameweek: Gameweek number
            competition: Competition name
            season: Season identifier
            home_position: Home team league position
            away_position: Away team league position
            
        Returns:
            Importance multiplier (typically 0.8-1.5)
        """
        try:
            importance = 1.0
            
            # Derby match boost
            if self._is_derby_match(home_team, away_team):
                importance *= self.config.derby_boost
                logger.debug(f"Derby match boost applied: {home_team} vs {away_team}")
            
            # Big six fixture boost
            if self._is_big_six_fixture(home_team, away_team):
                importance *= self.config.big_six_boost
            
            # Season phase adjustment
            importance *= self._get_season_phase_adjustment(gameweek)
            
            # Title race / relegation implications
            if home_position is not None and away_position is not None:
                importance *= self._get_league_position_importance(
                    home_position, away_position, gameweek
                )
            
            # Competition type adjustment
            if competition != "Premier League":
                cup_weight = self.config.cup_weights.get(competition, 1.0)
                importance *= cup_weight
            
            # Fixture congestion adjustment
            if self._is_fixture_congested(gameweek):
                importance *= self.config.congestion_penalty
            
            return max(0.5, min(2.0, importance))  # Clamp to reasonable range
            
        except Exception as e:
            logger.warning(f"Match importance calculation failed: {e}")
            return 1.0
    
    def calculate_opponent_quality_adjustment(
        self,
        opponent_team: str,
        season: str,
        player_team: str,
        is_home: bool = True
    ) -> float:
        """
        Calculate opponent quality adjustment factor.
        
        Args:
            opponent_team: Opponent team name
            season: Season identifier
            player_team: Player's team name
            is_home: Whether player's team is at home
            
        Returns:
            Quality adjustment factor (0.7-1.3)
        """
        try:
            # Get team strength ratings
            player_strength = self._get_team_strength(player_team, season)
            opponent_strength = self._get_team_strength(opponent_team, season)
            
            # Calculate relative strength
            if opponent_strength > 0:
                relative_strength = player_strength / opponent_strength
            else:
                relative_strength = 1.0
            
            # Apply home advantage
            if is_home:
                relative_strength *= 1.1
            else:
                relative_strength *= 0.95
            
            # Convert to adjustment factor
            # Stronger opponents make performances more valuable
            if relative_strength < 0.8:  # Much weaker team
                adjustment = 1.2
            elif relative_strength < 0.9:  # Weaker team
                adjustment = 1.15
            elif relative_strength > 1.2:  # Much stronger team
                adjustment = 0.8
            elif relative_strength > 1.1:  # Stronger team
                adjustment = 0.9
            else:  # Similar strength
                adjustment = 1.0
            
            return max(0.7, min(1.3, adjustment))
            
        except Exception as e:
            logger.warning(f"Opponent quality adjustment failed: {e}")
            return 1.0
    
    def _is_derby_match(self, home_team: str, away_team: str) -> bool:
        """Check if match is a derby."""
        cache_key = tuple(sorted([home_team, away_team]))
        
        if cache_key in self._derby_cache:
            return self._derby_cache[cache_key]
        
        # Define derby pairs
        derby_pairs = {
            ("Arsenal", "Tottenham"),
            ("Chelsea", "Fulham"),
            ("Crystal Palace", "Brighton"),
            ("Everton", "Liverpool"),
            ("Manchester City", "Manchester United"),
            ("Newcastle", "Sunderland"),  # If both in PL
            ("West Ham", "Millwall"),     # If both in PL
            # Add more derby classifications as needed
        }
        
        is_derby = cache_key in derby_pairs
        self._derby_cache[cache_key] = is_derby
        
        return is_derby
    
    def _is_big_six_fixture(self, home_team: str, away_team: str) -> bool:
        """Check if match involves big six teams."""
        big_six = {"Arsenal", "Chelsea", "Liverpool", "Manchester City", 
                   "Manchester United", "Tottenham"}
        
        return home_team in big_six and away_team in big_six
    
    def _get_season_phase_adjustment(self, gameweek: int) -> float:
        """Get adjustment factor based on season phase."""
        if gameweek <= 8:
            return self.config.early_season_factor
        elif gameweek >= 31:
            return self.config.late_season_factor
        else:
            return self.config.mid_season_factor
    
    def _get_league_position_importance(
        self, 
        home_position: int, 
        away_position: int, 
        gameweek: int
    ) -> float:
        """Calculate importance based on league positions."""
        # Title race (top 4)
        if max(home_position, away_position) <= 4 and gameweek >= 20:
            return self.config.title_race_boost
        
        # European competition (top 7)
        if max(home_position, away_position) <= 7 and gameweek >= 25:
            return self.config.european_boost
        
        # Relegation battle (bottom 6)
        if min(home_position, away_position) >= 15 and gameweek >= 25:
            return self.config.relegation_boost
        
        return 1.0
    
    def _is_fixture_congested(self, gameweek: int) -> bool:
        """Check if fixture is in congested period."""
        # Simplified check - could be enhanced with actual fixture density
        congested_periods = [12, 16, 26, 34]  # Typical busy periods
        return any(abs(gameweek - period) <= 1 for period in congested_periods)
    
    def _get_team_strength(self, team: str, season: str) -> float:
        """Get team strength rating (cached)."""
        cache_key = (team, season)
        
        if cache_key in self._team_strength_cache:
            return self._team_strength_cache[cache_key]
        
        # Default team strength ratings (could be loaded from database)
        default_strengths = {
            "Manchester City": 0.95,
            "Arsenal": 0.90,
            "Liverpool": 0.88,
            "Chelsea": 0.85,
            "Manchester United": 0.82,
            "Tottenham": 0.80,
            "Newcastle": 0.75,
            "Brighton": 0.70,
            "Aston Villa": 0.68,
            "West Ham": 0.65,
            # Add more teams as needed
        }
        
        strength = default_strengths.get(team, 0.60)  # Default to average
        self._team_strength_cache[cache_key] = strength
        
        return strength


class PositionSpecificWeighting:
    """
    Position-aware temporal weighting with different parameters per position.
    
    Implements position-specific memory lengths, decay rates, and form sensitivity
    based on the different characteristics of each position.
    """
    
    def __init__(self, config: Optional[PositionConfig] = None):
        """Initialize position-specific weighting."""
        self.config = config or PositionConfig()
        logger.info("Initialized PositionSpecificWeighting system")
    
    def get_position_adjusted_decay_config(
        self, 
        position: Position, 
        base_config: DecayConfig
    ) -> DecayConfig:
        """
        Get position-adjusted decay configuration.
        
        Args:
            position: Player position
            base_config: Base decay configuration
            
        Returns:
            Position-adjusted decay configuration
        """
        try:
            # Get position-specific adjustments
            decay_adjustment = self.config.decay_adjustments.get(position, 1.0)
            memory_length = self.config.memory_lengths.get(position, 8.0)
            
            # Create adjusted configuration
            adjusted_config = DecayConfig(
                decay_type=base_config.decay_type,
                lambda_=base_config.lambda_ * decay_adjustment,
                alpha=base_config.alpha * decay_adjustment,
                beta=base_config.beta * decay_adjustment,
                gamma=base_config.gamma * decay_adjustment,
                sigma=base_config.sigma * decay_adjustment,
                min_weight=base_config.min_weight,
                max_time_horizon=memory_length * 7  # Convert weeks to days
            )
            
            # Adjust step windows for position
            if base_config.decay_type == "step":
                adjusted_config.step_windows = self._adjust_step_windows_for_position(
                    base_config.step_windows, position
                )
            
            return adjusted_config
            
        except Exception as e:
            logger.warning(f"Position adjustment failed for {position}: {e}")
            return base_config
    
    def calculate_form_adjustment(
        self, 
        position: Position, 
        recent_performances: Array, 
        time_diffs: Array
    ) -> float:
        """
        Calculate form-based adjustment factor.
        
        Args:
            position: Player position
            recent_performances: Recent performance scores
            time_diffs: Time differences for recent performances
            
        Returns:
            Form adjustment factor
        """
        try:
            if len(recent_performances) < 2:
                return 1.0
            
            # Calculate recent form trend
            recent_mask = time_diffs <= 21  # Last 3 weeks
            if not jnp.any(recent_mask):
                return 1.0
            
            recent_scores = recent_performances[recent_mask]
            recent_times = time_diffs[recent_mask]
            
            # Calculate form trend (simple linear regression)
            if len(recent_scores) >= 2:
                trend = jnp.polyfit(recent_times, recent_scores, 1)[0]
            else:
                trend = 0.0
            
            # Apply position-specific form sensitivity
            form_sensitivity = self.config.form_sensitivity.get(position, 1.0)
            adjustment = 1.0 + (trend * form_sensitivity * 0.1)  # Scale factor
            
            return max(0.8, min(1.2, adjustment))
            
        except Exception as e:
            logger.warning(f"Form adjustment calculation failed: {e}")
            return 1.0
    
    def _adjust_step_windows_for_position(
        self, 
        base_windows: List[Tuple[float, float, float]], 
        position: Position
    ) -> List[Tuple[float, float, float]]:
        """Adjust step function windows for position."""
        memory_multiplier = self.config.memory_lengths.get(position, 8.0) / 8.0
        
        adjusted_windows = []
        for start, end, weight in base_windows:
            adjusted_start = start * memory_multiplier
            adjusted_end = end * memory_multiplier if end != float('inf') else end
            adjusted_windows.append((adjusted_start, adjusted_end, weight))
        
        return adjusted_windows


class WeightOptimizer:
    """
    Learn optimal decay parameters from historical data using cross-validation
    and Bayesian optimization techniques.
    """
    
    def __init__(
        self,
        optimization_method: Literal["grid_search", "bayesian", "gradient"] = "bayesian",
        cv_folds: int = 5,
        n_calls: int = 50
    ):
        """
        Initialize weight optimizer.
        
        Args:
            optimization_method: Optimization method to use
            cv_folds: Number of cross-validation folds
            n_calls: Number of optimization calls (for Bayesian optimization)
        """
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        
        # Parameter bounds for different decay functions
        self.parameter_bounds = {
            "exponential": {"lambda_": (0.01, 1.0)},
            "linear": {"alpha": (0.001, 0.1)},
            "power_law": {"beta": (0.5, 3.0)},
            "hyperbolic": {"gamma": (0.1, 5.0)},
            "gaussian": {"sigma": (7.0, 42.0)}
        }
        
        # Cache for optimization results
        self._optimization_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized WeightOptimizer with {optimization_method} method")
    
    def optimize_decay_parameters(
        self,
        historical_data: pd.DataFrame,
        target_column: str = "points",
        position: Optional[Position] = None,
        decay_type: DecayType = "exponential"
    ) -> Tuple[DecayConfig, float]:
        """
        Optimize decay parameters for best prediction accuracy.
        
        Args:
            historical_data: Historical player performance data
            target_column: Column to optimize predictions for
            position: Player position (None for all positions)
            decay_type: Type of decay function to optimize
            
        Returns:
            Tuple of (optimized_config, cross_validation_score)
        """
        try:
            # Prepare data
            X, y, time_diffs = self._prepare_optimization_data(
                historical_data, target_column, position
            )
            
            if len(X) < 10:
                logger.warning("Insufficient data for optimization")
                return DecayConfig(decay_type=decay_type), 0.0
            
            # Define objective function
            def objective(params):
                config = self._params_to_config(params, decay_type)
                score = self._evaluate_config(config, X, y, time_diffs)
                return -score  # Minimize negative score
            
            # Get parameter bounds
            bounds = self._get_parameter_bounds(decay_type)
            
            # Optimize parameters
            if self.optimization_method == "grid_search":
                best_params, best_score = self._grid_search_optimize(
                    objective, bounds, decay_type
                )
            elif self.optimization_method == "bayesian":
                best_params, best_score = self._bayesian_optimize(
                    objective, bounds
                )
            else:  # gradient
                best_params, best_score = self._gradient_optimize(
                    objective, bounds
                )
            
            # Create optimized configuration
            optimized_config = self._params_to_config(best_params, decay_type)
            
            logger.info(f"Optimized {decay_type} decay for {position}: score={-best_score:.4f}")
            
            return optimized_config, -best_score
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return DecayConfig(decay_type=decay_type), 0.0
    
    def optimize_all_positions(
        self,
        historical_data: pd.DataFrame,
        target_column: str = "points",
        decay_type: DecayType = "exponential"
    ) -> Dict[Position, Tuple[DecayConfig, float]]:
        """
        Optimize decay parameters for all positions.
        
        Args:
            historical_data: Historical player performance data
            target_column: Column to optimize predictions for
            decay_type: Type of decay function to optimize
            
        Returns:
            Dictionary mapping positions to (config, score) tuples
        """
        results = {}
        
        for position in ["GK", "DEF", "MID", "FWD"]:
            try:
                position_data = historical_data[
                    historical_data["position"] == position
                ]
                
                if len(position_data) >= 10:
                    config, score = self.optimize_decay_parameters(
                        position_data, target_column, position, decay_type
                    )
                    results[position] = (config, score)
                else:
                    logger.warning(f"Insufficient data for {position}")
                    results[position] = (DecayConfig(decay_type=decay_type), 0.0)
                    
            except Exception as e:
                logger.error(f"Optimization failed for {position}: {e}")
                results[position] = (DecayConfig(decay_type=decay_type), 0.0)
        
        return results
    
    def _prepare_optimization_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        position: Optional[Position]
    ) -> Tuple[Array, Array, Array]:
        """Prepare data for optimization."""
        # Filter by position if specified
        if position:
            data = data[data["position"] == position]
        
        # Ensure required columns exist
        required_cols = [target_column, "date", "player_id"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by player and date
        data = data.sort_values(["player_id", "date"])
        
        # Calculate time differences and features
        features = []
        targets = []
        time_diffs_list = []
        
        for player_id in data["player_id"].unique():
            player_data = data[data["player_id"] == player_id]
            
            if len(player_data) < 3:  # Need minimum history
                continue
            
            for i in range(2, len(player_data)):
                # Current observation
                current = player_data.iloc[i]
                
                # Historical observations
                history = player_data.iloc[:i]
                
                # Calculate time differences (in days)
                current_date = pd.to_datetime(current["date"])
                hist_dates = pd.to_datetime(history["date"])
                time_diffs = (current_date - hist_dates).dt.days.values
                
                # Features: historical performance values
                hist_values = history[target_column].values
                
                # Pad or truncate to fixed size
                max_history = 10
                if len(hist_values) > max_history:
                    hist_values = hist_values[-max_history:]
                    time_diffs = time_diffs[-max_history:]
                else:
                    pad_size = max_history - len(hist_values)
                    hist_values = np.pad(hist_values, (pad_size, 0), constant_values=0)
                    time_diffs = np.pad(time_diffs, (pad_size, 0), constant_values=365)
                
                features.append(hist_values)
                targets.append(current[target_column])
                time_diffs_list.append(time_diffs)
        
        return (
            jnp.array(features),
            jnp.array(targets),
            jnp.array(time_diffs_list)
        )
    
    def _evaluate_config(
        self,
        config: DecayConfig,
        X: Array,
        y: Array,
        time_diffs: Array
    ) -> float:
        """Evaluate configuration using cross-validation."""
        try:
            # Calculate weights for all samples
            decay_func = getattr(DecayFunctions, f"{config.decay_type}_decay")
            
            if config.decay_type == "exponential":
                weights = vmap(lambda td: decay_func(td, config.lambda_, config.min_weight))(time_diffs)
            elif config.decay_type == "linear":
                weights = vmap(lambda td: decay_func(td, config.alpha, config.min_weight))(time_diffs)
            elif config.decay_type == "power_law":
                weights = vmap(lambda td: decay_func(td, config.beta, config.min_weight))(time_diffs)
            elif config.decay_type == "hyperbolic":
                weights = vmap(lambda td: decay_func(td, config.gamma, config.min_weight))(time_diffs)
            elif config.decay_type == "gaussian":
                weights = vmap(lambda td: decay_func(td, config.sigma, config.min_weight))(time_diffs)
            else:
                weights = jnp.ones_like(time_diffs)
            
            # Calculate weighted predictions
            weighted_X = X * weights[:, :, np.newaxis]
            weighted_means = jnp.sum(weighted_X, axis=1) / jnp.sum(weights, axis=1)[:, np.newaxis]
            predictions = jnp.mean(weighted_means, axis=1)
            
            # Calculate cross-validation score using correlation
            if HAS_SKLEARN:
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = []
                
                X_np = np.array(X)
                y_np = np.array(y)
                predictions_np = np.array(predictions)
                
                for train_idx, test_idx in tscv.split(X_np):
                    test_pred = predictions_np[test_idx]
                    test_true = y_np[test_idx]
                    
                    if len(test_pred) > 1 and np.var(test_pred) > 0 and np.var(test_true) > 0:
                        corr = np.corrcoef(test_pred, test_true)[0, 1]
                        if not np.isnan(corr):
                            scores.append(corr)
                
                return np.mean(scores) if scores else 0.0
            else:
                # Simplified validation without sklearn
                predictions_np = np.array(predictions)
                y_np = np.array(y)
                
                if len(predictions_np) > 1 and np.var(predictions_np) > 0 and np.var(y_np) > 0:
                    corr = np.corrcoef(predictions_np, y_np)[0, 1]
                    return corr if not np.isnan(corr) else 0.0
                else:
                    return 0.0
            
        except Exception as e:
            logger.warning(f"Config evaluation failed: {e}")
            return 0.0
    
    def _params_to_config(self, params: List[float], decay_type: DecayType) -> DecayConfig:
        """Convert optimization parameters to DecayConfig."""
        config = DecayConfig(decay_type=decay_type)
        
        if decay_type == "exponential":
            config.lambda_ = params[0]
        elif decay_type == "linear":
            config.alpha = params[0]
        elif decay_type == "power_law":
            config.beta = params[0]
        elif decay_type == "hyperbolic":
            config.gamma = params[0]
        elif decay_type == "gaussian":
            config.sigma = params[0]
        
        return config
    
    def _get_parameter_bounds(self, decay_type: DecayType) -> List[Tuple[float, float]]:
        """Get parameter bounds for decay type."""
        bounds_dict = self.parameter_bounds.get(decay_type, {})
        return list(bounds_dict.values())
    
    def _grid_search_optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        decay_type: DecayType
    ) -> Tuple[List[float], float]:
        """Grid search optimization."""
        if len(bounds) != 1:
            raise ValueError("Grid search currently supports single parameter optimization")
        
        param_min, param_max = bounds[0]
        param_values = np.linspace(param_min, param_max, 20)
        
        best_score = float('inf')
        best_params = [param_min]
        
        for param in param_values:
            score = objective([param])
            if score < best_score:
                best_score = score
                best_params = [param]
        
        return best_params, best_score
    
    def _bayesian_optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]]
    ) -> Tuple[List[float], float]:
        """Bayesian optimization using scipy."""
        if not HAS_SCIPY:
            logger.warning("SciPy not available, falling back to grid search")
            return self._grid_search_optimize(objective, bounds, "exponential")
        
        try:
            result = optimize.differential_evolution(
                objective,
                bounds,
                maxiter=self.n_calls,
                seed=42
            )
            
            return result.x.tolist(), result.fun
            
        except Exception as e:
            logger.warning(f"Bayesian optimization failed: {e}, falling back to grid search")
            return self._grid_search_optimize(objective, bounds, "exponential")
    
    def _gradient_optimize(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]]
    ) -> Tuple[List[float], float]:
        """Gradient-based optimization."""
        if not HAS_SCIPY:
            logger.warning("SciPy not available, falling back to grid search")
            return self._grid_search_optimize(objective, bounds, "exponential")
        
        try:
            # Initial guess (middle of bounds)
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
            
            result = optimize.minimize(
                objective,
                x0,
                bounds=bounds,
                method="L-BFGS-B"
            )
            
            return result.x.tolist(), result.fun
            
        except Exception as e:
            logger.warning(f"Gradient optimization failed: {e}, falling back to grid search")
            return self._grid_search_optimize(objective, bounds, "exponential")


class TemporalWeightingSystem:
    """
    Main temporal weighting engine that combines all components.
    
    Provides a unified interface for calculating temporal weights with
    position-specific parameters, importance adjustments, and optimized
    decay functions.
    """
    
    def __init__(
        self,
        decay_config: Optional[DecayConfig] = None,
        importance_config: Optional[ImportanceConfig] = None,
        position_config: Optional[PositionConfig] = None,
        enable_caching: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize temporal weighting system.
        
        Args:
            decay_config: Decay function configuration
            importance_config: Match importance configuration
            position_config: Position-specific configuration
            enable_caching: Enable weight caching for performance
            cache_size: Maximum cache size
        """
        self.decay_config = decay_config or DecayConfig()
        self.importance_config = importance_config or ImportanceConfig()
        self.position_config = position_config or PositionConfig()
        
        # Initialize subsystems
        self.importance_weighting = ImportanceWeighting(self.importance_config)
        self.position_weighting = PositionSpecificWeighting(self.position_config)
        self.weight_optimizer = WeightOptimizer()
        
        # Caching setup
        self.enable_caching = enable_caching
        if enable_caching:
            self._weight_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Performance tracking
        self._calculation_times: List[float] = []
        
        logger.info("Initialized TemporalWeightingSystem")
    
    def calculate_weights(
        self,
        observation_dates: List[Union[str, datetime]],
        current_date: Union[str, datetime],
        position: Position = "MID",
        player_id: Optional[int] = None,
        match_contexts: Optional[List[Dict[str, Any]]] = None,
        custom_decay_config: Optional[DecayConfig] = None
    ) -> WeightingResult:
        """
        Calculate temporal weights for observations.
        
        Args:
            observation_dates: Dates of historical observations
            current_date: Current date for weight calculation
            position: Player position
            player_id: Player ID for personalized weighting
            match_contexts: List of match context dictionaries
            custom_decay_config: Custom decay configuration
            
        Returns:
            WeightingResult with calculated weights and metadata
        """
        try:
            start_time = time.perf_counter()
            
            # Parse dates
            obs_dates = [self._parse_date(d) for d in observation_dates]
            curr_date = self._parse_date(current_date)
            
            # Calculate time differences in days
            time_diffs = jnp.array([
                (curr_date - obs_date).days for obs_date in obs_dates
            ])
            
            # Filter out future dates and very old observations
            valid_mask = (time_diffs >= 0) & (time_diffs <= self.decay_config.max_time_horizon)
            valid_time_diffs = time_diffs[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_time_diffs) == 0:
                logger.warning("No valid observations for weight calculation")
                return self._create_empty_result()
            
            # Get position-adjusted decay configuration
            decay_config = custom_decay_config or self.decay_config
            position_adjusted_config = self.position_weighting.get_position_adjusted_decay_config(
                position, decay_config
            )
            
            # Calculate base temporal weights
            base_weights = self._calculate_base_weights(
                valid_time_diffs, position_adjusted_config
            )
            
            # Apply importance adjustments
            importance_weights = self._calculate_importance_weights(
                valid_indices, match_contexts, position
            )
            
            # Combine weights
            combined_weights = base_weights * importance_weights
            
            # Normalize weights
            total_weight = jnp.sum(combined_weights)
            if total_weight > 0:
                normalized_weights = combined_weights / total_weight
            else:
                normalized_weights = jnp.ones_like(combined_weights) / len(combined_weights)
            
            # Calculate effective sample size
            effective_sample_size = self._calculate_effective_sample_size(normalized_weights)
            
            # Find oldest significant observation
            significant_mask = normalized_weights >= (0.01 / len(normalized_weights))
            if jnp.any(significant_mask):
                oldest_significant_idx = valid_indices[jnp.where(significant_mask)[0][-1]]
                oldest_significant_date = obs_dates[oldest_significant_idx]
            else:
                oldest_significant_date = None
            
            # Create full-size weight arrays
            full_weights = jnp.zeros(len(observation_dates))
            full_normalized_weights = jnp.zeros(len(observation_dates))
            
            full_weights = full_weights.at[valid_indices].set(combined_weights)
            full_normalized_weights = full_normalized_weights.at[valid_indices].set(normalized_weights)
            
            # Track performance
            calculation_time = time.perf_counter() - start_time
            self._calculation_times.append(calculation_time)
            
            # Create result
            result = WeightingResult(
                weights=full_weights,
                normalized_weights=full_normalized_weights,
                total_weight=float(total_weight),
                effective_sample_size=float(effective_sample_size),
                oldest_significant_observation=oldest_significant_date,
                decay_function_used=position_adjusted_config.decay_type,
                position_adjustment_applied=True,
                importance_adjustment_applied=match_contexts is not None,
                confidence_score=self._calculate_confidence_score(
                    effective_sample_size, len(valid_time_diffs)
                )
            )
            
            logger.debug(f"Calculated weights for {len(observation_dates)} observations in {calculation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
            return self._create_empty_result()
    
    def apply_weights(
        self,
        measurements: Array,
        weights: Array,
        method: Literal["multiply", "weighted_average", "weighted_sum"] = "multiply"
    ) -> Array:
        """
        Apply temporal weights to measurements.
        
        Args:
            measurements: Measurement matrix (observations x features)
            weights: Weight vector (one per observation)
            method: Method for applying weights
            
        Returns:
            Weighted measurements
        """
        try:
            measurements = jnp.asarray(measurements)
            weights = jnp.asarray(weights)
            
            # Ensure compatible dimensions
            if measurements.ndim == 1:
                measurements = measurements.reshape(-1, 1)
            
            if len(weights) != measurements.shape[0]:
                raise ValueError(f"Weight dimension mismatch: {len(weights)} != {measurements.shape[0]}")
            
            if method == "multiply":
                # Element-wise multiplication
                return measurements * weights[:, np.newaxis]
            
            elif method == "weighted_average":
                # Weighted average across observations
                weighted_sum = jnp.sum(measurements * weights[:, np.newaxis], axis=0)
                weight_sum = jnp.sum(weights)
                return weighted_sum / jnp.maximum(weight_sum, 1e-8)
            
            elif method == "weighted_sum":
                # Weighted sum across observations
                return jnp.sum(measurements * weights[:, np.newaxis], axis=0)
            
            else:
                raise ValueError(f"Unknown weighting method: {method}")
                
        except Exception as e:
            logger.error(f"Weight application failed: {e}")
            return measurements
    
    def batch_calculate_weights(
        self,
        batch_data: List[Dict[str, Any]],
        parallel: bool = True
    ) -> List[WeightingResult]:
        """
        Calculate weights for batch of players/observations.
        
        Args:
            batch_data: List of weight calculation parameters
            parallel: Use parallel processing (if available)
            
        Returns:
            List of WeightingResults
        """
        try:
            if parallel and len(batch_data) > 1:
                # For now, process sequentially
                # Could be enhanced with actual parallel processing
                pass
            
            results = []
            for data in batch_data:
                result = self.calculate_weights(**data)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch weight calculation failed: {e}")
            return [self._create_empty_result() for _ in batch_data]
    
    def optimize_parameters(
        self,
        historical_data: pd.DataFrame,
        target_column: str = "points",
        positions: Optional[List[Position]] = None
    ) -> Dict[Position, DecayConfig]:
        """
        Optimize temporal weighting parameters for all positions.
        
        Args:
            historical_data: Historical performance data
            target_column: Target column for optimization
            positions: Positions to optimize (default: all)
            
        Returns:
            Dictionary of optimized configs by position
        """
        try:
            positions = positions or ["GK", "DEF", "MID", "FWD"]
            optimized_configs = {}
            
            for position in positions:
                logger.info(f"Optimizing parameters for {position}")
                
                config, score = self.weight_optimizer.optimize_decay_parameters(
                    historical_data, target_column, position, self.decay_config.decay_type
                )
                
                optimized_configs[position] = config
                logger.info(f"Optimization complete for {position}: score={score:.4f}")
            
            return optimized_configs
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        metrics = {
            "total_calculations": len(self._calculation_times),
            "average_calculation_time_ms": np.mean(self._calculation_times) * 1000 if self._calculation_times else 0,
            "max_calculation_time_ms": np.max(self._calculation_times) * 1000 if self._calculation_times else 0,
        }
        
        if self.enable_caching:
            total_requests = self._cache_hits + self._cache_misses
            metrics.update({
                "cache_hit_rate": self._cache_hits / max(total_requests, 1),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_size": len(self._weight_cache)
            })
        
        return metrics
    
    def _calculate_base_weights(
        self, 
        time_diffs: Array, 
        config: DecayConfig
    ) -> Array:
        """Calculate base temporal weights using decay function."""
        decay_type = config.decay_type
        
        if decay_type == "exponential":
            return DecayFunctions.exponential_decay(
                time_diffs, config.lambda_, config.min_weight
            )
        elif decay_type == "linear":
            return DecayFunctions.linear_decay(
                time_diffs, config.alpha, config.min_weight
            )
        elif decay_type == "power_law":
            return DecayFunctions.power_law_decay(
                time_diffs, config.beta, config.min_weight
            )
        elif decay_type == "hyperbolic":
            return DecayFunctions.hyperbolic_decay(
                time_diffs, config.gamma, config.min_weight
            )
        elif decay_type == "step":
            return DecayFunctions.step_function_decay(
                time_diffs, config.step_windows, config.min_weight
            )
        elif decay_type == "gaussian":
            return DecayFunctions.gaussian_decay(
                time_diffs, config.sigma, config.min_weight
            )
        else:
            logger.warning(f"Unknown decay type: {decay_type}, using exponential")
            return DecayFunctions.exponential_decay(
                time_diffs, config.lambda_, config.min_weight
            )
    
    def _calculate_importance_weights(
        self,
        observation_indices: Array,
        match_contexts: Optional[List[Dict[str, Any]]],
        position: Position
    ) -> Array:
        """Calculate importance-based weight adjustments."""
        if match_contexts is None:
            return jnp.ones(len(observation_indices))
        
        try:
            importance_weights = []
            
            for idx in observation_indices:
                if idx < len(match_contexts) and match_contexts[idx] is not None:
                    context = match_contexts[idx]
                    
                    # Calculate match importance
                    importance = self.importance_weighting.calculate_match_importance(
                        home_team=context.get("home_team", ""),
                        away_team=context.get("away_team", ""),
                        gameweek=context.get("gameweek", 20),
                        competition=context.get("competition", "Premier League"),
                        season=context.get("season", "2023"),
                        home_position=context.get("home_position"),
                        away_position=context.get("away_position")
                    )
                    
                    # Calculate opponent quality adjustment
                    opponent_adj = self.importance_weighting.calculate_opponent_quality_adjustment(
                        opponent_team=context.get("opponent_team", ""),
                        season=context.get("season", "2023"),
                        player_team=context.get("player_team", ""),
                        is_home=context.get("is_home", True)
                    )
                    
                    combined_importance = importance * opponent_adj
                    importance_weights.append(combined_importance)
                else:
                    importance_weights.append(1.0)
            
            return jnp.array(importance_weights)
            
        except Exception as e:
            logger.warning(f"Importance weight calculation failed: {e}")
            return jnp.ones(len(observation_indices))
    
    def _calculate_effective_sample_size(self, normalized_weights: Array) -> float:
        """Calculate effective sample size from normalized weights."""
        # Effective sample size = 1 / sum(weights^2)
        return 1.0 / jnp.sum(jnp.square(normalized_weights))
    
    def _calculate_confidence_score(
        self, 
        effective_sample_size: float, 
        total_observations: int
    ) -> float:
        """Calculate confidence score for weight calculation."""
        # Higher effective sample size and more observations = higher confidence
        ess_score = min(effective_sample_size / 5.0, 1.0)  # Normalize to max 5
        obs_score = min(total_observations / 10.0, 1.0)   # Normalize to max 10
        
        return (ess_score + obs_score) / 2.0
    
    def _parse_date(self, date_input: Union[str, datetime]) -> datetime:
        """Parse date input to datetime object."""
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, str):
            try:
                return pd.to_datetime(date_input)
            except Exception:
                return datetime.strptime(date_input, "%Y-%m-%d")
        else:
            raise ValueError(f"Invalid date input: {date_input}")
    
    def _create_empty_result(self) -> WeightingResult:
        """Create empty weighting result for error cases."""
        return WeightingResult(
            weights=jnp.array([]),
            normalized_weights=jnp.array([]),
            total_weight=0.0,
            effective_sample_size=0.0,
            oldest_significant_observation=None,
            decay_function_used=self.decay_config.decay_type,
            position_adjustment_applied=False,
            importance_adjustment_applied=False,
            confidence_score=0.0
        )


# Utility functions for easy integration

def create_default_weighting_system(
    decay_type: DecayType = "exponential",
    enable_optimization: bool = True
) -> TemporalWeightingSystem:
    """
    Create a default temporal weighting system with sensible defaults.
    
    Args:
        decay_type: Default decay function type
        enable_optimization: Enable parameter optimization
        
    Returns:
        Configured TemporalWeightingSystem
    """
    decay_config = DecayConfig(decay_type=decay_type)
    importance_config = ImportanceConfig()
    position_config = PositionConfig()
    
    system = TemporalWeightingSystem(
        decay_config=decay_config,
        importance_config=importance_config,
        position_config=position_config,
        enable_caching=True
    )
    
    logger.info(f"Created default temporal weighting system with {decay_type} decay")
    
    return system


def calculate_simple_temporal_weights(
    observation_dates: List[str],
    current_date: str,
    position: Position = "MID",
    decay_type: DecayType = "exponential"
) -> Array:
    """
    Quick utility for calculating simple temporal weights.
    
    Args:
        observation_dates: List of observation date strings
        current_date: Current date string
        position: Player position
        decay_type: Decay function type
        
    Returns:
        Array of normalized temporal weights
    """
    system = create_default_weighting_system(decay_type)
    
    result = system.calculate_weights(
        observation_dates=observation_dates,
        current_date=current_date,
        position=position
    )
    
    return result.normalized_weights


# Integration helpers for existing systems

def extend_measurement_processor_with_temporal_weights(
    measurement_processor,
    weighting_system: TemporalWeightingSystem
):
    """
    Extend MeasurementProcessor to apply temporal weights.
    
    Args:
        measurement_processor: Existing MeasurementProcessor instance
        weighting_system: TemporalWeightingSystem instance
    """
    original_normalize_features = measurement_processor.normalize_features
    
    def temporal_weighted_normalize_features(
        features, position, opponent_strength=None, is_home=True, gameweek=1,
        observation_dates=None, current_date=None, **kwargs
    ):
        # Get base normalized features
        normalized = original_normalize_features(
            features, position, opponent_strength, is_home, gameweek, **kwargs
        )
        
        # Apply temporal weights if dates provided
        if observation_dates and current_date:
            result = weighting_system.calculate_weights(
                observation_dates=observation_dates,
                current_date=current_date,
                position=position
            )
            
            # Apply weights
            if len(result.normalized_weights) == len(normalized):
                normalized = weighting_system.apply_weights(
                    normalized.reshape(1, -1),
                    result.normalized_weights.reshape(1, -1),
                    method="multiply"
                ).flatten()
        
        return normalized
    
    # Replace method
    measurement_processor.normalize_features = temporal_weighted_normalize_features
    
    logger.info("Extended MeasurementProcessor with temporal weighting")


def create_temporal_weighted_kalman_update(
    kalman_filter,
    weighting_system: TemporalWeightingSystem
):
    """
    Create temporal weight-aware Kalman filter update function.
    
    Args:
        kalman_filter: Kalman filter instance
        weighting_system: TemporalWeightingSystem instance
        
    Returns:
        Modified update function
    """
    original_update = kalman_filter.update
    
    def temporal_weighted_update(
        state, observation, noise_matrix, 
        observation_dates=None, current_date=None, position="MID", **kwargs
    ):
        # Calculate temporal weights if dates provided
        if observation_dates and current_date:
            result = weighting_system.calculate_weights(
                observation_dates=observation_dates,
                current_date=current_date,
                position=position
            )
            
            # Adjust noise matrix based on weights
            # Lower weights = higher uncertainty = higher noise
            if len(result.normalized_weights) > 0:
                avg_weight = jnp.mean(result.normalized_weights)
                weight_factor = 1.0 / jnp.maximum(avg_weight, 0.1)
                noise_matrix = noise_matrix * weight_factor
        
        # Perform standard update with adjusted noise
        return original_update(state, observation, noise_matrix, **kwargs)
    
    return temporal_weighted_update