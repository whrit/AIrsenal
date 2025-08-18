"""
Measurement Update System for AIrsenal's Kalman Filter Implementation

This module provides comprehensive measurement processing and update capabilities for
AIrsenal's Kalman filter-based player state tracking system. It handles the conversion
of raw match observations to measurement vectors, estimates measurement noise adaptively,
detects and handles outliers, and provides efficient batch update mechanisms.

Key Features:
- Match data to measurement conversion with position and opponent adjustments
- Adaptive noise estimation based on historical prediction errors
- Statistical outlier detection with robust estimation techniques
- Innovation monitoring for filter consistency and adaptive tuning
- Efficient batch processing for multiple players and gameweeks
- Integration with existing Kalman filter infrastructure

Classes:
    MeasurementProcessor: Convert raw match data to normalized measurements
    NoiseEstimator: Estimate measurement noise per feature and position
    OutlierDetector: Identify and handle anomalous observations
    InnovationMonitor: Track prediction errors for adaptive filtering
    BatchUpdater: Efficient updates for multiple players/gameweeks

Usage:
    ```python
    from airsenal.framework.measurement_update import (
        MeasurementProcessor, NoiseEstimator, BatchUpdater
    )
    
    # Initialize components
    processor = MeasurementProcessor()
    noise_estimator = NoiseEstimator()
    batch_updater = BatchUpdater(processor, noise_estimator)
    
    # Process batch of player scores
    measurements, noise_cov = batch_updater.process_batch(
        player_scores, gameweek=15, season="2023"
    )
    
    # Update Kalman filters
    updated_states = batch_updater.update_batch(
        kalman_model, measurements, noise_cov
    )
    ```
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from jax.scipy import stats
from jax.scipy.linalg import cholesky
from scipy import stats as scipy_stats
from sklearn.preprocessing import RobustScaler
from sqlalchemy.orm.session import Session

from airsenal.framework.schema import PlayerScore, Player, Fixture, Result
from airsenal.framework.kalman_filter import FilterState, FilterConfig
from airsenal.framework.kalman_player_model import KalmanPlayerModel

logger = logging.getLogger(__name__)

# Type aliases
Array = Union[np.ndarray, jnp.ndarray]
MeasurementVector = Array
NoiseMatrix = Array
PlayerData = Dict[str, Any]
BatchData = Dict[str, Any]


@dataclass
class MeasurementConfig:
    """Configuration for measurement processing."""
    
    # Feature selection and ordering
    measurement_features: List[str] = field(default_factory=lambda: [
        "goals", "assists", "minutes", "bonus", "expected_goals", "expected_assists"
    ])
    
    # Normalization parameters
    normalize_by_position: bool = True
    normalize_by_opponent: bool = True
    normalize_by_home_away: bool = True
    
    # Missing data handling
    missing_data_strategy: str = "interpolate"  # "skip", "interpolate", "default"
    default_minutes: float = 0.0
    min_minutes_for_valid: float = 10.0
    
    # Opponent strength adjustment
    use_opponent_adjustment: bool = True
    opponent_strength_source: str = "fifa_ratings"  # "fifa_ratings", "league_position"
    
    # Position-specific scaling factors
    position_scalings: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "GK": {"goals": 0.1, "assists": 0.3, "minutes": 1.0, "bonus": 0.8, 
               "expected_goals": 0.1, "expected_assists": 0.3},
        "DEF": {"goals": 0.5, "assists": 0.7, "minutes": 1.0, "bonus": 0.9,
                "expected_goals": 0.5, "expected_assists": 0.7},
        "MID": {"goals": 1.0, "assists": 1.0, "minutes": 1.0, "bonus": 1.0,
                "expected_goals": 1.0, "expected_assists": 1.0},
        "FWD": {"goals": 1.2, "assists": 0.8, "minutes": 0.9, "bonus": 1.1,
                "expected_goals": 1.2, "expected_assists": 0.8}
    })
    
    # Outlier detection parameters
    outlier_detection_method: str = "mahalanobis"  # "zscore", "iqr", "mahalanobis"
    outlier_threshold: float = 3.0
    max_outlier_ratio: float = 0.1  # Maximum fraction of outliers to handle
    
    # Robustness parameters
    use_robust_scaling: bool = True
    winsorize_percentile: float = 0.05  # Clip extreme values at this percentile


@dataclass
class NoiseConfig:
    """Configuration for noise estimation."""
    
    # Estimation window and adaptation
    estimation_window: int = 10
    adaptation_rate: float = 0.1
    min_samples_for_estimation: int = 3
    
    # Position-specific base noise levels
    base_noise_levels: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "GK": {"goals": 0.3, "assists": 0.4, "minutes": 0.2, "bonus": 0.5,
               "expected_goals": 0.25, "expected_assists": 0.35},
        "DEF": {"goals": 0.6, "assists": 0.5, "minutes": 0.3, "bonus": 0.6,
                "expected_goals": 0.55, "expected_assists": 0.45},
        "MID": {"goals": 0.8, "assists": 0.7, "minutes": 0.4, "bonus": 0.7,
                "expected_goals": 0.75, "expected_assists": 0.65},
        "FWD": {"goals": 1.0, "assists": 0.8, "minutes": 0.5, "bonus": 0.8,
                "expected_goals": 0.95, "expected_assists": 0.75}
    })
    
    # Temporal adaptation factors
    early_season_factor: float = 1.5  # Higher noise early in season
    late_season_factor: float = 0.8   # Lower noise late in season
    fixture_congestion_factor: float = 1.2  # Higher noise during busy periods
    
    # Correlation structure
    estimate_correlations: bool = True
    min_correlation_magnitude: float = 0.1


@dataclass
class OutlierResult:
    """Result of outlier detection."""
    is_outlier: bool
    outlier_score: float
    outlier_type: str  # "positive", "negative", "multivariate"
    confidence: float
    recommended_action: str  # "reject", "robust_update", "gradual_trust"


@dataclass
class InnovationStats:
    """Innovation statistics for monitoring filter performance."""
    innovation_sequence: Array
    innovation_covariance: Array
    normalized_innovation_squared: float  # NIS test statistic
    whiteness_test_statistic: float
    consistency_check_passed: bool
    degrees_of_freedom: int
    timestamp: float


class MeasurementProcessor:
    """
    Convert raw match observations to normalized measurement vectors.
    
    Handles position-specific scaling, opponent strength adjustment,
    home/away effects, and missing data interpolation.
    """
    
    def __init__(self, config: Optional[MeasurementConfig] = None):
        """Initialize measurement processor."""
        self.config = config or MeasurementConfig()
        
        # Caching for efficiency
        self._opponent_strengths: Dict[str, float] = {}
        self._position_scalers: Dict[str, RobustScaler] = {}
        self._feature_history: Dict[str, List[Array]] = defaultdict(list)
        
        # Normalization statistics
        self._global_stats: Dict[str, Dict[str, float]] = {}
        self._position_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        logger.info(f"Initialized MeasurementProcessor with {len(self.config.measurement_features)} features")
    
    def extract_features(self, player_score: PlayerScore) -> Dict[str, float]:
        """
        Extract features from PlayerScore object.
        
        Args:
            player_score: PlayerScore database record
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        for feature_name in self.config.measurement_features:
            if hasattr(player_score, feature_name):
                value = getattr(player_score, feature_name)
                features[feature_name] = float(value) if value is not None else np.nan
            else:
                # Handle derived features or missing attributes
                if feature_name == "expected_goals":
                    features[feature_name] = float(player_score.expected_goals or 0.0)
                elif feature_name == "expected_assists":
                    features[feature_name] = float(player_score.expected_assists or 0.0)
                elif feature_name == "bonus":
                    features[feature_name] = float(player_score.bonus or 0.0)
                else:
                    features[feature_name] = np.nan
        
        return features
    
    def normalize_features(
        self, 
        features: Dict[str, float], 
        position: str,
        opponent_strength: Optional[float] = None,
        is_home: bool = True,
        gameweek: int = 1
    ) -> MeasurementVector:
        """
        Normalize features based on position, opponent, and context.
        
        Args:
            features: Raw feature values
            position: Player position (GK, DEF, MID, FWD)
            opponent_strength: Strength of opponent team (0-1)
            is_home: Whether playing at home
            gameweek: Gameweek number for temporal adjustments
            
        Returns:
            Normalized measurement vector
        """
        try:
            normalized = []
            
            for feature_name in self.config.measurement_features:
                raw_value = features.get(feature_name, np.nan)
                
                if np.isnan(raw_value):
                    # Handle missing data
                    normalized_value = self._handle_missing_data(feature_name, position)
                else:
                    # Apply position scaling
                    if self.config.normalize_by_position and position in self.config.position_scalings:
                        scaling_factor = self.config.position_scalings[position].get(feature_name, 1.0)
                        normalized_value = raw_value * scaling_factor
                    else:
                        normalized_value = raw_value
                    
                    # Apply opponent strength adjustment
                    if self.config.normalize_by_opponent and opponent_strength is not None:
                        normalized_value = self._adjust_for_opponent(
                            normalized_value, feature_name, opponent_strength
                        )
                    
                    # Apply home/away adjustment
                    if self.config.normalize_by_home_away:
                        normalized_value = self._adjust_for_home_away(
                            normalized_value, feature_name, is_home
                        )
                    
                    # Apply temporal adjustments
                    normalized_value = self._adjust_for_gameweek(
                        normalized_value, feature_name, gameweek
                    )
                
                normalized.append(normalized_value)
            
            # Convert to JAX array
            measurement_vector = jnp.array(normalized, dtype=jnp.float32)
            
            # Apply winsorization if configured
            if self.config.use_robust_scaling and self.config.winsorize_percentile > 0:
                measurement_vector = self._winsorize_measurements(measurement_vector)
            
            return measurement_vector
            
        except Exception as e:
            logger.error(f"Feature normalization failed: {e}")
            # Return default measurement vector
            return jnp.zeros(len(self.config.measurement_features))
    
    def _handle_missing_data(self, feature_name: str, position: str) -> float:
        """Handle missing data based on strategy."""
        if self.config.missing_data_strategy == "skip":
            return np.nan
        elif self.config.missing_data_strategy == "default":
            if feature_name == "minutes":
                return self.config.default_minutes
            else:
                return 0.0
        elif self.config.missing_data_strategy == "interpolate":
            # Use position-specific historical average
            return self._get_position_average(feature_name, position)
        else:
            return 0.0
    
    def _adjust_for_opponent(self, value: float, feature_name: str, opponent_strength: float) -> float:
        """Adjust measurement based on opponent strength."""
        # Stronger opponents make it harder to score (negative features)
        # but might lead to more defensive actions (positive for defenders)
        if feature_name in ["goals", "assists", "expected_goals", "expected_assists"]:
            # Harder to score against strong teams
            difficulty_factor = 0.7 + 0.6 * (1 - opponent_strength)
            return value * difficulty_factor
        elif feature_name in ["minutes"]:
            # Minutes less affected by opponent strength
            return value * (0.95 + 0.1 * (1 - opponent_strength))
        else:
            return value
    
    def _adjust_for_home_away(self, value: float, feature_name: str, is_home: bool) -> float:
        """Adjust measurement for home/away effects."""
        if not is_home:
            # Away penalty for attacking stats
            if feature_name in ["goals", "assists", "expected_goals", "expected_assists"]:
                return value * 0.95
            else:
                return value * 0.98
        return value
    
    def _adjust_for_gameweek(self, value: float, feature_name: str, gameweek: int) -> float:
        """Apply temporal adjustments based on gameweek."""
        # Early season adjustment (higher variance)
        if gameweek <= 5:
            return value * 1.1
        # Mid-season stability
        elif gameweek <= 30:
            return value
        # Late season effects (fatigue, rotation)
        else:
            return value * 0.95
    
    def _winsorize_measurements(self, measurements: MeasurementVector) -> MeasurementVector:
        """Apply winsorization to clip extreme values."""
        # For each feature, clip at specified percentiles based on history
        clipped = measurements
        for i, feature_name in enumerate(self.config.measurement_features):
            if feature_name in self._feature_history and len(self._feature_history[feature_name]) > 10:
                history = jnp.array(self._feature_history[feature_name])
                lower_bound = jnp.percentile(history, self.config.winsorize_percentile * 100)
                upper_bound = jnp.percentile(history, (1 - self.config.winsorize_percentile) * 100)
                clipped = clipped.at[i].set(jnp.clip(measurements[i], lower_bound, upper_bound))
        
        return clipped
    
    def _get_position_average(self, feature_name: str, position: str) -> float:
        """Get historical average for feature by position."""
        if position in self._position_stats and feature_name in self._position_stats[position]:
            return self._position_stats[position][feature_name].get("mean", 0.0)
        return 0.0
    
    def update_statistics(
        self, 
        measurements: List[MeasurementVector], 
        positions: List[str]
    ):
        """Update normalization statistics with new data."""
        try:
            # Convert to numpy for statistical calculations
            measurements_array = np.array(measurements)
            
            # Update global statistics
            for i, feature_name in enumerate(self.config.measurement_features):
                feature_values = measurements_array[:, i]
                valid_values = feature_values[~np.isnan(feature_values)]
                
                if len(valid_values) > 0:
                    if feature_name not in self._global_stats:
                        self._global_stats[feature_name] = {}
                    
                    self._global_stats[feature_name]["mean"] = np.mean(valid_values)
                    self._global_stats[feature_name]["std"] = np.std(valid_values)
                    self._global_stats[feature_name]["count"] = len(valid_values)
                    
                    # Update feature history for winsorization
                    self._feature_history[feature_name].extend(valid_values.tolist())
                    if len(self._feature_history[feature_name]) > 1000:
                        self._feature_history[feature_name] = self._feature_history[feature_name][-1000:]
            
            # Update position-specific statistics
            for pos in set(positions):
                if pos not in self._position_stats:
                    self._position_stats[pos] = {}
                
                pos_mask = np.array(positions) == pos
                pos_measurements = measurements_array[pos_mask]
                
                for i, feature_name in enumerate(self.config.measurement_features):
                    feature_values = pos_measurements[:, i]
                    valid_values = feature_values[~np.isnan(feature_values)]
                    
                    if len(valid_values) > 0:
                        if feature_name not in self._position_stats[pos]:
                            self._position_stats[pos][feature_name] = {}
                        
                        self._position_stats[pos][feature_name]["mean"] = np.mean(valid_values)
                        self._position_stats[pos][feature_name]["std"] = np.std(valid_values)
                        self._position_stats[pos][feature_name]["count"] = len(valid_values)
            
            logger.debug(f"Updated statistics with {len(measurements)} measurements")
            
        except Exception as e:
            logger.warning(f"Failed to update statistics: {e}")
    
    def get_opponent_strength(self, opponent_team: str, season: str) -> float:
        """Get opponent strength rating (0=weak, 1=strong)."""
        # This would typically query FIFA ratings or league position
        # For now, return a placeholder implementation
        cache_key = f"{opponent_team}_{season}"
        if cache_key not in self._opponent_strengths:
            # Placeholder: random strength between 0.3-0.8
            # In practice, this would query actual team strength data
            strength = np.random.uniform(0.3, 0.8)
            self._opponent_strengths[cache_key] = strength
        
        return self._opponent_strengths[cache_key]


class NoiseEstimator:
    """
    Estimate measurement noise from historical data and prediction errors.
    
    Provides position-specific, time-varying noise estimates with correlation
    structure for robust Kalman filtering.
    """
    
    def __init__(self, config: Optional[NoiseConfig] = None):
        """Initialize noise estimator."""
        self.config = config or NoiseConfig()
        
        # Storage for innovation history
        self._innovation_history: Dict[str, List[Array]] = defaultdict(list)  # by position
        self._measurement_history: Dict[str, List[Array]] = defaultdict(list)
        
        # Current noise estimates
        self._noise_matrices: Dict[str, NoiseMatrix] = {}
        self._correlation_matrices: Dict[str, Array] = {}
        
        # Temporal factors
        self._temporal_adjustments: Dict[int, float] = {}
        
        logger.info("Initialized NoiseEstimator with adaptive estimation")
    
    def estimate_noise_matrix(
        self, 
        position: str, 
        gameweek: int = 1,
        innovation_sequence: Optional[List[Array]] = None
    ) -> NoiseMatrix:
        """
        Estimate measurement noise matrix for a position.
        
        Args:
            position: Player position
            gameweek: Current gameweek for temporal adjustment
            innovation_sequence: Recent innovation vectors
            
        Returns:
            Measurement noise covariance matrix
        """
        try:
            # Start with base noise levels
            base_noise = self.config.base_noise_levels.get(position, 
                                                          self.config.base_noise_levels["MID"])
            
            n_features = len(base_noise)
            noise_matrix = jnp.eye(n_features)
            
            # Set diagonal elements from base noise
            for i, feature_name in enumerate(base_noise.keys()):
                noise_matrix = noise_matrix.at[i, i].set(base_noise[feature_name] ** 2)
            
            # Apply temporal adjustments
            temporal_factor = self._get_temporal_factor(gameweek)
            noise_matrix = noise_matrix * temporal_factor
            
            # Adaptive adjustment from innovation history
            if innovation_sequence or position in self._innovation_history:
                innovations = innovation_sequence or self._innovation_history[position]
                if len(innovations) >= self.config.min_samples_for_estimation:
                    adaptive_noise = self._estimate_from_innovations(innovations)
                    
                    # Blend base and adaptive estimates
                    alpha = self.config.adaptation_rate
                    noise_matrix = (1 - alpha) * noise_matrix + alpha * adaptive_noise
            
            # Add correlation structure if enabled
            if self.config.estimate_correlations and position in self._correlation_matrices:
                correlation_matrix = self._correlation_matrices[position]
                # Convert covariance to correlation and back
                std_diag = jnp.sqrt(jnp.diag(noise_matrix))
                noise_matrix = jnp.outer(std_diag, std_diag) * correlation_matrix
            
            # Ensure positive definiteness
            noise_matrix = self._ensure_positive_definite(noise_matrix)
            
            # Cache for efficiency
            self._noise_matrices[position] = noise_matrix
            
            return noise_matrix
            
        except Exception as e:
            logger.error(f"Noise estimation failed for position {position}: {e}")
            # Fall back to identity matrix
            n_features = len(self.config.base_noise_levels.get(position, 
                                                              self.config.base_noise_levels["MID"]))
            return jnp.eye(n_features) * 0.5
    
    def _get_temporal_factor(self, gameweek: int) -> float:
        """Get temporal adjustment factor for gameweek."""
        if gameweek in self._temporal_adjustments:
            return self._temporal_adjustments[gameweek]
        
        # Early season: higher noise
        if gameweek <= 5:
            factor = self.config.early_season_factor
        # Late season: lower noise  
        elif gameweek >= 35:
            factor = self.config.late_season_factor
        # Mid-season: baseline
        else:
            factor = 1.0
        
        # Fixture congestion periods (assume every 3rd gameweek)
        if gameweek % 3 == 0:
            factor *= self.config.fixture_congestion_factor
        
        self._temporal_adjustments[gameweek] = factor
        return factor
    
    def _estimate_from_innovations(self, innovations: List[Array]) -> NoiseMatrix:
        """Estimate noise from innovation sequence."""
        try:
            # Convert to JAX array
            innovations_array = jnp.array(innovations[-self.config.estimation_window:])
            
            # Compute empirical covariance
            empirical_cov = jnp.cov(innovations_array.T)
            
            # Regularize to ensure numerical stability
            regularization = jnp.eye(empirical_cov.shape[0]) * 1e-6
            empirical_cov += regularization
            
            return empirical_cov
            
        except Exception as e:
            logger.warning(f"Innovation-based estimation failed: {e}")
            # Fall back to identity
            n_features = len(innovations[0]) if innovations else 4
            return jnp.eye(n_features) * 0.5
    
    def _ensure_positive_definite(self, matrix: NoiseMatrix) -> NoiseMatrix:
        """Ensure matrix is positive definite."""
        try:
            # Eigendecomposition
            eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
            
            # Clip negative eigenvalues
            min_eigenval = 1e-8
            eigenvals = jnp.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            return eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
            
        except Exception as e:
            logger.warning(f"PSD enforcement failed: {e}")
            # Add regularization
            return matrix + jnp.eye(matrix.shape[0]) * 1e-6
    
    def update_noise_estimates(
        self, 
        innovations: Dict[str, Array],
        measurements: Dict[str, Array]
    ):
        """
        Update noise estimates with new innovation and measurement data.
        
        Args:
            innovations: Innovation vectors by position
            measurements: Measurement vectors by position
        """
        try:
            for position, innovation in innovations.items():
                # Store innovation history
                self._innovation_history[position].append(innovation)
                if len(self._innovation_history[position]) > self.config.estimation_window * 2:
                    self._innovation_history[position] = self._innovation_history[position][-self.config.estimation_window:]
                
                # Store measurement history for correlation estimation
                if position in measurements:
                    self._measurement_history[position].append(measurements[position])
                    if len(self._measurement_history[position]) > self.config.estimation_window * 2:
                        self._measurement_history[position] = self._measurement_history[position][-self.config.estimation_window:]
            
            # Update correlation matrices if enabled
            if self.config.estimate_correlations:
                self._update_correlation_matrices()
            
            logger.debug(f"Updated noise estimates for {len(innovations)} positions")
            
        except Exception as e:
            logger.warning(f"Noise estimate update failed: {e}")
    
    def _update_correlation_matrices(self):
        """Update correlation structure from measurement history."""
        for position, measurements in self._measurement_history.items():
            if len(measurements) >= self.config.min_samples_for_estimation:
                try:
                    measurements_array = jnp.array(measurements)
                    correlation_matrix = jnp.corrcoef(measurements_array.T)
                    
                    # Filter weak correlations
                    correlation_matrix = jnp.where(
                        jnp.abs(correlation_matrix) < self.config.min_correlation_magnitude,
                        0.0,
                        correlation_matrix
                    )
                    
                    # Ensure diagonal is 1
                    correlation_matrix = correlation_matrix.at[jnp.diag_indices_from(correlation_matrix)].set(1.0)
                    
                    self._correlation_matrices[position] = correlation_matrix
                    
                except Exception as e:
                    logger.warning(f"Correlation estimation failed for {position}: {e}")


class OutlierDetector:
    """
    Statistical outlier detection and robust handling for measurement updates.
    
    Implements multiple detection methods and provides recommendations for
    handling outliers (reject, robust update, gradual trust increase).
    """
    
    def __init__(self, config: Optional[MeasurementConfig] = None):
        """Initialize outlier detector."""
        self.config = config or MeasurementConfig()
        
        # Outlier detection history for adaptive thresholds
        self._outlier_history: Dict[str, List[OutlierResult]] = defaultdict(list)
        self._feature_distributions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        logger.info(f"Initialized OutlierDetector with {self.config.outlier_detection_method} method")
    
    def detect_outliers(
        self, 
        measurement: MeasurementVector,
        position: str,
        historical_measurements: Optional[List[MeasurementVector]] = None,
        predicted_measurement: Optional[MeasurementVector] = None
    ) -> OutlierResult:
        """
        Detect if measurement is an outlier.
        
        Args:
            measurement: Current measurement vector
            position: Player position
            historical_measurements: Historical measurements for comparison
            predicted_measurement: Expected measurement from model
            
        Returns:
            OutlierResult with detection details and recommendations
        """
        try:
            # Choose detection method
            if self.config.outlier_detection_method == "zscore":
                result = self._zscore_detection(measurement, historical_measurements)
            elif self.config.outlier_detection_method == "iqr":
                result = self._iqr_detection(measurement, historical_measurements)
            elif self.config.outlier_detection_method == "mahalanobis":
                result = self._mahalanobis_detection(measurement, historical_measurements)
            else:
                # Default to z-score
                result = self._zscore_detection(measurement, historical_measurements)
            
            # Enhance with prediction-based detection if available
            if predicted_measurement is not None:
                prediction_result = self._prediction_based_detection(measurement, predicted_measurement)
                # Combine results
                result = self._combine_outlier_results(result, prediction_result)
            
            # Determine recommended action
            result.recommended_action = self._get_recommended_action(result, position)
            
            # Update outlier history
            self._outlier_history[position].append(result)
            if len(self._outlier_history[position]) > 100:
                self._outlier_history[position] = self._outlier_history[position][-100:]
            
            return result
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return OutlierResult(
                is_outlier=False,
                outlier_score=0.0,
                outlier_type="unknown",
                confidence=0.0,
                recommended_action="robust_update"
            )
    
    def _zscore_detection(
        self, 
        measurement: MeasurementVector,
        historical_measurements: Optional[List[MeasurementVector]]
    ) -> OutlierResult:
        """Z-score based outlier detection."""
        if not historical_measurements or len(historical_measurements) < 3:
            return OutlierResult(
                is_outlier=False,
                outlier_score=0.0,
                outlier_type="insufficient_data",
                confidence=0.0,
                recommended_action="robust_update"
            )
        
        try:
            historical_array = jnp.array(historical_measurements)
            
            # Compute z-scores for each feature
            means = jnp.mean(historical_array, axis=0)
            stds = jnp.std(historical_array, axis=0) + 1e-8  # Avoid division by zero
            
            z_scores = jnp.abs((measurement - means) / stds)
            max_z_score = jnp.max(z_scores)
            
            is_outlier = max_z_score > self.config.outlier_threshold
            outlier_type = "positive" if jnp.any(measurement > means + self.config.outlier_threshold * stds) else "negative"
            
            return OutlierResult(
                is_outlier=bool(is_outlier),
                outlier_score=float(max_z_score),
                outlier_type=outlier_type,
                confidence=min(float(max_z_score / self.config.outlier_threshold), 1.0),
                recommended_action=""
            )
            
        except Exception as e:
            logger.warning(f"Z-score detection failed: {e}")
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="error", 
                               confidence=0.0, recommended_action="robust_update")
    
    def _iqr_detection(
        self, 
        measurement: MeasurementVector,
        historical_measurements: Optional[List[MeasurementVector]]
    ) -> OutlierResult:
        """Interquartile range based outlier detection."""
        if not historical_measurements or len(historical_measurements) < 5:
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="insufficient_data",
                               confidence=0.0, recommended_action="robust_update")
        
        try:
            historical_array = jnp.array(historical_measurements)
            
            # Compute IQR for each feature
            q25 = jnp.percentile(historical_array, 25, axis=0)
            q75 = jnp.percentile(historical_array, 75, axis=0)
            iqr = q75 - q25
            
            # Outlier bounds
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Check for outliers
            is_below = measurement < lower_bound
            is_above = measurement > upper_bound
            is_outlier_vec = is_below | is_above
            
            is_outlier = jnp.any(is_outlier_vec)
            
            # Compute outlier score as maximum deviation
            deviation_below = jnp.maximum(0, lower_bound - measurement) / (iqr + 1e-8)
            deviation_above = jnp.maximum(0, measurement - upper_bound) / (iqr + 1e-8)
            outlier_score = jnp.max(jnp.maximum(deviation_below, deviation_above))
            
            outlier_type = "negative" if jnp.any(is_below) else "positive"
            
            return OutlierResult(
                is_outlier=bool(is_outlier),
                outlier_score=float(outlier_score),
                outlier_type=outlier_type,
                confidence=min(float(outlier_score), 1.0),
                recommended_action=""
            )
            
        except Exception as e:
            logger.warning(f"IQR detection failed: {e}")
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="error",
                               confidence=0.0, recommended_action="robust_update")
    
    def _mahalanobis_detection(
        self, 
        measurement: MeasurementVector,
        historical_measurements: Optional[List[MeasurementVector]]
    ) -> OutlierResult:
        """Mahalanobis distance based multivariate outlier detection."""
        if not historical_measurements or len(historical_measurements) < len(measurement) + 2:
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="insufficient_data",
                               confidence=0.0, recommended_action="robust_update")
        
        try:
            historical_array = jnp.array(historical_measurements)
            
            # Compute mean and covariance
            mean = jnp.mean(historical_array, axis=0)
            cov = jnp.cov(historical_array.T)
            
            # Regularize covariance matrix
            cov += jnp.eye(cov.shape[0]) * 1e-8
            
            # Compute Mahalanobis distance
            diff = measurement - mean
            inv_cov = jnp.linalg.inv(cov)
            mahal_dist = jnp.sqrt(diff.T @ inv_cov @ diff)
            
            # Chi-square threshold for given degrees of freedom
            dof = len(measurement)
            threshold = scipy_stats.chi2.ppf(0.95, dof)
            
            is_outlier = mahal_dist > threshold
            
            return OutlierResult(
                is_outlier=bool(is_outlier),
                outlier_score=float(mahal_dist),
                outlier_type="multivariate",
                confidence=min(float(mahal_dist / threshold), 1.0),
                recommended_action=""
            )
            
        except Exception as e:
            logger.warning(f"Mahalanobis detection failed: {e}")
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="error",
                               confidence=0.0, recommended_action="robust_update")
    
    def _prediction_based_detection(
        self, 
        measurement: MeasurementVector,
        predicted_measurement: MeasurementVector
    ) -> OutlierResult:
        """Detect outliers based on prediction residuals."""
        try:
            residual = measurement - predicted_measurement
            residual_norm = jnp.linalg.norm(residual)
            
            # Simple threshold based on residual magnitude
            threshold = 2.0  # This could be adaptive
            is_outlier = residual_norm > threshold
            
            outlier_type = "prediction_mismatch"
            if jnp.any(residual > 1.0):
                outlier_type = "positive_prediction_mismatch"
            elif jnp.any(residual < -1.0):
                outlier_type = "negative_prediction_mismatch"
            
            return OutlierResult(
                is_outlier=bool(is_outlier),
                outlier_score=float(residual_norm),
                outlier_type=outlier_type,
                confidence=min(float(residual_norm / threshold), 1.0),
                recommended_action=""
            )
            
        except Exception as e:
            logger.warning(f"Prediction-based detection failed: {e}")
            return OutlierResult(is_outlier=False, outlier_score=0.0, outlier_type="error",
                               confidence=0.0, recommended_action="robust_update")
    
    def _combine_outlier_results(self, result1: OutlierResult, result2: OutlierResult) -> OutlierResult:
        """Combine multiple outlier detection results."""
        # Use maximum outlier score and OR logic for is_outlier
        combined_score = max(result1.outlier_score, result2.outlier_score)
        is_outlier = result1.is_outlier or result2.is_outlier
        confidence = max(result1.confidence, result2.confidence)
        
        # Combine outlier types
        if result1.outlier_type == result2.outlier_type:
            outlier_type = result1.outlier_type
        else:
            outlier_type = f"{result1.outlier_type}+{result2.outlier_type}"
        
        return OutlierResult(
            is_outlier=is_outlier,
            outlier_score=combined_score,
            outlier_type=outlier_type,
            confidence=confidence,
            recommended_action=""
        )
    
    def _get_recommended_action(self, result: OutlierResult, position: str) -> str:
        """Determine recommended action for handling outlier."""
        if not result.is_outlier:
            return "normal_update"
        
        # Check outlier history for this position
        recent_outliers = [r for r in self._outlier_history[position][-10:] if r.is_outlier]
        outlier_rate = len(recent_outliers) / min(len(self._outlier_history[position]), 10)
        
        # High confidence outliers
        if result.confidence > 0.8:
            if outlier_rate > self.config.max_outlier_ratio:
                # Too many recent outliers, might be regime change
                return "gradual_trust"
            else:
                # Clear outlier, reject or use robust method
                if result.outlier_type.startswith("positive"):
                    return "gradual_trust"  # Positive outliers (goals, assists) might be real
                else:
                    return "robust_update"
        
        # Medium confidence outliers
        elif result.confidence > 0.5:
            return "robust_update"
        
        # Low confidence outliers
        else:
            return "normal_update"


class InnovationMonitor:
    """
    Monitor innovation sequences for filter consistency and adaptive tuning.
    
    Tracks prediction errors, performs statistical tests for filter
    consistency, and provides adaptive filter parameter recommendations.
    """
    
    def __init__(self, window_size: int = 20):
        """Initialize innovation monitor."""
        self.window_size = window_size
        
        # Innovation storage
        self._innovation_sequences: Dict[str, List[Array]] = defaultdict(list)
        self._innovation_covariances: Dict[str, List[Array]] = defaultdict(list)
        
        # Test statistics history
        self._nis_history: Dict[str, List[float]] = defaultdict(list)
        self._whiteness_history: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Initialized InnovationMonitor with window size {window_size}")
    
    def add_innovation(
        self, 
        innovation: Array,
        innovation_covariance: Array,
        position: str,
        timestamp: float
    ) -> InnovationStats:
        """
        Add new innovation and compute statistics.
        
        Args:
            innovation: Innovation vector (observation - prediction)
            innovation_covariance: Innovation covariance matrix
            position: Player position for position-specific monitoring
            timestamp: Timestamp of observation
            
        Returns:
            InnovationStats with test results
        """
        try:
            # Store innovation
            self._innovation_sequences[position].append(innovation)
            self._innovation_covariances[position].append(innovation_covariance)
            
            # Maintain window size
            if len(self._innovation_sequences[position]) > self.window_size:
                self._innovation_sequences[position] = self._innovation_sequences[position][-self.window_size:]
                self._innovation_covariances[position] = self._innovation_covariances[position][-self.window_size:]
            
            # Compute statistics
            stats = self._compute_innovation_stats(position, timestamp)
            
            # Store test statistics
            self._nis_history[position].append(stats.normalized_innovation_squared)
            self._whiteness_history[position].append(stats.whiteness_test_statistic)
            
            # Maintain history
            if len(self._nis_history[position]) > self.window_size:
                self._nis_history[position] = self._nis_history[position][-self.window_size:]
                self._whiteness_history[position] = self._whiteness_history[position][-self.window_size:]
            
            return stats
            
        except Exception as e:
            logger.error(f"Innovation monitoring failed: {e}")
            # Return dummy stats
            return InnovationStats(
                innovation_sequence=innovation,
                innovation_covariance=innovation_covariance,
                normalized_innovation_squared=0.0,
                whiteness_test_statistic=0.0,
                consistency_check_passed=True,
                degrees_of_freedom=len(innovation),
                timestamp=timestamp
            )
    
    def _compute_innovation_stats(self, position: str, timestamp: float) -> InnovationStats:
        """Compute innovation statistics for consistency testing."""
        try:
            innovations = self._innovation_sequences[position]
            covariances = self._innovation_covariances[position]
            
            if len(innovations) < 2:
                # Insufficient data
                return InnovationStats(
                    innovation_sequence=innovations[-1] if innovations else jnp.array([]),
                    innovation_covariance=covariances[-1] if covariances else jnp.eye(1),
                    normalized_innovation_squared=0.0,
                    whiteness_test_statistic=0.0,
                    consistency_check_passed=True,
                    degrees_of_freedom=1,
                    timestamp=timestamp
                )
            
            # Current innovation and covariance
            current_innovation = innovations[-1]
            current_covariance = covariances[-1]
            
            # Normalized Innovation Squared (NIS) test
            nis = self._compute_nis(current_innovation, current_covariance)
            
            # Whiteness test on innovation sequence
            whiteness_stat = self._compute_whiteness_test(innovations)
            
            # Consistency check
            dof = len(current_innovation)
            consistency_passed = self._check_consistency(nis, dof)
            
            return InnovationStats(
                innovation_sequence=jnp.array(innovations),
                innovation_covariance=current_covariance,
                normalized_innovation_squared=nis,
                whiteness_test_statistic=whiteness_stat,
                consistency_check_passed=consistency_passed,
                degrees_of_freedom=dof,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.warning(f"Innovation stats computation failed: {e}")
            # Return safe defaults
            return InnovationStats(
                innovation_sequence=jnp.array([0.0]),
                innovation_covariance=jnp.eye(1),
                normalized_innovation_squared=0.0,
                whiteness_test_statistic=0.0,
                consistency_check_passed=True,
                degrees_of_freedom=1,
                timestamp=timestamp
            )
    
    def _compute_nis(self, innovation: Array, innovation_covariance: Array) -> float:
        """Compute Normalized Innovation Squared statistic."""
        try:
            # NIS = y^T * S^(-1) * y where y is innovation, S is covariance
            inv_cov = jnp.linalg.inv(innovation_covariance + jnp.eye(innovation_covariance.shape[0]) * 1e-8)
            nis = innovation.T @ inv_cov @ innovation
            return float(nis)
        except Exception as e:
            logger.warning(f"NIS computation failed: {e}")
            return 0.0
    
    def _compute_whiteness_test(self, innovations: List[Array]) -> float:
        """Compute whiteness test statistic for innovation sequence."""
        try:
            if len(innovations) < 5:
                return 0.0
            
            # Convert to array
            innovation_array = jnp.array(innovations)
            n_samples, n_features = innovation_array.shape
            
            # Compute sample autocorrelation at lag 1
            innovation_shifted = innovation_array[1:]
            innovation_lagged = innovation_array[:-1]
            
            # Correlation coefficient
            correlation_matrix = jnp.corrcoef(innovation_shifted.flatten(), innovation_lagged.flatten())
            autocorr = correlation_matrix[0, 1]
            
            # Test statistic (approximate)
            test_stat = jnp.sqrt(n_samples - 1) * jnp.abs(autocorr)
            
            return float(test_stat)
            
        except Exception as e:
            logger.warning(f"Whiteness test failed: {e}")
            return 0.0
    
    def _check_consistency(self, nis: float, dof: int) -> bool:
        """Check filter consistency using NIS test."""
        # Chi-square test at 95% confidence level
        lower_bound = scipy_stats.chi2.ppf(0.025, dof)
        upper_bound = scipy_stats.chi2.ppf(0.975, dof)
        
        return lower_bound <= nis <= upper_bound
    
    def get_adaptive_recommendations(self, position: str) -> Dict[str, Any]:
        """Get recommendations for adaptive filter tuning."""
        recommendations = {
            "increase_process_noise": False,
            "decrease_process_noise": False,
            "increase_measurement_noise": False,
            "decrease_measurement_noise": False,
            "filter_diverging": False,
            "confidence": 0.0
        }
        
        if position not in self._nis_history or len(self._nis_history[position]) < 5:
            return recommendations
        
        try:
            # Analyze recent NIS values
            recent_nis = self._nis_history[position][-10:]
            mean_nis = np.mean(recent_nis)
            
            # Expected NIS should be close to degrees of freedom
            expected_nis = len(self._innovation_sequences[position][-1]) if self._innovation_sequences[position] else 4
            
            confidence = min(len(recent_nis) / 10.0, 1.0)
            recommendations["confidence"] = confidence
            
            # Consistently high NIS suggests underestimated uncertainty
            if mean_nis > expected_nis * 1.5:
                recommendations["increase_measurement_noise"] = True
                recommendations["increase_process_noise"] = True
            
            # Consistently low NIS suggests overestimated uncertainty
            elif mean_nis < expected_nis * 0.5:
                recommendations["decrease_measurement_noise"] = True
                recommendations["decrease_process_noise"] = True
            
            # Check for filter divergence
            if any(nis > expected_nis * 3.0 for nis in recent_nis[-3:]):
                recommendations["filter_diverging"] = True
            
            # Analyze whiteness
            if position in self._whiteness_history:
                recent_whiteness = self._whiteness_history[position][-5:]
                if np.mean(recent_whiteness) > 2.0:  # Critical value for normal distribution
                    recommendations["increase_process_noise"] = True
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Adaptive recommendations failed for {position}: {e}")
            return recommendations


class BatchUpdater:
    """
    Efficient batch processing for multiple players and gameweeks.
    
    Provides vectorized operations and parallel processing capabilities
    for updating multiple Kalman filters simultaneously.
    """
    
    def __init__(
        self, 
        measurement_processor: MeasurementProcessor,
        noise_estimator: NoiseEstimator,
        outlier_detector: Optional[OutlierDetector] = None,
        innovation_monitor: Optional[InnovationMonitor] = None
    ):
        """Initialize batch updater."""
        self.measurement_processor = measurement_processor
        self.noise_estimator = noise_estimator
        self.outlier_detector = outlier_detector or OutlierDetector()
        self.innovation_monitor = innovation_monitor or InnovationMonitor()
        
        # Caching for efficiency
        self._cached_noise_matrices: Dict[str, NoiseMatrix] = {}
        self._batch_processing_stats = {
            "total_processed": 0,
            "outliers_detected": 0,
            "missing_data_handled": 0,
            "processing_time": 0.0
        }
        
        logger.info("Initialized BatchUpdater for efficient multi-player processing")
    
    def process_batch(
        self, 
        player_scores: List[PlayerScore],
        gameweek: int,
        season: str,
        session: Optional[Session] = None
    ) -> Tuple[Dict[int, MeasurementVector], Dict[str, NoiseMatrix]]:
        """
        Process batch of player scores into measurements and noise estimates.
        
        Args:
            player_scores: List of PlayerScore objects
            gameweek: Current gameweek
            season: Season identifier
            session: Database session for additional queries
            
        Returns:
            Tuple of (player_measurements, position_noise_matrices)
        """
        try:
            start_time = time.time()
            
            # Group by position for efficient processing
            position_groups = defaultdict(list)
            player_measurements = {}
            
            for score in player_scores:
                if score.player and score.player.position:
                    position = score.player.position
                    position_groups[position].append(score)
            
            # Process each position group
            position_noise_matrices = {}
            
            for position, scores in position_groups.items():
                # Extract features for all players in position
                batch_features = []
                player_ids = []
                
                for score in scores:
                    features = self.measurement_processor.extract_features(score)
                    
                    # Get opponent strength
                    opponent_strength = self.measurement_processor.get_opponent_strength(
                        score.opponent, season
                    )
                    
                    # Normalize features
                    measurement = self.measurement_processor.normalize_features(
                        features, position, opponent_strength,
                        is_home=self._is_home_game(score),
                        gameweek=gameweek
                    )
                    
                    batch_features.append(measurement)
                    player_measurements[score.player_id] = measurement
                    player_ids.append(score.player_id)
                
                # Update measurement processor statistics
                if batch_features:
                    self.measurement_processor.update_statistics(
                        batch_features, [position] * len(batch_features)
                    )
                
                # Estimate noise matrix for position
                noise_matrix = self.noise_estimator.estimate_noise_matrix(
                    position, gameweek
                )
                position_noise_matrices[position] = noise_matrix
                
                # Cache for efficiency
                self._cached_noise_matrices[f"{position}_{gameweek}"] = noise_matrix
            
            # Update processing statistics
            self._batch_processing_stats["total_processed"] += len(player_scores)
            self._batch_processing_stats["processing_time"] += time.time() - start_time
            
            logger.debug(f"Processed batch of {len(player_scores)} player scores in {time.time() - start_time:.3f}s")
            
            return player_measurements, position_noise_matrices
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {}, {}
    
    def update_batch(
        self,
        kalman_model: KalmanPlayerModel,
        player_measurements: Dict[int, MeasurementVector],
        position_noise_matrices: Dict[str, NoiseMatrix],
        player_positions: Dict[int, str],
        gameweek: int,
        season: str
    ) -> Dict[int, Any]:
        """
        Update Kalman filters for batch of players.
        
        Args:
            kalman_model: KalmanPlayerModel instance
            player_measurements: Measurements by player ID
            position_noise_matrices: Noise matrices by position
            player_positions: Position mapping for players
            gameweek: Current gameweek
            season: Season identifier
            
        Returns:
            Dictionary of updated player states
        """
        try:
            updated_states = {}
            outlier_counts = defaultdict(int)
            
            for player_id, measurement in player_measurements.items():
                try:
                    position = player_positions.get(player_id, "MID")
                    
                    # Get current player state or initialize
                    if player_id in kalman_model.player_states:
                        current_state = kalman_model.player_states[player_id]
                    else:
                        current_state = kalman_model.initialize_state(
                            player_id, None, gameweek, season, position
                        )
                    
                    # Predict state forward
                    predicted_state = kalman_model.predict_state(
                        current_state, gameweeks_ahead=1, position=position
                    )
                    
                    # Get predicted observation for outlier detection
                    predicted_obs = kalman_model.observation_model(
                        predicted_state.state_mean, position
                    )
                    
                    # Outlier detection
                    historical_measurements = self._get_historical_measurements(
                        player_id, position
                    )
                    
                    outlier_result = self.outlier_detector.detect_outliers(
                        measurement, position, historical_measurements, predicted_obs
                    )
                    
                    if outlier_result.is_outlier:
                        outlier_counts[position] += 1
                        self._batch_processing_stats["outliers_detected"] += 1
                        
                        # Handle outlier based on recommendation
                        if outlier_result.recommended_action == "reject":
                            # Skip update for this player
                            updated_states[player_id] = predicted_state
                            continue
                        elif outlier_result.recommended_action == "robust_update":
                            # Use inflated noise matrix
                            noise_matrix = position_noise_matrices[position] * 2.0
                        elif outlier_result.recommended_action == "gradual_trust":
                            # Gradually trust the observation
                            trust_factor = 0.5
                            measurement = trust_factor * measurement + (1 - trust_factor) * predicted_obs
                            noise_matrix = position_noise_matrices[position]
                        else:
                            noise_matrix = position_noise_matrices[position]
                    else:
                        noise_matrix = position_noise_matrices[position]
                    
                    # Update state with measurement
                    updated_state = kalman_model.update_state(
                        predicted_state, measurement, noise_matrix, position
                    )
                    
                    updated_states[player_id] = updated_state
                    
                    # Monitor innovation if enabled
                    if self.innovation_monitor:
                        innovation = measurement - predicted_obs
                        # Estimate innovation covariance (simplified)
                        H = jnp.eye(len(measurement), len(predicted_state.state_mean))
                        innovation_cov = H @ predicted_state.state_cov @ H.T + noise_matrix
                        
                        innovation_stats = self.innovation_monitor.add_innovation(
                            innovation, innovation_cov, position, float(gameweek)
                        )
                        
                        # Apply adaptive recommendations if needed
                        if not innovation_stats.consistency_check_passed:
                            recommendations = self.innovation_monitor.get_adaptive_recommendations(position)
                            # This could trigger filter recalibration
                            logger.info(f"Filter inconsistency detected for {position}: {recommendations}")
                    
                except Exception as e:
                    logger.warning(f"Failed to update player {player_id}: {e}")
                    continue
            
            # Update noise estimates with innovations
            self._update_noise_estimates_from_batch(
                player_measurements, position_noise_matrices, player_positions
            )
            
            logger.info(f"Updated {len(updated_states)} players, detected {sum(outlier_counts.values())} outliers")
            
            return updated_states
            
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            return {}
    
    def _is_home_game(self, player_score: PlayerScore) -> bool:
        """Determine if player is playing at home."""
        # This is a simplified check - in practice would need fixture data
        if hasattr(player_score, 'fixture') and player_score.fixture:
            return player_score.player_team == player_score.fixture.home_team
        return True  # Default assumption
    
    def _get_historical_measurements(
        self, 
        player_id: int, 
        position: str
    ) -> Optional[List[MeasurementVector]]:
        """Get historical measurements for outlier detection."""
        # This would typically query database for recent measurements
        # For now, return None to indicate no historical data
        return None
    
    def _update_noise_estimates_from_batch(
        self,
        player_measurements: Dict[int, MeasurementVector],
        position_noise_matrices: Dict[str, NoiseMatrix],
        player_positions: Dict[int, str]
    ):
        """Update noise estimates from batch processing results."""
        try:
            # Group measurements by position
            position_measurements = defaultdict(list)
            
            for player_id, measurement in player_measurements.items():
                position = player_positions.get(player_id, "MID")
                position_measurements[position].append(measurement)
            
            # Convert to format expected by noise estimator
            innovations = {}  # Would compute from actual vs predicted
            measurements = {pos: jnp.array(measurements) for pos, measurements in position_measurements.items()}
            
            self.noise_estimator.update_noise_estimates(innovations, measurements)
            
        except Exception as e:
            logger.warning(f"Noise estimate update failed: {e}")
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about batch processing performance."""
        stats = self._batch_processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["outlier_rate"] = stats["outliers_detected"] / stats["total_processed"]
            stats["avg_processing_time"] = stats["processing_time"] / stats["total_processed"]
        else:
            stats["outlier_rate"] = 0.0
            stats["avg_processing_time"] = 0.0
        
        # Add cache statistics
        stats["cached_noise_matrices"] = len(self._cached_noise_matrices)
        
        return stats
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._cached_noise_matrices.clear()
        logger.debug("Cleared batch updater cache")


# Utility functions for integration

def create_measurement_pipeline(
    measurement_config: Optional[MeasurementConfig] = None,
    noise_config: Optional[NoiseConfig] = None
) -> BatchUpdater:
    """
    Create a complete measurement processing pipeline.
    
    Args:
        measurement_config: Configuration for measurement processing
        noise_config: Configuration for noise estimation
        
    Returns:
        Configured BatchUpdater instance
    """
    processor = MeasurementProcessor(measurement_config)
    noise_estimator = NoiseEstimator(noise_config)
    outlier_detector = OutlierDetector(measurement_config)
    innovation_monitor = InnovationMonitor()
    
    return BatchUpdater(processor, noise_estimator, outlier_detector, innovation_monitor)


def process_gameweek_measurements(
    player_scores: List[PlayerScore],
    kalman_model: KalmanPlayerModel,
    gameweek: int,
    season: str,
    batch_updater: Optional[BatchUpdater] = None,
    session: Optional[Session] = None
) -> Dict[int, Any]:
    """
    Convenience function to process a full gameweek of measurements.
    
    Args:
        player_scores: List of PlayerScore objects for the gameweek
        kalman_model: KalmanPlayerModel to update
        gameweek: Gameweek number
        season: Season identifier
        batch_updater: Optional pre-configured BatchUpdater
        session: Database session
        
    Returns:
        Dictionary of updated player states
    """
    if batch_updater is None:
        batch_updater = create_measurement_pipeline()
    
    # Process measurements
    player_measurements, position_noise_matrices = batch_updater.process_batch(
        player_scores, gameweek, season, session
    )
    
    # Extract player positions
    player_positions = {}
    for score in player_scores:
        if score.player and score.player.position:
            player_positions[score.player_id] = score.player.position
    
    # Update Kalman filters
    updated_states = batch_updater.update_batch(
        kalman_model, player_measurements, position_noise_matrices,
        player_positions, gameweek, season
    )
    
    return updated_states