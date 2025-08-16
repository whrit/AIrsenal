"""
Base model interfaces for AIrsenal advanced prediction system.

This module defines abstract base classes for different types of models used in the
AIrsenal FPL prediction pipeline. These interfaces provide a consistent API for
model development while ensuring compatibility with the existing JAX/NumPyro framework.

All models follow SOLID principles and provide extensible interfaces for:
- Adaptive learning systems
- Player availability prediction
- Form calculation and trend analysis
- Feature engineering pipelines

Classes:
    AdaptivePlayerModel: Interface for models that adapt to recent performance
    AvailabilityPredictor: Interface for predicting player availability
    FormCalculator: Interface for calculating player form metrics
    FeatureEngineer: Interface for feature engineering pipelines
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

# Type aliases for better readability
PlayerData = dict[str, Any]
FeatureMatrix = Union[np.ndarray, jnp.ndarray, pd.DataFrame]
PredictionOutput = dict[str, float | np.ndarray | jnp.ndarray]
AvailabilityRisk = dict[str, float]
FormMetrics = dict[str, float | dict[str, float]]


class ModelState(Protocol):
    """Protocol for model state objects used in adaptive models."""

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> ModelState:
        """Create state from dictionary representation."""
        ...


class AdaptivePlayerModel(ABC):
    """
    Abstract base class for player models that adapt to recent performance.
    
    This interface defines the contract for models that can:
    - Learn incrementally from new data
    - Adapt to changing player performance patterns
    - Maintain numerical stability during online updates
    - Integrate with the existing JAX/NumPyro framework
    
    Example usage:
        ```python
        model = ConcreteAdaptiveModel()
        model.fit(training_data)
        
        # Online learning with new gameweek data
        new_data = get_latest_gameweek_data()
        model.update_with_recent_data(new_data)
        
        # Generate predictions
        predictions = model.predict(player_ids, gameweeks_ahead=3)
        ```
    """

    def __init__(self, learning_rate: float = 0.01, decay_factor: float = 0.95):
        """
        Initialize adaptive model.
        
        Args:
            learning_rate: Rate of adaptation to new data (0 < learning_rate <= 1)
            decay_factor: Exponential decay for historical data importance (0 < decay_factor < 1)
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.is_fitted = False
        self.feature_importance_: dict[str, float] | None = None
        self.last_update_gameweek: int | None = None

    @abstractmethod
    def fit(
        self,
        data: PlayerData,
        season: str,
        max_gameweek: int,
        dbsession: Session | None = None,
        **kwargs
    ) -> AdaptivePlayerModel:
        """
        Fit the model to historical data.
        
        Args:
            data: Dictionary containing player data with keys:
                - 'player_ids': array of player IDs
                - 'features': feature matrix (n_players, n_gameweeks, n_features)  
                - 'targets': target values (points, minutes, etc.)
                - 'gameweeks': array of gameweek numbers
            season: Season identifier (e.g., "2023")
            max_gameweek: Maximum gameweek number to use for training
            dbsession: Database session for accessing additional data
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If data format is invalid
            RuntimeError: If fitting fails due to numerical issues
        """
        ...

    @abstractmethod
    def predict(
        self,
        player_ids: list[int] | np.ndarray,
        gameweeks_ahead: int = 3,
        features: FeatureMatrix | None = None,
        **kwargs
    ) -> PredictionOutput:
        """
        Generate predictions for specified players and gameweeks.
        
        Args:
            player_ids: Array or list of player IDs to predict for
            gameweeks_ahead: Number of gameweeks to predict ahead
            features: Optional feature matrix for prediction
                     If None, will generate features internally
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with prediction results:
                - 'player_ids': array of player IDs
                - 'predictions': prediction matrix (n_players, n_gameweeks)
                - 'uncertainty': uncertainty estimates (optional)
                - 'metadata': additional prediction metadata
                
        Raises:
            RuntimeError: If model not fitted or prediction fails
        """
        ...

    @abstractmethod
    def update_with_recent_data(
        self,
        recent_data: PlayerData,
        gameweek: int,
        season: str,
        **kwargs
    ) -> AdaptivePlayerModel:
        """
        Update model parameters with recent performance data.
        
        This method implements incremental learning, allowing the model to
        adapt to recent performance changes without full retraining.
        
        Args:
            recent_data: Dictionary containing recent player data
            gameweek: Current gameweek number
            season: Current season identifier
            **kwargs: Additional update parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If model not fitted or update fails
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores (0-1 scale)
            
        Raises:
            RuntimeError: If model not fitted
        """
        ...

    def get_model_state(self) -> ModelState:
        """
        Get current model state for serialization.
        
        Returns:
            Model state object containing all necessary information
            to restore the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting state")

        # Default implementation - subclasses should override
        return {
            'learning_rate': self.learning_rate,
            'decay_factor': self.decay_factor,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance_,
            'last_update_gameweek': self.last_update_gameweek
        }

    def load_model_state(self, state: ModelState) -> AdaptivePlayerModel:
        """
        Load model state from serialized representation.
        
        Args:
            state: Model state object
            
        Returns:
            Self for method chaining
        """
        # Default implementation - subclasses should override
        if hasattr(state, 'to_dict'):
            state_dict = state.to_dict()
        else:
            state_dict = state

        self.learning_rate = state_dict.get('learning_rate', self.learning_rate)
        self.decay_factor = state_dict.get('decay_factor', self.decay_factor)
        self.is_fitted = state_dict.get('is_fitted', False)
        self.feature_importance_ = state_dict.get('feature_importance')
        self.last_update_gameweek = state_dict.get('last_update_gameweek')

        return self


class AvailabilityPredictor(ABC):
    """
    Abstract base class for predicting player availability.
    
    This interface defines the contract for models that predict:
    - Injury probability and duration
    - Suspension likelihood  
    - Rotation risk based on team strategy
    - General unavailability factors
    
    Example usage:
        ```python
        predictor = ConcreteAvailabilityPredictor()
        predictor.fit(player_data, injury_data, suspension_data)
        
        # Predict availability for next 3 gameweeks
        availability = predictor.predict_availability(
            player_ids=[123, 456], 
            gameweeks_ahead=3
        )
        
        # Get risk factor breakdown
        risks = predictor.get_risk_factors(player_id=123)
        ```
    """

    def __init__(self, risk_threshold: float = 0.3):
        """
        Initialize availability predictor.
        
        Args:
            risk_threshold: Threshold above which a player is considered high risk
        """
        self.risk_threshold = risk_threshold
        self.is_fitted = False
        self.supported_risk_types = ['injury', 'suspension', 'rotation', 'other']

    @abstractmethod
    def predict_availability(
        self,
        player_ids: list[int] | np.ndarray,
        gameweeks_ahead: int = 3,
        season: str = None,
        **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Predict player availability probabilities.
        
        Args:
            player_ids: Array or list of player IDs
            gameweeks_ahead: Number of gameweeks to predict ahead
            season: Season identifier
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary containing:
                - 'player_ids': array of player IDs
                - 'availability_prob': matrix (n_players, n_gameweeks) 
                  with availability probabilities
                - 'risk_breakdown': dictionary with risk type probabilities
                - 'confidence': confidence scores for predictions
                
        Raises:
            RuntimeError: If model not fitted
            ValueError: If input parameters are invalid
        """
        ...

    @abstractmethod
    def get_risk_factors(
        self,
        player_id: int,
        gameweek: int = None,
        **kwargs
    ) -> AvailabilityRisk:
        """
        Get detailed risk factor breakdown for a specific player.
        
        Args:
            player_id: Player ID to analyze
            gameweek: Specific gameweek to analyze (if None, uses next gameweek)
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary with risk factors:
                - 'injury_risk': probability of injury (0-1)
                - 'suspension_risk': probability of suspension (0-1) 
                - 'rotation_risk': probability of rotation (0-1)
                - 'other_risk': other unavailability factors (0-1)
                - 'overall_risk': combined risk score (0-1)
                - 'risk_description': human-readable description
                
        Raises:
            RuntimeError: If model not fitted
            ValueError: If player_id not found
        """
        ...

    @abstractmethod
    def update_injury_data(
        self,
        injury_updates: dict[str, Any],
        gameweek: int,
        season: str,
        **kwargs
    ) -> AvailabilityPredictor:
        """
        Update model with latest injury/unavailability data.
        
        Args:
            injury_updates: Dictionary containing injury updates:
                - 'player_ids': array of affected player IDs
                - 'injury_types': array of injury types
                - 'expected_return': array of expected return gameweeks
                - 'severity': array of injury severity scores
            gameweek: Current gameweek
            season: Current season
            **kwargs: Additional update parameters
            
        Returns:
            Self for method chaining
        """
        ...

    def fit(
        self,
        player_data: PlayerData,
        injury_history: dict[str, Any],
        suspension_history: dict[str, Any],
        season: str,
        dbsession: Session | None = None,
        **kwargs
    ) -> AvailabilityPredictor:
        """
        Fit the availability prediction model.
        
        Args:
            player_data: Player performance and attribute data
            injury_history: Historical injury data
            suspension_history: Historical suspension data  
            season: Season identifier
            dbsession: Database session
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        # Default implementation - subclasses should override
        self.is_fitted = True
        return self

    def get_high_risk_players(
        self,
        gameweek: int,
        season: str,
        threshold: float | None = None
    ) -> list[tuple[int, float]]:
        """
        Get list of players with high unavailability risk.
        
        Args:
            gameweek: Gameweek to analyze
            season: Season identifier
            threshold: Risk threshold (if None, uses instance threshold)
            
        Returns:
            List of tuples (player_id, risk_score) for high-risk players
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting high-risk players")

        threshold = threshold or self.risk_threshold
        # Default implementation - subclasses should override with actual logic
        return []


class FormCalculator(ABC):
    """
    Abstract base class for calculating player form metrics.
    
    This interface defines the contract for models that calculate:
    - Short-term and long-term form
    - Performance momentum and trends
    - Form change detection
    - Context-aware form metrics
    
    Example usage:
        ```python
        calculator = ConcreteFormCalculator()
        calculator.fit(player_data, season="2023")
        
        # Calculate current form
        form = calculator.calculate_form(
            player_id=123,
            time_window=5,
            gameweek=15
        )
        
        # Detect trend changes
        trend_change = calculator.detect_trend_change(
            player_id=123,
            lookback_window=10
        )
        ```
    """

    def __init__(self, default_windows: list[int] = [3, 5, 10]):
        """
        Initialize form calculator.
        
        Args:
            default_windows: Default time windows for form calculation
        """
        self.default_windows = default_windows
        self.is_fitted = False
        self.form_metrics = ['points_form', 'goals_form', 'assists_form',
                           'minutes_form', 'bonus_form']

    @abstractmethod
    def calculate_form(
        self,
        player_id: int,
        time_window: int = 5,
        gameweek: int = None,
        season: str = None,
        metrics: list[str] | None = None,
        **kwargs
    ) -> FormMetrics:
        """
        Calculate form metrics for a specific player.
        
        Args:
            player_id: Player ID to calculate form for
            time_window: Number of recent games to consider
            gameweek: Current gameweek (if None, uses latest)
            season: Season identifier
            metrics: List of metrics to calculate (if None, uses all)
            **kwargs: Additional calculation parameters
            
        Returns:
            Dictionary containing form metrics:
                - 'form_score': overall form score (0-10 scale)
                - 'points_per_game': average points in time window
                - 'trend': trend direction ('improving', 'stable', 'declining')
                - 'consistency': consistency score (0-1)
                - 'momentum': momentum indicator (-1 to 1)
                - 'detailed_metrics': breakdown by specific metrics
                
        Raises:
            RuntimeError: If model not fitted
            ValueError: If player_id not found or invalid parameters
        """
        ...

    @abstractmethod
    def get_momentum(
        self,
        player_id: int,
        lookback_window: int = 10,
        season: str = None,
        **kwargs
    ) -> dict[str, float]:
        """
        Calculate performance momentum indicators.
        
        Args:
            player_id: Player ID to analyze
            lookback_window: Number of games to analyze for momentum
            season: Season identifier
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing momentum metrics:
                - 'momentum_score': overall momentum (-1 to 1)
                - 'acceleration': rate of form change
                - 'volatility': form consistency measure
                - 'peak_distance': games since peak performance
                - 'trough_distance': games since worst performance
        """
        ...

    @abstractmethod
    def detect_trend_change(
        self,
        player_id: int,
        lookback_window: int = 10,
        sensitivity: float = 0.1,
        **kwargs
    ) -> dict[str, Any]:
        """
        Detect significant changes in player form trends.
        
        Args:
            player_id: Player ID to analyze
            lookback_window: Number of games to analyze
            sensitivity: Sensitivity threshold for detecting changes
            **kwargs: Additional detection parameters
            
        Returns:
            Dictionary containing trend change information:
                - 'change_detected': boolean indicating if change detected
                - 'change_type': 'improvement' or 'decline'
                - 'change_magnitude': magnitude of change (0-1)
                - 'change_gameweek': gameweek when change occurred
                - 'confidence': confidence in detection (0-1)
                - 'description': human-readable description
        """
        ...

    def fit(
        self,
        player_data: PlayerData,
        season: str,
        dbsession: Session | None = None,
        **kwargs
    ) -> FormCalculator:
        """
        Fit the form calculation model.
        
        Args:
            player_data: Historical player performance data
            season: Season identifier
            dbsession: Database session
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        # Default implementation - subclasses should override
        self.is_fitted = True
        return self

    def get_form_distribution(
        self,
        position: str = None,
        season: str = None,
        time_window: int = 5
    ) -> dict[str, np.ndarray]:
        """
        Get form distribution statistics across players.
        
        Args:
            position: Player position to filter by (optional)
            season: Season identifier
            time_window: Time window for form calculation
            
        Returns:
            Dictionary with distribution statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting form distribution")

        # Default implementation - subclasses should override
        return {
            'mean_form': 5.0,
            'std_form': 1.5,
            'percentiles': np.array([0, 25, 50, 75, 100])
        }


class FeatureEngineer(ABC):
    """
    Abstract base class for feature engineering pipelines.
    
    This interface defines the contract for feature engineering systems that:
    - Transform raw player/team data into ML-ready features
    - Handle missing data and outliers
    - Create derived features and interactions
    - Maintain feature consistency between training and inference
    
    Example usage:
        ```python
        engineer = ConcreteFeatureEngineer()
        engineer.fit(training_data, target_variable='points')
        
        # Transform training data
        train_features = engineer.transform(training_data)
        
        # Transform new data for prediction
        pred_features = engineer.transform(new_data)
        
        # Get feature names and importance
        feature_names = engineer.get_feature_names()
        ```
    """

    def __init__(self, handle_missing: str = 'impute', scale_features: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            handle_missing: Strategy for missing data ('impute', 'drop', 'flag')
            scale_features: Whether to scale numerical features
        """
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.is_fitted = False
        self.feature_names_: list[str] | None = None
        self.feature_metadata_: dict[str, Any] | None = None

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame | dict[str, Any],
        target_variable: str | None = None,
        **kwargs
    ) -> FeatureEngineer:
        """
        Fit the feature engineering pipeline.
        
        Args:
            data: Raw data to learn feature transformations from
            target_variable: Target variable name for supervised feature selection
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        ...

    @abstractmethod
    def transform(
        self,
        data: pd.DataFrame | dict[str, Any],
        **kwargs
    ) -> FeatureMatrix:
        """
        Transform data using fitted feature engineering pipeline.
        
        Args:
            data: Raw data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed feature matrix
            
        Raises:
            RuntimeError: If pipeline not fitted
            ValueError: If data format incompatible
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Get names of engineered features.
        
        Returns:
            List of feature names in order
            
        Raises:
            RuntimeError: If pipeline not fitted
        """
        ...

    def fit_transform(
        self,
        data: pd.DataFrame | dict[str, Any],
        target_variable: str | None = None,
        **kwargs
    ) -> FeatureMatrix:
        """
        Fit pipeline and transform data in one step.
        
        Args:
            data: Raw data to fit and transform
            target_variable: Target variable for supervised methods
            **kwargs: Additional parameters
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(data, target_variable, **kwargs).transform(data, **kwargs)

    def get_feature_metadata(self) -> dict[str, Any]:
        """
        Get metadata about engineered features.
        
        Returns:
            Dictionary with feature metadata including:
                - 'feature_types': types of each feature
                - 'missing_rates': missing data rates
                - 'importance_scores': feature importance (if available)
                - 'correlation_matrix': feature correlations
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before getting metadata")

        return self.feature_metadata_ or {}

    def validate_data_compatibility(
        self,
        data: pd.DataFrame | dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate that data is compatible with fitted pipeline.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        if not self.is_fitted:
            return False, ["Pipeline not fitted"]

        # Default implementation - subclasses should override
        return True, []


# Utility functions for model validation and testing

def validate_model_interface(model: Any, interface_class: type) -> tuple[bool, list[str]]:
    """
    Validate that a model correctly implements a base interface.
    
    Args:
        model: Model instance to validate
        interface_class: Base interface class to validate against
        
    Returns:
        Tuple of (is_valid, list_of_missing_methods)
    """
    if not isinstance(model, interface_class):
        return False, [f"Model is not instance of {interface_class.__name__}"]

    missing_methods = []
    for method_name in interface_class.__abstractmethods__:
        if not hasattr(model, method_name):
            missing_methods.append(method_name)
        elif not callable(getattr(model, method_name)):
            missing_methods.append(f"{method_name} (not callable)")

    return len(missing_methods) == 0, missing_methods


def create_mock_player_data(
    n_players: int = 10,
    n_gameweeks: int = 15,
    n_features: int = 5,
    include_targets: bool = True
) -> PlayerData:
    """
    Create mock player data for testing model interfaces.
    
    Args:
        n_players: Number of players to generate
        n_gameweeks: Number of gameweeks
        n_features: Number of features per player
        include_targets: Whether to include target variables
        
    Returns:
        Dictionary with mock player data
    """
    np.random.seed(42)  # For reproducible testing

    data = {
        'player_ids': np.arange(1, n_players + 1),
        'features': np.random.randn(n_players, n_gameweeks, n_features),
        'gameweeks': np.arange(1, n_gameweeks + 1),
        'season': '2023'
    }

    if include_targets:
        data['targets'] = np.random.poisson(3, (n_players, n_gameweeks))
        data['minutes'] = np.random.randint(0, 91, (n_players, n_gameweeks))

    return data
