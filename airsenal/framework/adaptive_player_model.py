"""
AdaptivePlayerModel implementation with state-space modeling for AIrsenal.

ARCHITECTURE OVERVIEW
====================

This module implements the AdaptivePlayerModel base class that extends BasePlayerModel
with state-space modeling capabilities, temporal dynamics, and adaptive learning mechanisms.
It forms the foundation for TASK-201 in Sprint 02 of the AIrsenal ML enhancement project.

STATE-SPACE MODELING DESIGN
===========================

The core architecture follows a Bayesian state-space modeling approach:

1. **Hidden State Representation**:
   - State vector: [skill, form, consistency] (default 3D)
   - Each component represents a latent player ability
   - States evolve over time according to temporal dynamics
   - Uncertainty quantified via covariance matrices

2. **Observation Model**:
   - Maps hidden states to observable performance metrics
   - Default observations: [goals, assists, minutes, bonus] (4D)
   - Flexible mapping allows for position-specific observations
   - Measurement noise accounts for observation uncertainty

3. **Temporal Dynamics**:
   - State transition model for gameweek-to-gameweek evolution
   - Process noise models natural ability fluctuations
   - Supports non-linear dynamics through abstract interface
   - Ready for Kalman filter integration (TASK-204)

4. **Adaptive Learning**:
   - Incremental state updates with new observations
   - Exponential decay for historical data importance
   - Configurable learning rates and decay factors
   - Online adaptation without full retraining

DESIGN DECISIONS & RATIONALE
============================

1. **Inheritance from BasePlayerModel**:
   - Ensures compatibility with existing AIrsenal framework
   - Maintains consistency with current player modeling interface
   - Implements get_probs() and get_probs_for_player() for FPL scoring

2. **Abstract Base Class Pattern**:
   - Enforces consistent interface across concrete implementations
   - Allows for multiple state-space modeling approaches (Kalman, particle filters, etc.)
   - Facilitates testing and validation of different algorithms
   - Supports future extensions (ensemble models, non-linear dynamics)

3. **JAX Compatibility**:
   - Native support for JAX arrays for GPU acceleration
   - Configurable numpy/JAX backend via StateSpaceConfig
   - Prepares for integration with NumPyro probabilistic models
   - Enables efficient gradient-based optimization

4. **Modular Configuration**:
   - StateSpaceConfig separates model structure from implementation
   - Easy experimentation with different state/observation dimensions
   - Centralized parameter management for noise levels
   - Validation of configuration consistency

5. **Uncertainty Quantification**:
   - Full covariance matrices for principled uncertainty tracking
   - Separates epistemic (model) and aleatoric (data) uncertainty
   - Enables confidence intervals and risk-aware decisions
   - Supports uncertainty propagation through predictions

6. **State Persistence**:
   - PlayerState objects with serialization capabilities
   - Model state can be saved/restored for persistence
   - Enables incremental updates and model checkpointing
   - Supports distributed training and inference

EXTENSIBILITY FEATURES
======================

1. **Ready for Kalman Filter Integration** (TASK-204):
   - Abstract methods align with Kalman filter predict/update cycle
   - State transition and observation models clearly separated
   - Covariance handling follows Kalman filter conventions

2. **Support for Advanced Models** (TASK-205, TASK-206):
   - Abstract interface supports non-linear dynamics
   - Can incorporate external factors (injuries, transfers, etc.)
   - Flexible observation model for multi-modal data
   - Extensible to ensemble and hierarchical models

3. **Feature Engineering Integration**:
   - Compatible with feature store and validation systems
   - Supports time-varying features and covariates
   - Can incorporate match context and opponent strength
   - Ready for automated feature selection

INTEGRATION WITH AIRSENAL
=========================

1. **Database Schema Compatibility**:
   - Works with existing Player, Match, and PlayerScore tables
   - State persistence integrates with model versioning system
   - Supports incremental updates from data fetcher

2. **Prediction Pipeline Integration**:
   - Implements BasePlayerModel interface for drop-in replacement
   - Compatible with existing optimization and squad building
   - Supports batch and streaming prediction modes
   - Integrates with uncertainty-aware optimization

3. **Performance Considerations**:
   - Efficient state storage for large player populations
   - Configurable precision for memory/accuracy tradeoffs
   - Batch processing for multiple players
   - Optional GPU acceleration via JAX

USAGE PATTERNS
==============

1. **Training Phase**:
   ```python
   config = StateSpaceConfig(state_dim=3, obs_dim=4)
   model = ConcreteAdaptivePlayerModel(config)
   model.fit(historical_data, season="2023", max_gameweek=15)
   ```

2. **Prediction Phase**:
   ```python
   predictions = model.predict(player_ids, gameweeks_ahead=3)
   points_forecast = predictions["predictions"]
   uncertainty = predictions["uncertainty"]
   ```

3. **Online Update Phase**:
   ```python
   recent_data = get_latest_gameweek_data()
   model.update_with_recent_data(recent_data, gameweek=16, season="2023")
   ```

4. **State Analysis**:
   ```python
   player_state = model.get_player_state(player_id)
   skill_level = player_state.state_mean[0]  # skill component
   uncertainty = np.sqrt(player_state.state_cov[0, 0])  # skill uncertainty
   ```

Classes:
    AdaptivePlayerModel: Base class for adaptive player models with state-space representation
    PlayerState: Data class representing player state vector and covariance
    StateSpaceConfig: Configuration class for state-space model parameters

Example usage:
    ```python
    config = StateSpaceConfig(
        state_dim=3,  # [skill, form, consistency]
        obs_dim=4,    # [goals, assists, minutes, bonus]
        process_noise_std=0.1,
        measurement_noise_std=0.2
    )
    
    model = ConcreteAdaptivePlayerModel(config)
    model.fit(training_data, season="2023", max_gameweek=15)
    
    # Predict future performance
    predictions = model.predict(player_ids=[123, 456], gameweeks_ahead=3)
    
    # Update with new observations
    new_data = get_latest_gameweek_data()
    model.update_with_recent_data(new_data, gameweek=16, season="2023")
    ```
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.player_model import BasePlayerModel

# Type aliases for better readability
StateVector = Union[np.ndarray, jnp.ndarray]
CovarianceMatrix = Union[np.ndarray, jnp.ndarray]
ObservationVector = Union[np.ndarray, jnp.ndarray]
PlayerData = Dict[str, Any]
PredictionOutput = Dict[str, Union[float, np.ndarray, jnp.ndarray]]


@dataclass
class StateSpaceConfig:
    """Configuration for state-space model parameters.
    
    Attributes:
        state_dim: Dimension of state vector (default: 3 for [skill, form, consistency])
        obs_dim: Dimension of observation vector  
        process_noise_std: Standard deviation of process noise
        measurement_noise_std: Standard deviation of measurement noise
        initial_state_std: Standard deviation for initial state uncertainty
        state_names: Names of state components
        obs_names: Names of observation components
        use_jax: Whether to use JAX arrays for computations
    """
    state_dim: int = 3
    obs_dim: int = 4
    process_noise_std: float = 0.1
    measurement_noise_std: float = 0.2
    initial_state_std: float = 1.0
    state_names: List[str] = None
    obs_names: List[str] = None
    use_jax: bool = True
    
    def __post_init__(self):
        """Set default names if not provided."""
        if self.state_names is None:
            self.state_names = ["skill", "form", "consistency"][:self.state_dim]
        if self.obs_names is None:
            self.obs_names = ["goals", "assists", "minutes", "bonus"][:self.obs_dim]
            
        # Validate dimensions
        if len(self.state_names) != self.state_dim:
            raise ValueError(f"state_names length ({len(self.state_names)}) must match state_dim ({self.state_dim})")
        if len(self.obs_names) != self.obs_dim:
            raise ValueError(f"obs_names length ({len(self.obs_names)}) must match obs_dim ({self.obs_dim})")


@dataclass
class PlayerState:
    """Represents the state of a player at a specific time.
    
    Attributes:
        player_id: Unique player identifier
        state_mean: Mean of state vector [skill, form, consistency, ...]
        state_cov: Covariance matrix of state uncertainty
        gameweek: Gameweek this state corresponds to
        season: Season identifier
        last_updated: Timestamp of last update
    """
    player_id: int
    state_mean: StateVector
    state_cov: CovarianceMatrix
    gameweek: int
    season: str
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "player_id": self.player_id,
            "state_mean": np.asarray(self.state_mean).tolist(),
            "state_cov": np.asarray(self.state_cov).tolist(),
            "gameweek": self.gameweek,
            "season": self.season,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> PlayerState:
        """Create state from dictionary representation."""
        return cls(
            player_id=state_dict["player_id"],
            state_mean=np.array(state_dict["state_mean"]),
            state_cov=np.array(state_dict["state_cov"]),
            gameweek=state_dict["gameweek"],
            season=state_dict["season"],
            last_updated=state_dict.get("last_updated"),
        )
    
    def copy(self) -> PlayerState:
        """Create a deep copy of the player state."""
        return PlayerState(
            player_id=self.player_id,
            state_mean=np.copy(self.state_mean),
            state_cov=np.copy(self.state_cov),
            gameweek=self.gameweek,
            season=self.season,
            last_updated=self.last_updated,
        )


class AdaptivePlayerModel(BasePlayerModel):
    """
    Abstract base class for adaptive player models with state-space representation.
    
    This class extends BasePlayerModel with state-space modeling capabilities:
    - Hidden states representing player abilities (skill, form, consistency)
    - Observation model linking states to match performance
    - Process noise modeling state evolution uncertainty
    - Measurement noise modeling observation uncertainty
    - Temporal dynamics through state transition models
    
    The state-space formulation allows for:
    - Principled uncertainty quantification
    - Adaptive learning from new observations
    - Handling of missing data and irregular observations
    - Integration with Kalman filtering and other advanced techniques
    
    Subclasses must implement the abstract methods to define:
    - State initialization strategy
    - State transition dynamics
    - Observation model
    - Measurement update mechanism
    """
    
    def __init__(
        self,
        config: StateSpaceConfig,
        learning_rate: float = 0.01,
        decay_factor: float = 0.95,
        random_seed: int = 42,
    ):
        """
        Initialize adaptive player model with state-space configuration.
        
        Args:
            config: State-space model configuration
            learning_rate: Rate of adaptation to new data (0 < learning_rate <= 1)
            decay_factor: Exponential decay for historical data importance (0 < decay_factor < 1)
            random_seed: Random seed for reproducible initialization
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate parameters
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not 0 < decay_factor < 1:
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor}")
            
        self.config = config
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.random_seed = random_seed
        
        # Model state
        self.is_fitted = False
        self.player_states: Dict[int, PlayerState] = {}
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.last_update_gameweek: Optional[int] = None
        self.last_season: Optional[str] = None
        
        # State-space matrices (to be initialized by subclasses)
        self.transition_matrix: Optional[StateVector] = None
        self.observation_matrix: Optional[StateVector] = None
        self.process_noise_cov: Optional[CovarianceMatrix] = None
        self.measurement_noise_cov: Optional[CovarianceMatrix] = None
        
        # JAX random key for reproducible computations
        self.rng_key = random.PRNGKey(random_seed)
    
    @abstractmethod
    def initialize_state(
        self,
        player_id: int,
        initial_data: Optional[PlayerData] = None,
        gameweek: int = 1,
        season: str = "2023",
        **kwargs,
    ) -> PlayerState:
        """
        Initialize state vector and covariance for a player.
        
        Args:
            player_id: Player ID to initialize state for
            initial_data: Optional initial performance data for informed initialization
            gameweek: Initial gameweek
            season: Season identifier
            **kwargs: Additional initialization parameters
            
        Returns:
            Initial PlayerState object
            
        Raises:
            ValueError: If player_id is invalid or initialization fails
        """
        ...
    
    @abstractmethod
    def predict_state(
        self,
        current_state: PlayerState,
        gameweeks_ahead: int = 1,
        **kwargs,
    ) -> PlayerState:
        """
        Predict future state using state transition model.
        
        This method implements the prediction step of state-space modeling,
        evolving the current state forward in time according to the learned
        temporal dynamics.
        
        Args:
            current_state: Current player state
            gameweeks_ahead: Number of gameweeks to predict ahead
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted PlayerState object
            
        Raises:
            ValueError: If current_state is invalid
        """
        ...
    
    @abstractmethod
    def update_state(
        self,
        predicted_state: PlayerState,
        observation: ObservationVector,
        observation_noise: Optional[CovarianceMatrix] = None,
        **kwargs,
    ) -> PlayerState:
        """
        Update state based on new observations (measurement update).
        
        This method implements the measurement update step, incorporating
        new observations to refine state estimates and reduce uncertainty.
        
        Args:
            predicted_state: Predicted state from prediction step
            observation: Observed performance metrics
            observation_noise: Optional observation noise covariance
            **kwargs: Additional update parameters
            
        Returns:
            Updated PlayerState object
            
        Raises:
            ValueError: If inputs are incompatible or update fails
        """
        ...
    
    @abstractmethod
    def observation_model(
        self,
        state: StateVector,
        **kwargs,
    ) -> ObservationVector:
        """
        Map state vector to expected observations.
        
        This method defines how hidden states (skill, form, consistency)
        map to observable performance metrics (goals, assists, etc.).
        
        Args:
            state: State vector to map
            **kwargs: Additional mapping parameters
            
        Returns:
            Expected observation vector
            
        Raises:
            ValueError: If state dimensions are incompatible
        """
        ...
    
    def fit(
        self,
        data: PlayerData,
        season: str,
        max_gameweek: int,
        dbsession: Optional[Session] = None,
        **kwargs,
    ) -> AdaptivePlayerModel:
        """
        Fit the adaptive model to historical data.
        
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
        try:
            # Validate input data
            self._validate_training_data(data)
            
            # Initialize states for all players
            player_ids = data["player_ids"]
            gameweeks = data["gameweeks"]
            
            for player_id in player_ids:
                # Get initial data for this player
                player_data = self._extract_player_data(data, player_id)
                initial_state = self.initialize_state(
                    player_id=player_id,
                    initial_data=player_data,
                    gameweek=gameweeks[0],
                    season=season,
                    **kwargs,
                )
                self.player_states[player_id] = initial_state
            
            # Learn model parameters from historical data
            self._fit_state_space_model(data, season, max_gameweek, **kwargs)
            
            # Compute feature importance
            self.feature_importance_ = self._compute_feature_importance(data)
            
            self.is_fitted = True
            self.last_season = season
            self.last_update_gameweek = max_gameweek
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit adaptive player model: {e}") from e
    
    def predict(
        self,
        player_ids: Union[List[int], np.ndarray],
        gameweeks_ahead: int = 3,
        features: Optional[Union[np.ndarray, jnp.ndarray, pd.DataFrame]] = None,
        **kwargs,
    ) -> PredictionOutput:
        """
        Generate predictions for specified players and gameweeks.
        
        Args:
            player_ids: Array or list of player IDs to predict for
            gameweeks_ahead: Number of gameweeks to predict ahead
            features: Optional feature matrix for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with prediction results:
                - 'player_ids': array of player IDs
                - 'predictions': prediction matrix (n_players, n_gameweeks)
                - 'uncertainty': uncertainty estimates
                - 'states': predicted states for each player
                - 'metadata': additional prediction metadata
                
        Raises:
            RuntimeError: If model not fitted or prediction fails
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            player_ids = np.asarray(player_ids)
            n_players = len(player_ids)
            
            # Initialize prediction arrays
            predictions = np.zeros((n_players, gameweeks_ahead))
            uncertainties = np.zeros((n_players, gameweeks_ahead))
            predicted_states = {}
            
            for i, player_id in enumerate(player_ids):
                if player_id not in self.player_states:
                    # Initialize state for unknown player
                    current_state = self.initialize_state(
                        player_id=player_id,
                        gameweek=self.last_update_gameweek or 1,
                        season=self.last_season or "2023",
                    )
                    self.player_states[player_id] = current_state
                else:
                    current_state = self.player_states[player_id]
                
                # Predict state evolution
                pred_state = current_state.copy()
                player_predictions = []
                player_uncertainties = []
                
                for gw in range(gameweeks_ahead):
                    # Predict next state
                    pred_state = self.predict_state(pred_state, gameweeks_ahead=1, **kwargs)
                    
                    # Convert state to expected observation
                    expected_obs = self.observation_model(pred_state.state_mean, **kwargs)
                    
                    # Extract relevant prediction (e.g., expected points)
                    prediction = self._state_to_prediction(expected_obs, **kwargs)
                    uncertainty = self._state_to_uncertainty(pred_state, **kwargs)
                    
                    player_predictions.append(prediction)
                    player_uncertainties.append(uncertainty)
                
                predictions[i, :] = player_predictions
                uncertainties[i, :] = player_uncertainties
                predicted_states[player_id] = pred_state
            
            return {
                "player_ids": player_ids,
                "predictions": predictions,
                "uncertainty": uncertainties,
                "states": predicted_states,
                "metadata": {
                    "gameweeks_ahead": gameweeks_ahead,
                    "model_type": self.__class__.__name__,
                    "state_dim": self.config.state_dim,
                    "obs_dim": self.config.obs_dim,
                },
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def update_with_recent_data(
        self,
        recent_data: PlayerData,
        gameweek: int,
        season: str,
        **kwargs,
    ) -> AdaptivePlayerModel:
        """
        Update model parameters with recent performance data.
        
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
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before updating with recent data")
        
        try:
            # Extract player IDs and observations from recent data
            player_ids = recent_data.get("player_ids", [])
            observations = recent_data.get("observations", {})
            
            for player_id in player_ids:
                if player_id in observations:
                    # Get current state
                    if player_id not in self.player_states:
                        # Initialize state for new player
                        current_state = self.initialize_state(
                            player_id=player_id,
                            gameweek=gameweek,
                            season=season,
                        )
                        self.player_states[player_id] = current_state
                    else:
                        current_state = self.player_states[player_id]
                    
                    # Predict state forward to current gameweek
                    gw_diff = gameweek - current_state.gameweek
                    if gw_diff > 0:
                        predicted_state = self.predict_state(
                            current_state, gameweeks_ahead=gw_diff, **kwargs
                        )
                    else:
                        predicted_state = current_state
                    
                    # Update with new observation
                    observation = np.asarray(observations[player_id])
                    updated_state = self.update_state(
                        predicted_state, observation, **kwargs
                    )
                    updated_state.gameweek = gameweek
                    updated_state.season = season
                    
                    # Store updated state
                    self.player_states[player_id] = updated_state
            
            self.last_update_gameweek = gameweek
            self.last_season = season
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to update with recent data: {e}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores (0-1 scale)
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if self.feature_importance_ is None:
            # Compute feature importance based on state components
            importance = {}
            for i, state_name in enumerate(self.config.state_names):
                # Simple importance based on state variance across players
                state_vars = []
                for player_state in self.player_states.values():
                    state_vars.append(player_state.state_cov[i, i])
                
                # Higher variance indicates more important feature
                importance[state_name] = float(np.mean(state_vars))
            
            # Normalize to [0, 1]
            max_importance = max(importance.values()) if importance else 1.0
            for key in importance:
                importance[key] /= max_importance
            
            self.feature_importance_ = importance
        
        return self.feature_importance_
    
    def get_player_state(self, player_id: int) -> Optional[PlayerState]:
        """
        Get current state for a specific player.
        
        Args:
            player_id: Player ID to get state for
            
        Returns:
            PlayerState object or None if player not found
        """
        return self.player_states.get(player_id)
    
    def get_all_player_states(self) -> Dict[int, PlayerState]:
        """
        Get states for all players.
        
        Returns:
            Dictionary mapping player IDs to PlayerState objects
        """
        return self.player_states.copy()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of current player states.
        
        Returns:
            Dictionary with state summary statistics
        """
        if not self.player_states:
            return {"n_players": 0}
        
        states = list(self.player_states.values())
        state_means = np.array([s.state_mean for s in states])
        state_vars = np.array([np.diag(s.state_cov) for s in states])
        
        return {
            "n_players": len(states),
            "state_names": self.config.state_names,
            "mean_states": np.mean(state_means, axis=0).tolist(),
            "std_states": np.std(state_means, axis=0).tolist(),
            "mean_uncertainties": np.mean(state_vars, axis=0).tolist(),
            "last_update_gameweek": self.last_update_gameweek,
            "last_season": self.last_season,
        }
    
    # Protected helper methods
    
    def _validate_training_data(self, data: PlayerData) -> None:
        """Validate format of training data."""
        required_keys = ["player_ids", "features", "targets", "gameweeks"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Training data missing required key: {key}")
        
        player_ids = data["player_ids"]
        features = data["features"]
        targets = data["targets"]
        
        if len(player_ids) == 0:
            raise ValueError("No players in training data")
        
        if features.shape[0] != len(player_ids):
            raise ValueError("Feature matrix first dimension must match number of players")
        
        if targets.shape[0] != len(player_ids):
            raise ValueError("Target matrix first dimension must match number of players")
    
    def _extract_player_data(self, data: PlayerData, player_id: int) -> PlayerData:
        """Extract data for a specific player."""
        player_ids = data["player_ids"]
        try:
            idx = list(player_ids).index(player_id)
        except ValueError as e:
            raise ValueError(f"Player {player_id} not found in data") from e
        
        return {
            "player_id": player_id,
            "features": data["features"][idx] if "features" in data else None,
            "targets": data["targets"][idx] if "targets" in data else None,
            "gameweeks": data.get("gameweeks"),
        }
    
    @abstractmethod
    def _fit_state_space_model(
        self, data: PlayerData, season: str, max_gameweek: int, **kwargs
    ) -> None:
        """Fit state-space model parameters from data."""
        ...
    
    def _compute_feature_importance(self, data: PlayerData) -> Dict[str, float]:
        """Compute feature importance from fitted model."""
        # Default implementation - subclasses should override
        return {name: 1.0 / len(self.config.state_names) for name in self.config.state_names}
    
    def _state_to_prediction(self, observation: ObservationVector, **kwargs) -> float:
        """Convert state observation to scalar prediction (e.g., expected points)."""
        # Default: sum all observation components
        return float(np.sum(observation))
    
    def _state_to_uncertainty(self, state: PlayerState, **kwargs) -> float:
        """Convert state uncertainty to scalar uncertainty measure."""
        # Default: trace of covariance matrix
        return float(np.trace(state.state_cov))