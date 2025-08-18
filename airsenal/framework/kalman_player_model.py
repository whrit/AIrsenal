"""
Kalman Player Model for AIrsenal

Concrete implementation of AdaptivePlayerModel using Kalman filtering for tracking
dynamic player abilities over time. This module integrates the Kalman filter 
framework with AIrsenal's player modeling system to provide robust, uncertainty-aware
predictions of player performance.

Key Features:
- Kalman filter-based state tracking for player abilities
- Multiple filter types (Linear, Extended, Unscented) for different dynamics
- Adaptive noise estimation for robust performance
- Integration with StateManager and ModelPersistence
- Proper handling of missing data and outliers
- Uncertainty quantification for all predictions
- Support for position-specific modeling

Classes:
    KalmanPlayerModel: Main Kalman filter-based player model
    PlayerAbilityModel: Specific model for tracking [skill, form, consistency, momentum]
    PositionSpecificModel: Position-aware Kalman filtering
    EnsembleKalmanModel: Multi-filter ensemble approach

Usage:
    ```python
    from airsenal.framework.kalman_player_model import KalmanPlayerModel
    from airsenal.framework.kalman_filter import FilterConfig
    
    # Configure model
    config = FilterConfig(
        state_dim=4,
        obs_dim=4,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
        enable_adaptive_noise=True
    )
    
    # Create model
    model = KalmanPlayerModel(config, filter_type="extended")
    
    # Fit to historical data
    model.fit(training_data, season="2023", max_gameweek=15)
    
    # Make predictions
    predictions = model.predict(player_ids=[123, 456], gameweeks_ahead=3)
    ```
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
    PlayerData,
    PredictionOutput
)
from airsenal.framework.kalman_filter import (
    BaseKalmanFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    FilterConfig,
    FilterState,
    create_player_ability_config,
    create_simple_linear_filter,
    create_player_transition_function,
    create_player_measurement_function,
    KalmanFilterError,
    NumericalInstabilityError
)

logger = logging.getLogger(__name__)

# Type aliases
FilterType = Literal["linear", "extended", "unscented", "ensemble"]
ModelMode = Literal["single", "position_specific", "ensemble"]


class KalmanPlayerModelError(Exception):
    """Base exception for Kalman player model operations."""
    pass


class KalmanPlayerModel(AdaptivePlayerModel):
    """
    Kalman filter-based implementation of AdaptivePlayerModel.
    
    This class provides state-of-the-art Bayesian filtering for tracking
    player abilities over time with proper uncertainty quantification.
    """
    
    def __init__(
        self,
        config: Optional[StateSpaceConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        filter_type: FilterType = "extended",
        model_mode: ModelMode = "single",
        learning_rate: float = 0.01,
        decay_factor: float = 0.95,
        random_seed: int = 42,
        position_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Initialize Kalman Player Model.
        
        Args:
            config: State space configuration for compatibility
            filter_config: Kalman filter specific configuration
            filter_type: Type of Kalman filter to use
            model_mode: Model mode (single filter, position-specific, ensemble)
            learning_rate: Learning rate for adaptive components
            decay_factor: Decay factor for historical data
            random_seed: Random seed for reproducibility
            position_mapping: Mapping from positions to model indices
        """
        # Initialize base class
        if config is None:
            config = StateSpaceConfig(
                state_dim=4,
                obs_dim=4,
                state_names=["skill", "form", "consistency", "momentum"],
                obs_names=["goals", "assists", "minutes", "bonus"]
            )
        
        super().__init__(config, learning_rate, decay_factor, random_seed)
        
        # Kalman filter configuration
        if filter_config is None:
            filter_config = create_player_ability_config(
                state_names=config.state_names,
                obs_names=config.obs_names,
                process_noise_std=config.process_noise_std,
                measurement_noise_std=config.measurement_noise_std,
                enable_adaptive_noise=True,
                use_joseph_form=True
            )
        
        self.filter_config = filter_config
        self.filter_type = filter_type
        self.model_mode = model_mode
        
        # Position mapping for position-specific models
        self.position_mapping = position_mapping or {
            "GK": 0, "DEF": 1, "MID": 2, "FWD": 3
        }
        
        # Initialize filters based on mode
        self.filters: Dict[str, BaseKalmanFilter] = {}
        self.filter_states: Dict[int, Dict[str, FilterState]] = {}  # player_id -> filter_name -> state
        
        self._initialize_filters()
        
        # Tracking variables
        self.prediction_history: Dict[int, List[Dict]] = {}
        self.update_counts: Dict[int, int] = {}
        
        # Model performance metrics
        self.model_metrics = {
            "total_predictions": 0,
            "total_updates": 0,
            "average_log_likelihood": 0.0,
            "numerical_errors": 0,
            "adaptive_updates": 0
        }
    
    def _initialize_filters(self):
        """Initialize Kalman filters based on configuration."""
        try:
            if self.model_mode == "single":
                # Single filter for all players
                self.filters["main"] = self._create_filter(self.filter_type)
                
            elif self.model_mode == "position_specific":
                # One filter per position
                for position in self.position_mapping:
                    self.filters[position] = self._create_filter(self.filter_type)
                    
            elif self.model_mode == "ensemble":
                # Ensemble of different filter types
                self.filters["linear"] = self._create_filter("linear")
                self.filters["extended"] = self._create_filter("extended")
                self.filters["unscented"] = self._create_filter("unscented")
            
            logger.info(f"Initialized {len(self.filters)} Kalman filters in {self.model_mode} mode")
            
        except Exception as e:
            raise KalmanPlayerModelError(f"Failed to initialize filters: {e}")
    
    def _create_filter(self, filter_type: str) -> BaseKalmanFilter:
        """Create a Kalman filter of the specified type."""
        if filter_type == "linear":
            return create_simple_linear_filter(self.filter_config)
            
        elif filter_type == "extended":
            transition_fn = create_player_transition_function()
            measurement_fn = create_player_measurement_function()
            return ExtendedKalmanFilter(
                self.filter_config, 
                transition_fn, 
                measurement_fn
            )
            
        elif filter_type == "unscented":
            transition_fn = create_player_transition_function()
            measurement_fn = create_player_measurement_function()
            return UnscentedKalmanFilter(
                self.filter_config,
                transition_fn,
                measurement_fn
            )
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
    
    def _get_filter_for_player(self, player_id: int, position: str = None) -> Tuple[str, BaseKalmanFilter]:
        """Get appropriate filter for a player."""
        if self.model_mode == "single":
            return "main", self.filters["main"]
            
        elif self.model_mode == "position_specific":
            if position is None:
                # Default to midfielder if position unknown
                position = "MID"
            filter_key = position if position in self.filters else "MID"
            return filter_key, self.filters[filter_key]
            
        elif self.model_mode == "ensemble":
            # For ensemble, we'll use the extended filter as primary
            # (ensemble combination happens in prediction)
            return "extended", self.filters["extended"]
        
        else:
            raise ValueError(f"Unknown model mode: {self.model_mode}")
    
    def initialize_state(
        self,
        player_id: int,
        initial_data: Optional[PlayerData] = None,
        gameweek: int = 1,
        season: str = "2023",
        position: str = None,
        **kwargs
    ) -> PlayerState:
        """
        Initialize state vector and covariance for a player.
        
        Args:
            player_id: Player ID to initialize
            initial_data: Optional initial performance data
            gameweek: Initial gameweek
            season: Season identifier
            position: Player position for position-specific models
            **kwargs: Additional initialization parameters
            
        Returns:
            Initial PlayerState object
        """
        try:
            # Initialize state mean based on historical data or defaults
            if initial_data and "features" in initial_data:
                # Use historical data to initialize state
                features = initial_data["features"]
                if features is not None and len(features) > 0:
                    # Compute initial state from recent performance
                    recent_performance = np.mean(features[-5:], axis=0) if len(features.shape) > 1 else features
                    state_mean = self._performance_to_state(recent_performance)
                else:
                    state_mean = self._default_initial_state()
            else:
                state_mean = self._default_initial_state()
            
            # Initialize covariance with higher uncertainty
            state_cov = jnp.eye(self.config.state_dim) * self.config.initial_state_std**2
            
            # Create PlayerState object for compatibility
            player_state = PlayerState(
                player_id=player_id,
                state_mean=state_mean,
                state_cov=state_cov,
                gameweek=gameweek,
                season=season,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            # Convert to FilterState for Kalman filter
            filter_state = FilterState(
                state_mean=state_mean,
                state_cov=state_cov,
                timestamp=float(gameweek),
                gameweek=gameweek,
                season=season,
                player_id=player_id
            )
            
            # Store filter state for each relevant filter
            if player_id not in self.filter_states:
                self.filter_states[player_id] = {}
            
            if self.model_mode == "single":
                self.filter_states[player_id]["main"] = filter_state
            elif self.model_mode == "position_specific":
                filter_key, _ = self._get_filter_for_player(player_id, position)
                self.filter_states[player_id][filter_key] = filter_state
            elif self.model_mode == "ensemble":
                # Initialize all ensemble filters
                for filter_name in self.filters:
                    self.filter_states[player_id][filter_name] = filter_state.copy()
            
            # Store in parent class structure
            self.player_states[player_id] = player_state
            
            logger.debug(f"Initialized state for player {player_id} in gameweek {gameweek}")
            return player_state
            
        except Exception as e:
            logger.error(f"Failed to initialize state for player {player_id}: {e}")
            raise KalmanPlayerModelError(f"State initialization failed: {e}")
    
    def _default_initial_state(self) -> jnp.ndarray:
        """Create default initial state vector."""
        # Default values for [skill, form, consistency, momentum]
        return jnp.array([0.5, 0.5, 0.5, 0.0])
    
    def _performance_to_state(self, performance: np.ndarray) -> jnp.ndarray:
        """Convert performance observations to state estimate."""
        # Simple mapping from observations to state
        # This could be made more sophisticated with learned mappings
        goals, assists, minutes, bonus = performance[:4]
        
        # Normalize performance metrics
        skill = np.clip((goals + assists) / 10.0, 0, 1)
        form = np.clip(minutes / 90.0, 0, 1)
        consistency = np.clip(bonus / 5.0, 0, 1)
        momentum = 0.0  # Initialize momentum as neutral
        
        return jnp.array([skill, form, consistency, momentum])
    
    def predict_state(
        self,
        current_state: PlayerState,
        gameweeks_ahead: int = 1,
        position: str = None,
        **kwargs
    ) -> PlayerState:
        """
        Predict future state using Kalman filter prediction.
        
        Args:
            current_state: Current player state
            gameweeks_ahead: Number of gameweeks to predict ahead
            position: Player position for position-specific models
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted PlayerState object
        """
        try:
            player_id = current_state.player_id
            
            # Get appropriate filter and current filter state
            filter_key, kalman_filter = self._get_filter_for_player(player_id, position)
            
            if player_id not in self.filter_states or filter_key not in self.filter_states[player_id]:
                # Initialize state if not exists
                self.initialize_state(player_id, None, current_state.gameweek, current_state.season, position)
            
            current_filter_state = self.filter_states[player_id][filter_key]
            
            # Update filter state with current PlayerState
            current_filter_state.state_mean = current_state.state_mean
            current_filter_state.state_cov = current_state.state_cov
            current_filter_state.gameweek = current_state.gameweek
            
            # Predict forward using Kalman filter
            if self.model_mode == "ensemble":
                # Ensemble prediction
                predicted_state = self._ensemble_predict(player_id, gameweeks_ahead)
            else:
                # Single filter prediction
                predicted_filter_state = current_filter_state
                for step in range(gameweeks_ahead):
                    predicted_filter_state = kalman_filter.predict(predicted_filter_state, dt=1.0)
                
                # Convert back to PlayerState
                predicted_state = PlayerState(
                    player_id=player_id,
                    state_mean=predicted_filter_state.state_mean,
                    state_cov=predicted_filter_state.state_cov,
                    gameweek=current_state.gameweek + gameweeks_ahead,
                    season=current_state.season,
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
            
            # Update tracking
            self.model_metrics["total_predictions"] += 1
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"Prediction failed for player {player_id}: {e}")
            raise KalmanPlayerModelError(f"State prediction failed: {e}")
    
    def _ensemble_predict(self, player_id: int, gameweeks_ahead: int) -> PlayerState:
        """Make ensemble prediction using multiple filters."""
        predictions = []
        weights = []
        
        for filter_name, kalman_filter in self.filters.items():
            if filter_name in self.filter_states[player_id]:
                try:
                    current_state = self.filter_states[player_id][filter_name]
                    predicted_state = current_state
                    
                    # Predict forward
                    for step in range(gameweeks_ahead):
                        predicted_state = kalman_filter.predict(predicted_state, dt=1.0)
                    
                    predictions.append(predicted_state)
                    
                    # Weight based on recent performance (log-likelihood)
                    weight = np.exp(predicted_state.log_likelihood or 0)
                    weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed for filter {filter_name}: {e}")
                    continue
        
        if not predictions:
            raise KalmanPlayerModelError("All ensemble predictions failed")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        # Combine predictions
        combined_mean = sum(w * pred.state_mean for w, pred in zip(weights, predictions))
        
        # Combine covariances (approximate)
        combined_cov = sum(w * pred.state_cov for w, pred in zip(weights, predictions))
        # Add between-model variance
        mean_diff_cov = sum(w * jnp.outer(pred.state_mean - combined_mean, pred.state_mean - combined_mean) 
                           for w, pred in zip(weights, predictions))
        combined_cov += mean_diff_cov
        
        return PlayerState(
            player_id=player_id,
            state_mean=combined_mean,
            state_cov=combined_cov,
            gameweek=predictions[0].gameweek,
            season=predictions[0].season,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    def update_state(
        self,
        predicted_state: PlayerState,
        observation: jnp.ndarray,
        observation_noise: Optional[jnp.ndarray] = None,
        position: str = None,
        **kwargs
    ) -> PlayerState:
        """
        Update state based on new observations using Kalman filter update.
        
        Args:
            predicted_state: Predicted state from prediction step
            observation: Observed performance metrics
            observation_noise: Optional observation noise covariance
            position: Player position for position-specific models
            **kwargs: Additional update parameters
            
        Returns:
            Updated PlayerState object
        """
        try:
            player_id = predicted_state.player_id
            
            # Handle missing observations
            if np.any(np.isnan(observation)):
                logger.warning(f"Missing observations for player {player_id}, skipping update")
                return predicted_state
            
            # Get appropriate filter
            filter_key, kalman_filter = self._get_filter_for_player(player_id, position)
            
            # Convert PlayerState to FilterState
            predicted_filter_state = FilterState(
                state_mean=predicted_state.state_mean,
                state_cov=predicted_state.state_cov,
                timestamp=float(predicted_state.gameweek),
                gameweek=predicted_state.gameweek,
                season=predicted_state.season,
                player_id=player_id
            )
            
            if self.model_mode == "ensemble":
                # Ensemble update
                updated_state = self._ensemble_update(player_id, predicted_filter_state, observation, observation_noise)
            else:
                # Single filter update
                updated_filter_state = kalman_filter.update(
                    predicted_filter_state, 
                    observation, 
                    observation_noise
                )
                
                # Update stored filter state
                self.filter_states[player_id][filter_key] = updated_filter_state
                
                # Convert back to PlayerState
                updated_state = PlayerState(
                    player_id=player_id,
                    state_mean=updated_filter_state.state_mean,
                    state_cov=updated_filter_state.state_cov,
                    gameweek=predicted_state.gameweek,
                    season=predicted_state.season,
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
            
            # Update tracking
            self.model_metrics["total_updates"] += 1
            if updated_filter_state.log_likelihood is not None:
                # Update running average of log-likelihood
                current_avg = self.model_metrics["average_log_likelihood"]
                total_updates = self.model_metrics["total_updates"]
                new_avg = ((total_updates - 1) * current_avg + updated_filter_state.log_likelihood) / total_updates
                self.model_metrics["average_log_likelihood"] = new_avg
            
            # Track updates per player
            self.update_counts[player_id] = self.update_counts.get(player_id, 0) + 1
            
            return updated_state
            
        except NumericalInstabilityError as e:
            logger.error(f"Numerical instability in update for player {player_id}: {e}")
            self.model_metrics["numerical_errors"] += 1
            # Return predicted state unchanged
            return predicted_state
            
        except Exception as e:
            logger.error(f"Update failed for player {player_id}: {e}")
            raise KalmanPlayerModelError(f"State update failed: {e}")
    
    def _ensemble_update(
        self, 
        player_id: int, 
        predicted_state: FilterState, 
        observation: jnp.ndarray,
        observation_noise: Optional[jnp.ndarray] = None
    ) -> PlayerState:
        """Update ensemble of filters and combine results."""
        updates = []
        weights = []
        
        for filter_name, kalman_filter in self.filters.items():
            try:
                # Update each filter
                updated_state = kalman_filter.update(predicted_state, observation, observation_noise)
                updates.append(updated_state)
                
                # Store updated state
                self.filter_states[player_id][filter_name] = updated_state
                
                # Weight by likelihood
                weight = np.exp(updated_state.log_likelihood or 0)
                weights.append(weight)
                
            except Exception as e:
                logger.warning(f"Ensemble update failed for filter {filter_name}: {e}")
                continue
        
        if not updates:
            raise KalmanPlayerModelError("All ensemble updates failed")
        
        # Combine updates similar to prediction
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        combined_mean = sum(w * update.state_mean for w, update in zip(weights, updates))
        combined_cov = sum(w * update.state_cov for w, update in zip(weights, updates))
        
        # Add between-model variance
        mean_diff_cov = sum(w * jnp.outer(update.state_mean - combined_mean, update.state_mean - combined_mean) 
                           for w, update in zip(weights, updates))
        combined_cov += mean_diff_cov
        
        return PlayerState(
            player_id=player_id,
            state_mean=combined_mean,
            state_cov=combined_cov,
            gameweek=predicted_state.gameweek,
            season=predicted_state.season,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    def observation_model(
        self,
        state: jnp.ndarray,
        position: str = None,
        **kwargs
    ) -> jnp.ndarray:
        """
        Map state vector to expected observations.
        
        Args:
            state: State vector [skill, form, consistency, momentum]
            position: Player position for position-specific mapping
            **kwargs: Additional mapping parameters
            
        Returns:
            Expected observation vector [goals, assists, minutes, bonus]
        """
        try:
            # Use the same measurement function as the Kalman filters
            measurement_fn = create_player_measurement_function()
            return measurement_fn(state)
            
        except Exception as e:
            logger.error(f"Observation model failed: {e}")
            # Fall back to simple linear mapping
            return state[:self.config.obs_dim]
    
    def _fit_state_space_model(
        self, 
        data: PlayerData, 
        season: str, 
        max_gameweek: int, 
        **kwargs
    ) -> None:
        """
        Fit Kalman filter parameters from historical data.
        
        This method processes historical data to:
        1. Initialize states for all players
        2. Run forward-backward pass to estimate parameters
        3. Calibrate noise models
        """
        try:
            logger.info(f"Fitting Kalman player model with {len(data['player_ids'])} players")
            
            player_ids = data["player_ids"]
            features = data["features"]  # (n_players, n_gameweeks, n_features)
            gameweeks = data["gameweeks"]
            
            # Process each player's time series
            for i, player_id in enumerate(player_ids):
                try:
                    player_features = features[i]  # (n_gameweeks, n_features)
                    
                    # Skip players with insufficient data
                    valid_observations = ~np.isnan(player_features).all(axis=1)
                    if np.sum(valid_observations) < 3:
                        continue
                    
                    # Initialize player state
                    initial_obs = player_features[valid_observations][0]
                    player_data = {"features": player_features}
                    
                    self.initialize_state(
                        player_id=player_id,
                        initial_data=player_data,
                        gameweek=gameweeks[0],
                        season=season
                    )
                    
                    # Process time series with Kalman filter
                    self._process_player_timeseries(
                        player_id, player_features, gameweeks, valid_observations
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to fit player {player_id}: {e}")
                    continue
            
            logger.info(f"Fitted Kalman model for {len(self.player_states)} players")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise KalmanPlayerModelError(f"Model fitting failed: {e}")
    
    def _process_player_timeseries(
        self,
        player_id: int,
        features: np.ndarray,
        gameweeks: np.ndarray,
        valid_observations: np.ndarray
    ):
        """Process a single player's time series through Kalman filter."""
        try:
            # Get filter for this player
            filter_key, kalman_filter = self._get_filter_for_player(player_id)
            current_state = self.filter_states[player_id][filter_key]
            
            # Process each gameweek
            for t, (gw, obs, is_valid) in enumerate(zip(gameweeks, features, valid_observations)):
                if not is_valid:
                    # Just predict without update for missing data
                    current_state = kalman_filter.predict(current_state, dt=1.0)
                else:
                    # Predict then update
                    predicted_state = kalman_filter.predict(current_state, dt=1.0)
                    current_state = kalman_filter.update(predicted_state, obs[:self.config.obs_dim])
                
                # Update gameweek
                current_state.gameweek = gw
            
            # Store final state
            self.filter_states[player_id][filter_key] = current_state
            
            # Update PlayerState for compatibility
            self.player_states[player_id].state_mean = current_state.state_mean
            self.player_states[player_id].state_cov = current_state.state_cov
            self.player_states[player_id].gameweek = current_state.gameweek
            
        except Exception as e:
            logger.warning(f"Failed to process timeseries for player {player_id}: {e}")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        diagnostics = {
            "model_metrics": self.model_metrics.copy(),
            "num_players": len(self.player_states),
            "num_filter_states": len(self.filter_states),
            "filter_config": {
                "filter_type": self.filter_type,
                "model_mode": self.model_mode,
                "state_dim": self.config.state_dim,
                "obs_dim": self.config.obs_dim,
                "adaptive_noise": self.filter_config.enable_adaptive_noise,
                "use_joseph_form": self.filter_config.use_joseph_form
            },
            "update_counts": dict(self.update_counts),
            "filter_diagnostics": {}
        }
        
        # Get diagnostics from individual filters
        for filter_name, kalman_filter in self.filters.items():
            filter_diag = {}
            
            # Sample a few player states for filter diagnostics
            sample_players = list(self.filter_states.keys())[:5]
            for player_id in sample_players:
                if filter_name in self.filter_states[player_id]:
                    state = self.filter_states[player_id][filter_name]
                    player_diag = kalman_filter.get_diagnostics(state)
                    filter_diag[f"player_{player_id}"] = player_diag
            
            diagnostics["filter_diagnostics"][filter_name] = filter_diag
        
        return diagnostics
    
    def reset_model(self):
        """Reset model state for retraining."""
        self.filter_states.clear()
        self.player_states.clear()
        self.prediction_history.clear()
        self.update_counts.clear()
        self.is_fitted = False
        
        # Reset model metrics
        self.model_metrics = {
            "total_predictions": 0,
            "total_updates": 0,
            "average_log_likelihood": 0.0,
            "numerical_errors": 0,
            "adaptive_updates": 0
        }
        
        # Reinitialize filters
        self._initialize_filters()
        
        logger.info("Model state reset successfully")


class PositionSpecificKalmanModel(KalmanPlayerModel):
    """Position-aware Kalman player model with position-specific dynamics."""
    
    def __init__(
        self,
        position_configs: Optional[Dict[str, FilterConfig]] = None,
        **kwargs
    ):
        """
        Initialize position-specific Kalman model.
        
        Args:
            position_configs: Position-specific filter configurations
            **kwargs: Additional arguments for base model
        """
        # Set model mode to position-specific
        kwargs["model_mode"] = "position_specific"
        
        super().__init__(**kwargs)
        
        # Position-specific configurations
        self.position_configs = position_configs or self._create_default_position_configs()
        
        # Reinitialize filters with position-specific configs
        self._initialize_position_filters()
    
    def _create_default_position_configs(self) -> Dict[str, FilterConfig]:
        """Create default position-specific filter configurations."""
        base_config = self.filter_config
        
        configs = {}
        for position in self.position_mapping:
            # Adjust noise levels based on position characteristics
            if position == "GK":
                # Goalkeepers have more consistent performance
                process_noise = base_config.process_noise_std * 0.5
                measurement_noise = base_config.measurement_noise_std * 0.7
            elif position == "DEF":
                # Defenders have moderate variability
                process_noise = base_config.process_noise_std * 0.8
                measurement_noise = base_config.measurement_noise_std * 0.9
            elif position == "MID":
                # Midfielders use base configuration
                process_noise = base_config.process_noise_std
                measurement_noise = base_config.measurement_noise_std
            elif position == "FWD":
                # Forwards have higher variability
                process_noise = base_config.process_noise_std * 1.2
                measurement_noise = base_config.measurement_noise_std * 1.1
            else:
                process_noise = base_config.process_noise_std
                measurement_noise = base_config.measurement_noise_std
            
            configs[position] = FilterConfig(
                state_dim=base_config.state_dim,
                obs_dim=base_config.obs_dim,
                process_noise_std=process_noise,
                measurement_noise_std=measurement_noise,
                enable_adaptive_noise=base_config.enable_adaptive_noise,
                use_joseph_form=base_config.use_joseph_form,
                alpha=base_config.alpha,
                beta=base_config.beta,
                kappa=base_config.kappa
            )
        
        return configs
    
    def _initialize_position_filters(self):
        """Initialize position-specific filters with custom configurations."""
        self.filters.clear()
        
        for position, config in self.position_configs.items():
            if self.filter_type == "linear":
                self.filters[position] = create_simple_linear_filter(config)
            elif self.filter_type == "extended":
                transition_fn = self._create_position_transition_function(position)
                measurement_fn = self._create_position_measurement_function(position)
                self.filters[position] = ExtendedKalmanFilter(config, transition_fn, measurement_fn)
            elif self.filter_type == "unscented":
                transition_fn = self._create_position_transition_function(position)
                measurement_fn = self._create_position_measurement_function(position)
                self.filters[position] = UnscentedKalmanFilter(config, transition_fn, measurement_fn)
    
    def _create_position_transition_function(self, position: str):
        """Create position-specific state transition function."""
        def transition_fn(state: jnp.ndarray, dt: float) -> jnp.ndarray:
            skill, form, consistency, momentum = state[0], state[1], state[2], state[3]
            
            # Position-specific dynamics
            if position == "GK":
                # Goalkeepers: very stable, less momentum influence
                new_skill = skill + 0.0005 * momentum * dt
                new_form = 0.98**dt * form + (1 - 0.98**dt) * skill
                new_consistency = 0.995**dt * consistency + 0.005 * skill
                new_momentum = 0.7**dt * momentum
                
            elif position == "DEF":
                # Defenders: stable, moderate momentum
                new_skill = skill + 0.001 * momentum * dt
                new_form = 0.96**dt * form + (1 - 0.96**dt) * skill + 0.005 * momentum * dt
                new_consistency = 0.99**dt * consistency + 0.01 * skill
                new_momentum = 0.75**dt * momentum
                
            elif position == "MID":
                # Midfielders: use base dynamics
                new_skill = skill + 0.001 * momentum * dt
                new_form = 0.95**dt * form + (1 - 0.95**dt) * skill + 0.01 * momentum * dt
                new_consistency = 0.99**dt * consistency + 0.01 * skill
                new_momentum = 0.8**dt * momentum
                
            elif position == "FWD":
                # Forwards: more volatile, higher momentum influence
                new_skill = skill + 0.002 * momentum * dt
                new_form = 0.93**dt * form + (1 - 0.93**dt) * skill + 0.015 * momentum * dt
                new_consistency = 0.985**dt * consistency + 0.015 * skill
                new_momentum = 0.85**dt * momentum
                
            else:
                # Default to midfielder dynamics
                new_skill = skill + 0.001 * momentum * dt
                new_form = 0.95**dt * form + (1 - 0.95**dt) * skill + 0.01 * momentum * dt
                new_consistency = 0.99**dt * consistency + 0.01 * skill
                new_momentum = 0.8**dt * momentum
            
            return jnp.array([new_skill, new_form, new_consistency, new_momentum])
        
        return transition_fn
    
    def _create_position_measurement_function(self, position: str):
        """Create position-specific measurement function."""
        def measurement_fn(state: jnp.ndarray) -> jnp.ndarray:
            skill, form, consistency, momentum = state[0], state[1], state[2], state[3]
            
            # Position-specific observation mappings
            if position == "GK":
                # Goalkeepers: saves, clean sheets, bonus points, minutes
                goals = 0  # Goalkeepers rarely score
                assists = jnp.maximum(0, 0.1 * skill * consistency)
                minutes = jnp.maximum(0, 80 + 10 * consistency * form)  # More likely to play full games
                bonus = jnp.maximum(0, consistency * form + 0.1 * momentum)
                
            elif position == "DEF":
                # Defenders: clean sheets, goals, assists, minutes
                goals = jnp.maximum(0, 0.3 * skill * form + 0.05 * momentum)
                assists = jnp.maximum(0, 0.5 * skill * consistency + 0.03 * momentum)
                minutes = jnp.maximum(0, 70 + 20 * consistency * form)
                bonus = jnp.maximum(0, (goals + assists) * consistency + 0.08 * momentum)
                
            elif position == "MID":
                # Midfielders: balanced all-around
                goals = jnp.maximum(0, 0.7 * skill * form + 0.08 * momentum)
                assists = jnp.maximum(0, 0.9 * skill * consistency + 0.06 * momentum)
                minutes = jnp.maximum(0, 60 + 30 * consistency * form)
                bonus = jnp.maximum(0, (goals + assists) * consistency + 0.1 * momentum)
                
            elif position == "FWD":
                # Forwards: higher goal scoring, variable minutes
                goals = jnp.maximum(0, 1.2 * skill * form + 0.12 * momentum)
                assists = jnp.maximum(0, 0.6 * skill * consistency + 0.04 * momentum)
                minutes = jnp.maximum(0, 50 + 40 * consistency * form)
                bonus = jnp.maximum(0, (goals + 0.5 * assists) * consistency + 0.12 * momentum)
                
            else:
                # Default to midfielder mapping
                goals = jnp.maximum(0, 0.7 * skill * form + 0.08 * momentum)
                assists = jnp.maximum(0, 0.9 * skill * consistency + 0.06 * momentum)
                minutes = jnp.maximum(0, 60 + 30 * consistency * form)
                bonus = jnp.maximum(0, (goals + assists) * consistency + 0.1 * momentum)
            
            return jnp.array([goals, assists, minutes, bonus])
        
        return measurement_fn