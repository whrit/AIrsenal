"""
Integration between StateManager and AdaptivePlayerModel.

This module provides integration utilities and examples for using the StateManager
with the AdaptivePlayerModel, demonstrating how state management enhances the
model's capabilities for tracking player abilities over time.

Key Features:
- StateAwareAdaptiveModel: Enhanced AdaptivePlayerModel with state management
- Automatic state persistence and recovery
- State-driven model updates and predictions
- Historical state analysis and reporting
- Integration with existing AIrsenal pipeline

Usage:
    ```python
    from airsenal.framework.state_model_integration import StateAwareAdaptiveModel
    from airsenal.framework.state_manager import StateManager, MemoryStateRepository
    
    # Create state-aware model
    state_manager = StateManager(MemoryStateRepository())
    model = StateAwareAdaptiveModel(
        config=StateSpaceConfig(),
        state_manager=state_manager
    )
    
    # Train and use as normal AdaptivePlayerModel
    model.fit(training_data, season="2023", max_gameweek=15)
    predictions = model.predict(player_ids=[123, 456], gameweeks_ahead=3)
    
    # Access state management features
    state_history = await model.get_player_state_history(123)
    model.create_model_checkpoint("checkpoint_1")
    ```
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sqlalchemy.orm import Session

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
)
from airsenal.framework.state_manager import (
    StateManager,
    StateSnapshot,
    StateOperationType,
)

logger = logging.getLogger(__name__)


class StateAwareAdaptiveModel(AdaptivePlayerModel):
    """
    Enhanced AdaptivePlayerModel with integrated state management.
    
    This class extends the AdaptivePlayerModel to provide:
    - Automatic state persistence and recovery
    - State history tracking for analysis
    - Enhanced rollback capabilities
    - State-driven model updates
    - Integration with external state storage
    """
    
    def __init__(
        self,
        config: StateSpaceConfig,
        state_manager: StateManager,
        learning_rate: float = 0.01,
        decay_factor: float = 0.95,
        random_seed: int = 42,
        enable_auto_checkpoint: bool = True,
        checkpoint_interval: int = 10,  # gameweeks
    ):
        """
        Initialize state-aware adaptive model.
        
        Args:
            config: State-space model configuration
            state_manager: StateManager instance for state persistence
            learning_rate: Rate of adaptation to new data
            decay_factor: Exponential decay for historical data importance
            random_seed: Random seed for reproducible initialization
            enable_auto_checkpoint: Whether to automatically create checkpoints
            checkpoint_interval: Gameweeks between automatic checkpoints
        """
        super().__init__(config, learning_rate, decay_factor, random_seed)
        
        self.state_manager = state_manager
        self.enable_auto_checkpoint = enable_auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self._last_checkpoint_gameweek = 0
        
        # Override player_states to use StateManager
        self._use_external_state_manager = True
    
    async def get_player_state(self, player_id: int) -> Optional[PlayerState]:
        """Get player state from StateManager."""
        return await self.state_manager.get_player_state(player_id)
    
    async def set_player_state(
        self, 
        player_id: int, 
        state: PlayerState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set player state using StateManager."""
        return await self.state_manager.update_player_state(
            player_id, state, metadata
        )
    
    async def initialize_state(
        self,
        player_id: int,
        initial_data: Optional[Dict[str, Any]] = None,
        gameweek: int = 1,
        season: str = "2023",
        **kwargs,
    ) -> PlayerState:
        """
        Initialize state vector and covariance for a player with persistence.
        
        This implementation first checks if a state already exists in the
        StateManager before creating a new one.
        """
        # Check if state already exists
        existing_state = await self.get_player_state(player_id)
        if existing_state is not None:
            logger.info(f"Found existing state for player {player_id}")
            return existing_state
        
        # Create new initial state
        if initial_data is not None and "features" in initial_data:
            # Initialize from historical data
            features = np.array(initial_data["features"])
            if features.ndim == 2 and features.shape[0] > 0:
                # Use mean and covariance of historical performance
                state_mean = np.mean(features[:, :self.config.state_dim], axis=0)
                state_cov = np.cov(features[:, :self.config.state_dim].T)
                
                # Add small regularization to ensure positive definiteness
                state_cov += np.eye(self.config.state_dim) * 0.01
            else:
                # Fallback to default initialization
                state_mean = np.zeros(self.config.state_dim)
                state_cov = np.eye(self.config.state_dim) * self.config.initial_state_std**2
        else:
            # Default initialization
            state_mean = np.zeros(self.config.state_dim)
            state_cov = np.eye(self.config.state_dim) * self.config.initial_state_std**2
        
        # Create initial state
        initial_state = PlayerState(
            player_id=player_id,
            state_mean=state_mean,
            state_cov=state_cov,
            gameweek=gameweek,
            season=season,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        
        # Store in StateManager
        success = await self.set_player_state(
            player_id, 
            initial_state,
            metadata={
                "operation": "initialization",
                "source": "adaptive_model",
                "has_initial_data": initial_data is not None,
            }
        )
        
        if not success:
            logger.warning(f"Failed to store initial state for player {player_id}")
        
        return initial_state
    
    async def predict_state(
        self,
        current_state: PlayerState,
        gameweeks_ahead: int = 1,
        **kwargs,
    ) -> PlayerState:
        """
        Predict future state with automatic state tracking.
        """
        # Simple state transition model - can be enhanced with learned dynamics
        predicted_mean = current_state.state_mean.copy()
        predicted_cov = current_state.state_cov.copy()
        
        # Add process noise for each gameweek step
        process_noise = np.eye(self.config.state_dim) * (
            self.config.process_noise_std**2 * gameweeks_ahead
        )
        predicted_cov += process_noise
        
        # Create predicted state
        predicted_state = PlayerState(
            player_id=current_state.player_id,
            state_mean=predicted_mean,
            state_cov=predicted_cov,
            gameweek=current_state.gameweek + gameweeks_ahead,
            season=current_state.season,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        
        return predicted_state
    
    async def update_state(
        self,
        predicted_state: PlayerState,
        observation: Union[np.ndarray, List[float]],
        observation_noise: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PlayerState:
        """
        Update state based on new observations with persistence.
        """
        observation = np.array(observation)
        
        if observation_noise is None:
            observation_noise = np.eye(len(observation)) * self.config.measurement_noise_std**2
        
        # Simple Kalman-like update
        # Prediction error
        predicted_obs = self.observation_model(predicted_state.state_mean)
        innovation = observation - predicted_obs
        
        # Innovation covariance
        H = self._get_observation_jacobian(predicted_state.state_mean)
        S = H @ predicted_state.state_cov @ H.T + observation_noise
        
        # Kalman gain
        K = predicted_state.state_cov @ H.T @ np.linalg.inv(S)
        
        # State update
        updated_mean = predicted_state.state_mean + K @ innovation
        updated_cov = (np.eye(self.config.state_dim) - K @ H) @ predicted_state.state_cov
        
        # Create updated state
        updated_state = PlayerState(
            player_id=predicted_state.player_id,
            state_mean=updated_mean,
            state_cov=updated_cov,
            gameweek=predicted_state.gameweek,
            season=predicted_state.season,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        
        # Store updated state
        success = await self.set_player_state(
            predicted_state.player_id,
            updated_state,
            metadata={
                "operation": "measurement_update",
                "observation": observation.tolist(),
                "innovation_magnitude": float(np.linalg.norm(innovation)),
            }
        )
        
        if not success:
            logger.warning(f"Failed to store updated state for player {predicted_state.player_id}")
        
        return updated_state
    
    def observation_model(
        self,
        state: Union[np.ndarray, List[float]],
        **kwargs,
    ) -> np.ndarray:
        """
        Map state vector to expected observations.
        
        Simple linear observation model - can be enhanced with learned mappings.
        """
        state = np.array(state)
        
        # Simple mapping: state components directly map to observations
        # In practice, this would be learned from data
        if len(state) >= self.config.obs_dim:
            return state[:self.config.obs_dim]
        else:
            # Pad with zeros if state dimension is smaller
            obs = np.zeros(self.config.obs_dim)
            obs[:len(state)] = state
            return obs
    
    def _get_observation_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Get Jacobian matrix of observation model."""
        # For linear observation model, Jacobian is constant
        H = np.zeros((self.config.obs_dim, self.config.state_dim))
        min_dim = min(self.config.obs_dim, self.config.state_dim)
        np.fill_diagonal(H[:min_dim, :min_dim], 1.0)
        return H
    
    async def fit(
        self,
        data: Dict[str, Any],
        season: str,
        max_gameweek: int,
        dbsession: Optional[Session] = None,
        **kwargs,
    ) -> StateAwareAdaptiveModel:
        """
        Fit the model with automatic state initialization and checkpointing.
        """
        logger.info(f"Fitting StateAwareAdaptiveModel for season {season}, max_gameweek {max_gameweek}")
        
        # Initialize states from StateManager if available
        if hasattr(self.state_manager, 'initialize_from_historical_data') and dbsession:
            try:
                init_results = await self.state_manager.initialize_from_historical_data(
                    dbsession=dbsession,
                    season=season,
                    max_gameweek=max_gameweek,
                )
                logger.info(f"Initialized {init_results['initialized_count']} states from historical data")
            except Exception as e:
                logger.warning(f"Failed to initialize from historical data: {e}")
        
        # Call parent fit method
        fitted_model = super().fit(data, season, max_gameweek, dbsession, **kwargs)
        
        # Create checkpoint after fitting
        if self.enable_auto_checkpoint:
            await self.create_model_checkpoint(f"post_fit_{season}_{max_gameweek}")
        
        return fitted_model
    
    async def update_with_recent_data(
        self,
        recent_data: Dict[str, Any],
        gameweek: int,
        season: str,
        **kwargs,
    ) -> StateAwareAdaptiveModel:
        """
        Update model with recent data and automatic checkpointing.
        """
        logger.info(f"Updating StateAwareAdaptiveModel with data from gameweek {gameweek}")
        
        # Call parent update method
        updated_model = super().update_with_recent_data(recent_data, gameweek, season, **kwargs)
        
        # Create automatic checkpoint if needed
        if (self.enable_auto_checkpoint and 
            gameweek - self._last_checkpoint_gameweek >= self.checkpoint_interval):
            await self.create_model_checkpoint(f"auto_{season}_gw{gameweek}")
            self._last_checkpoint_gameweek = gameweek
        
        return updated_model
    
    async def predict(
        self,
        player_ids: Union[List[int], np.ndarray],
        gameweeks_ahead: int = 3,
        features: Optional[Union[np.ndarray, List[List[float]]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate predictions with enhanced state tracking.
        """
        logger.debug(f"Generating predictions for {len(player_ids)} players, {gameweeks_ahead} gameweeks ahead")
        
        player_ids = np.asarray(player_ids)
        n_players = len(player_ids)
        
        # Initialize prediction arrays
        predictions = np.zeros((n_players, gameweeks_ahead))
        uncertainties = np.zeros((n_players, gameweeks_ahead))
        predicted_states = {}
        state_trajectories = {}
        
        for i, player_id in enumerate(player_ids):
            # Get current state from StateManager
            current_state = await self.get_player_state(player_id)
            
            if current_state is None:
                # Initialize state for unknown player
                current_state = await self.initialize_state(
                    player_id=player_id,
                    gameweek=self.last_update_gameweek or 1,
                    season=self.last_season or "2023",
                )
            
            # Track state trajectory for this player
            trajectory = [current_state]
            
            # Predict state evolution
            pred_state = current_state
            player_predictions = []
            player_uncertainties = []
            
            for gw in range(gameweeks_ahead):
                # Predict next state
                pred_state = await self.predict_state(pred_state, gameweeks_ahead=1, **kwargs)
                trajectory.append(pred_state)
                
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
            state_trajectories[player_id] = trajectory
        
        return {
            "player_ids": player_ids,
            "predictions": predictions,
            "uncertainty": uncertainties,
            "states": predicted_states,
            "state_trajectories": state_trajectories,
            "metadata": {
                "gameweeks_ahead": gameweeks_ahead,
                "model_type": self.__class__.__name__,
                "state_dim": self.config.state_dim,
                "obs_dim": self.config.obs_dim,
                "state_manager_enabled": True,
            },
        }
    
    async def get_player_state_history(
        self,
        player_id: int,
        from_gameweek: Optional[int] = None,
        to_gameweek: Optional[int] = None,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[StateSnapshot]:
        """Get state history for a player."""
        return await self.state_manager.get_state_history(
            player_id=player_id,
            from_gameweek=from_gameweek,
            to_gameweek=to_gameweek,
            season=season,
            limit=limit,
        )
    
    async def create_model_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """
        Create a checkpoint of all current model states.
        
        Args:
            checkpoint_name: Name for the checkpoint
            
        Returns:
            Dictionary with checkpoint information
        """
        logger.info(f"Creating model checkpoint: {checkpoint_name}")
        
        try:
            # Get all current player states
            player_ids = await self.state_manager.repository.list_player_ids()
            checkpoint_count = 0
            
            for player_id in player_ids:
                snapshot_id = await self.state_manager.create_state_snapshot(
                    player_id=player_id,
                    metadata={
                        "checkpoint_name": checkpoint_name,
                        "model_type": self.__class__.__name__,
                        "checkpoint_time": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if snapshot_id:
                    checkpoint_count += 1
            
            checkpoint_info = {
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "player_count": len(player_ids),
                "snapshots_created": checkpoint_count,
                "success": checkpoint_count == len(player_ids),
            }
            
            logger.info(f"Checkpoint created: {checkpoint_count}/{len(player_ids)} players")
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_name}: {e}")
            return {
                "checkpoint_name": checkpoint_name,
                "error": str(e),
                "success": False,
            }
    
    async def rollback_to_checkpoint(
        self, 
        checkpoint_name: str,
        player_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Rollback model states to a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to rollback to
            player_ids: Optional list of specific players to rollback
            
        Returns:
            Dictionary with rollback results
        """
        logger.info(f"Rolling back to checkpoint: {checkpoint_name}")
        
        try:
            if player_ids is None:
                player_ids = await self.state_manager.repository.list_player_ids()
            
            rollback_count = 0
            failed_rollbacks = []
            
            for player_id in player_ids:
                # Find checkpoint snapshot
                history = await self.get_player_state_history(player_id)
                checkpoint_snapshot = None
                
                for snapshot in history:
                    if (snapshot.metadata.get("checkpoint_name") == checkpoint_name):
                        checkpoint_snapshot = snapshot
                        break
                
                if checkpoint_snapshot:
                    success = await self.state_manager.rollback_player_state(
                        player_id, to_snapshot_id=checkpoint_snapshot.snapshot_id
                    )
                    if success:
                        rollback_count += 1
                    else:
                        failed_rollbacks.append(player_id)
                else:
                    failed_rollbacks.append(player_id)
            
            rollback_info = {
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requested_players": len(player_ids),
                "successful_rollbacks": rollback_count,
                "failed_rollbacks": len(failed_rollbacks),
                "failed_player_ids": failed_rollbacks,
                "success": len(failed_rollbacks) == 0,
            }
            
            logger.info(f"Rollback completed: {rollback_count}/{len(player_ids)} players")
            return rollback_info
            
        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint {checkpoint_name}: {e}")
            return {
                "checkpoint_name": checkpoint_name,
                "error": str(e),
                "success": False,
            }
    
    async def analyze_state_evolution(
        self,
        player_id: int,
        analysis_window: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze how a player's state has evolved over time.
        
        Args:
            player_id: Player to analyze
            analysis_window: Number of recent states to analyze
            
        Returns:
            Dictionary with analysis results
        """
        history = await self.get_player_state_history(
            player_id=player_id, 
            limit=analysis_window
        )
        
        if len(history) < 2:
            return {
                "player_id": player_id,
                "error": "Insufficient history for analysis",
                "history_length": len(history),
            }
        
        # Extract state means over time
        states = np.array([snapshot.state.state_mean for snapshot in history])
        gameweeks = [snapshot.state.gameweek for snapshot in history]
        timestamps = [snapshot.timestamp for snapshot in history]
        
        # Compute statistics
        state_evolution = {
            "player_id": player_id,
            "analysis_window": len(history),
            "gameweek_range": [min(gameweeks), max(gameweeks)],
            "time_range": [
                min(timestamps).isoformat(),
                max(timestamps).isoformat()
            ],
            "state_statistics": {
                "initial_state": states[0].tolist(),
                "final_state": states[-1].tolist(),
                "mean_state": np.mean(states, axis=0).tolist(),
                "std_state": np.std(states, axis=0).tolist(),
                "total_change": (states[-1] - states[0]).tolist(),
                "change_magnitude": float(np.linalg.norm(states[-1] - states[0])),
            },
            "trends": {
                "state_names": self.config.state_names,
                "trends_per_component": [],
            }
        }
        
        # Analyze trends for each state component
        for i, state_name in enumerate(self.config.state_names):
            component_values = states[:, i]
            
            # Simple linear trend
            x = np.arange(len(component_values))
            trend_coef = np.polyfit(x, component_values, 1)[0]
            
            state_evolution["trends"]["trends_per_component"].append({
                "component": state_name,
                "trend_coefficient": float(trend_coef),
                "trend_direction": "increasing" if trend_coef > 0 else "decreasing",
                "volatility": float(np.std(component_values)),
                "range": [float(np.min(component_values)), float(np.max(component_values))],
            })
        
        return state_evolution
    
    async def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics including state management metrics."""
        diagnostics = {
            "model_info": {
                "class": self.__class__.__name__,
                "config": {
                    "state_dim": self.config.state_dim,
                    "obs_dim": self.config.obs_dim,
                    "learning_rate": self.learning_rate,
                    "decay_factor": self.decay_factor,
                },
                "fitted": self.is_fitted,
                "last_update": {
                    "gameweek": self.last_update_gameweek,
                    "season": self.last_season,
                },
            },
            "state_management": await self.state_manager.get_manager_metrics(),
        }
        
        # Add model-specific statistics
        if self.is_fitted:
            player_ids = await self.state_manager.repository.list_player_ids()
            diagnostics["model_statistics"] = {
                "tracked_players": len(player_ids),
                "feature_importance": self.get_feature_importance(),
            }
        
        return diagnostics
    
    def _fit_state_space_model(
        self, data: Dict[str, Any], season: str, max_gameweek: int, **kwargs
    ) -> None:
        """
        Fit state-space model parameters from data.
        
        This is a simple implementation - in practice, you would use
        more sophisticated parameter learning techniques.
        """
        # Initialize state-space matrices with simple values
        self.transition_matrix = np.eye(self.config.state_dim)
        self.observation_matrix = np.eye(min(self.config.state_dim, self.config.obs_dim))
        
        self.process_noise_cov = (
            np.eye(self.config.state_dim) * self.config.process_noise_std**2
        )
        self.measurement_noise_cov = (
            np.eye(self.config.obs_dim) * self.config.measurement_noise_std**2
        )
        
        logger.info("Fitted simple state-space model parameters")


# Utility functions for integration
async def migrate_model_states(
    old_model: AdaptivePlayerModel,
    new_state_manager: StateManager,
    season: str,
    gameweek: int,
) -> Dict[str, Any]:
    """
    Migrate states from an existing AdaptivePlayerModel to StateManager.
    
    Args:
        old_model: Existing model with states to migrate
        new_state_manager: StateManager to migrate states to
        season: Season identifier for migrated states
        gameweek: Gameweek identifier for migrated states
        
    Returns:
        Dictionary with migration results
    """
    migration_results = {
        "migrated_count": 0,
        "failed_count": 0,
        "total_players": len(old_model.player_states),
        "failed_players": [],
    }
    
    for player_id, player_state in old_model.player_states.items():
        try:
            # Update state with migration metadata
            migrated_state = PlayerState(
                player_id=player_state.player_id,
                state_mean=player_state.state_mean,
                state_cov=player_state.state_cov,
                gameweek=gameweek,
                season=season,
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            
            success = await new_state_manager.update_player_state(
                player_id,
                migrated_state,
                metadata={
                    "migration_source": "AdaptivePlayerModel",
                    "original_gameweek": player_state.gameweek,
                    "original_season": player_state.season,
                }
            )
            
            if success:
                migration_results["migrated_count"] += 1
            else:
                migration_results["failed_count"] += 1
                migration_results["failed_players"].append(player_id)
                
        except Exception as e:
            logger.error(f"Failed to migrate state for player {player_id}: {e}")
            migration_results["failed_count"] += 1
            migration_results["failed_players"].append(player_id)
    
    logger.info(f"Migration completed: {migration_results['migrated_count']}/{migration_results['total_players']} states migrated")
    return migration_results


def create_state_aware_model_factory(
    state_manager: StateManager,
    default_config: Optional[StateSpaceConfig] = None,
) -> callable:
    """
    Create a factory function for StateAwareAdaptiveModel instances.
    
    Args:
        state_manager: StateManager instance to use for all models
        default_config: Default configuration for models
        
    Returns:
        Factory function that creates StateAwareAdaptiveModel instances
    """
    def factory(
        config: Optional[StateSpaceConfig] = None,
        **kwargs
    ) -> StateAwareAdaptiveModel:
        """Create a new StateAwareAdaptiveModel instance."""
        effective_config = config or default_config or StateSpaceConfig()
        return StateAwareAdaptiveModel(
            config=effective_config,
            state_manager=state_manager,
            **kwargs
        )
    
    return factory