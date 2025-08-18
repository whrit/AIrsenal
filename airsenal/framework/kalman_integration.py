"""
Kalman Filter Integration with AIrsenal State Management

This module provides integration between the Kalman filter system and AIrsenal's
StateManager and ModelPersistence frameworks. It enables seamless saving/loading
of Kalman filter states and model checkpoints.

Key Features:
- KalmanStateRepository for efficient state storage
- Integration with StateManager for distributed state management
- ModelPersistence integration for Kalman filter checkpoints
- Conversion utilities between FilterState and PlayerState
- Batch operations for multiple players
- State history tracking and rollback capabilities

Classes:
    KalmanStateRepository: StateRepository implementation for Kalman filters
    KalmanModelPersistence: ModelPersistence extension for Kalman models
    FilterStateConverter: Utility for state format conversions
    KalmanStateManager: StateManager integration for Kalman filters

Usage:
    ```python
    from airsenal.framework.kalman_integration import (
        KalmanStateManager, KalmanModelPersistence
    )
    
    # Initialize integrated state management
    state_manager = KalmanStateManager(
        repository=redis_repository,
        kalman_model=kalman_player_model
    )
    
    # Save/load player states
    await state_manager.save_kalman_state(player_id, filter_state)
    loaded_state = await state_manager.load_kalman_state(player_id)
    
    # Model persistence
    persistence = KalmanModelPersistence(kalman_model)
    checkpoint_id = await persistence.save_kalman_checkpoint(
        "best_model", gameweek=15, season="2023"
    )
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from sqlalchemy.orm import Session

from airsenal.framework.adaptive_player_model import PlayerState, StateSpaceConfig
from airsenal.framework.kalman_filter import FilterState, FilterConfig
from airsenal.framework.kalman_player_model import KalmanPlayerModel, KalmanPlayerModelError
from airsenal.framework.model_persistence import (
    ModelPersistence,
    PersistenceConfig,
    CheckpointType,
    ModelPersistenceError
)
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.state_manager import (
    StateManager,
    StateRepository,
    StateSnapshot,
    StateTransition,
    StateOperationType,
    StateManagerError
)

logger = logging.getLogger(__name__)


class KalmanIntegrationError(Exception):
    """Base exception for Kalman integration operations."""
    pass


class FilterStateConverter:
    """Utility class for converting between FilterState and PlayerState formats."""
    
    @staticmethod
    def filter_to_player_state(filter_state: FilterState) -> PlayerState:
        """Convert FilterState to PlayerState for compatibility."""
        return PlayerState(
            player_id=filter_state.player_id,
            state_mean=filter_state.state_mean,
            state_cov=filter_state.state_cov,
            gameweek=filter_state.gameweek,
            season=filter_state.season,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    @staticmethod
    def player_to_filter_state(player_state: PlayerState) -> FilterState:
        """Convert PlayerState to FilterState for Kalman operations."""
        return FilterState(
            state_mean=player_state.state_mean,
            state_cov=player_state.state_cov,
            timestamp=float(player_state.gameweek),
            gameweek=player_state.gameweek,
            season=player_state.season,
            player_id=player_state.player_id
        )
    
    @staticmethod
    def serialize_filter_state(filter_state: FilterState) -> Dict[str, Any]:
        """Serialize FilterState to dictionary for storage."""
        return {
            "state_mean": filter_state.state_mean.tolist(),
            "state_cov": filter_state.state_cov.tolist(),
            "timestamp": filter_state.timestamp,
            "gameweek": filter_state.gameweek,
            "season": filter_state.season,
            "player_id": filter_state.player_id,
            "innovation": filter_state.innovation.tolist() if filter_state.innovation is not None else None,
            "innovation_cov": filter_state.innovation_cov.tolist() if filter_state.innovation_cov is not None else None,
            "kalman_gain": filter_state.kalman_gain.tolist() if filter_state.kalman_gain is not None else None,
            "log_likelihood": filter_state.log_likelihood,
            "condition_number": filter_state.condition_number,
            "trace_ratio": filter_state.trace_ratio
        }
    
    @staticmethod
    def deserialize_filter_state(data: Dict[str, Any]) -> FilterState:
        """Deserialize FilterState from dictionary."""
        return FilterState(
            state_mean=jnp.array(data["state_mean"]),
            state_cov=jnp.array(data["state_cov"]),
            timestamp=data["timestamp"],
            gameweek=data["gameweek"],
            season=data["season"],
            player_id=data["player_id"],
            innovation=jnp.array(data["innovation"]) if data.get("innovation") else None,
            innovation_cov=jnp.array(data["innovation_cov"]) if data.get("innovation_cov") else None,
            kalman_gain=jnp.array(data["kalman_gain"]) if data.get("kalman_gain") else None,
            log_likelihood=data.get("log_likelihood"),
            condition_number=data.get("condition_number"),
            trace_ratio=data.get("trace_ratio")
        )


class KalmanStateRepository(StateRepository):
    """
    StateRepository implementation specifically for Kalman filter states.
    
    Provides efficient storage and retrieval of FilterState objects with
    compression and serialization optimized for Kalman filter data.
    """
    
    def __init__(
        self,
        redis_cache: RedisCache,
        key_prefix: str = "kalman_state",
        ttl: Optional[int] = None,
        compression_enabled: bool = True
    ):
        """
        Initialize Kalman state repository.
        
        Args:
            redis_cache: Redis cache instance
            key_prefix: Prefix for Kalman state keys
            ttl: Time-to-live for state entries
            compression_enabled: Whether to compress state data
        """
        self.redis_cache = redis_cache
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.compression_enabled = compression_enabled
        self.converter = FilterStateConverter()
    
    def _get_filter_state_key(self, player_id: int, filter_name: str = "main") -> str:
        """Generate Redis key for filter state."""
        return f"{self.key_prefix}:filter:{filter_name}:player:{player_id}"
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if self.compression_enabled:
            import zlib
            return zlib.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression was used."""
        if self.compression_enabled:
            import zlib
            try:
                return zlib.decompress(data)
            except zlib.error:
                # Data might not be compressed (backward compatibility)
                return data
        return data
    
    async def get_state(self, player_id: int) -> Optional[PlayerState]:
        """Get current state for a player (converted to PlayerState)."""
        filter_state = await self.get_filter_state(player_id)
        if filter_state is None:
            return None
        return self.converter.filter_to_player_state(filter_state)
    
    async def set_state(self, player_id: int, state: PlayerState) -> bool:
        """Set current state for a player (converted from PlayerState)."""
        filter_state = self.converter.player_to_filter_state(state)
        return await self.set_filter_state(player_id, filter_state)
    
    async def get_filter_state(
        self, 
        player_id: int, 
        filter_name: str = "main"
    ) -> Optional[FilterState]:
        """Get FilterState for a specific player and filter."""
        key = self._get_filter_state_key(player_id, filter_name)
        compressed_data = self.redis_cache.get(key)
        
        if compressed_data is None:
            return None
        
        try:
            # Decompress and deserialize
            data = self._decompress_data(compressed_data)
            state_dict = pickle.loads(data)
            return self.converter.deserialize_filter_state(state_dict)
        except Exception as e:
            logger.error(f"Error deserializing filter state for player {player_id}: {e}")
            return None
    
    async def set_filter_state(
        self, 
        player_id: int, 
        filter_state: FilterState,
        filter_name: str = "main"
    ) -> bool:
        """Set FilterState for a specific player and filter."""
        try:
            # Serialize and compress
            state_dict = self.converter.serialize_filter_state(filter_state)
            data = pickle.dumps(state_dict)
            compressed_data = self._compress_data(data)
            
            key = self._get_filter_state_key(player_id, filter_name)
            return self.redis_cache.set(key, compressed_data, self.ttl)
        except Exception as e:
            logger.error(f"Error storing filter state for player {player_id}: {e}")
            return False
    
    async def get_multiple_filter_states(
        self, 
        player_ids: List[int],
        filter_name: str = "main"
    ) -> Dict[int, Optional[FilterState]]:
        """Get FilterStates for multiple players efficiently."""
        if not player_ids:
            return {}
        
        keys = [self._get_filter_state_key(player_id, filter_name) for player_id in player_ids]
        raw_data_list = self.redis_cache.mget(keys)
        
        result = {}
        for i, player_id in enumerate(player_ids):
            if i < len(raw_data_list) and raw_data_list[i] is not None:
                try:
                    data = self._decompress_data(raw_data_list[i])
                    state_dict = pickle.loads(data)
                    result[player_id] = self.converter.deserialize_filter_state(state_dict)
                except Exception as e:
                    logger.error(f"Error deserializing filter state for player {player_id}: {e}")
                    result[player_id] = None
            else:
                result[player_id] = None
        
        return result
    
    async def set_multiple_filter_states(
        self, 
        states: Dict[int, FilterState],
        filter_name: str = "main"
    ) -> Dict[int, bool]:
        """Set FilterStates for multiple players efficiently."""
        if not states:
            return {}
        
        # Prepare data for batch set
        key_value_pairs = {}
        for player_id, filter_state in states.items():
            try:
                state_dict = self.converter.serialize_filter_state(filter_state)
                data = pickle.dumps(state_dict)
                compressed_data = self._compress_data(data)
                
                key = self._get_filter_state_key(player_id, filter_name)
                key_value_pairs[key] = compressed_data
            except Exception as e:
                logger.error(f"Error serializing filter state for player {player_id}: {e}")
                continue
        
        # Use batch set operation
        success = self.redis_cache.mset(key_value_pairs, self.ttl)
        
        # Return individual results
        return {player_id: success for player_id in states.keys()}
    
    async def delete_state(self, player_id: int) -> bool:
        """Delete state for a player (all filters)."""
        # Delete main filter state
        key = self._get_filter_state_key(player_id, "main")
        success = self.redis_cache.delete(key)
        
        # Delete any other filter states for this player
        pattern = f"{self.key_prefix}:filter:*:player:{player_id}"
        deleted_count = self.redis_cache.delete_pattern(pattern)
        
        return success or deleted_count > 0
    
    async def get_multiple_states(self, player_ids: List[int]) -> Dict[int, Optional[PlayerState]]:
        """Get states for multiple players efficiently (converted to PlayerState)."""
        filter_states = await self.get_multiple_filter_states(player_ids)
        
        result = {}
        for player_id, filter_state in filter_states.items():
            if filter_state is not None:
                result[player_id] = self.converter.filter_to_player_state(filter_state)
            else:
                result[player_id] = None
        
        return result
    
    async def set_multiple_states(self, states: Dict[int, PlayerState]) -> Dict[int, bool]:
        """Set states for multiple players efficiently (converted from PlayerState)."""
        filter_states = {}
        for player_id, player_state in states.items():
            filter_states[player_id] = self.converter.player_to_filter_state(player_state)
        
        return await self.set_multiple_filter_states(filter_states)
    
    async def list_player_ids(self) -> List[int]:
        """List all player IDs with stored states."""
        pattern = f"{self.key_prefix}:filter:*:player:*"
        
        if not self.redis_cache.is_available():
            return []
        
        try:
            with self.redis_cache.connection_manager.get_connection() as conn:
                keys = conn.keys(pattern)
                
                player_ids = set()
                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    # Extract player ID from key: kalman_state:filter:main:player:123
                    parts = key_str.split(':')
                    if len(parts) >= 5 and parts[-2] == 'player':
                        try:
                            player_id = int(parts[-1])
                            player_ids.add(player_id)
                        except ValueError:
                            continue
                
                return sorted(list(player_ids))
                
        except Exception as e:
            logger.error(f"Error listing player IDs from Redis: {e}")
            return []
    
    async def clear_all_states(self) -> bool:
        """Clear all stored states."""
        pattern = f"{self.key_prefix}:filter:*:player:*"
        deleted_count = self.redis_cache.delete_pattern(pattern)
        return deleted_count > 0
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage backend."""
        info = {
            "type": "kalman_redis",
            "available": self.redis_cache.is_available(),
            "key_prefix": self.key_prefix,
            "ttl": self.ttl,
            "compression_enabled": self.compression_enabled
        }
        
        if self.redis_cache.is_available():
            info.update(self.redis_cache.get_metrics())
            
            # Get approximate count of filter state keys
            try:
                player_ids = await self.list_player_ids()
                info["player_count"] = len(player_ids)
            except Exception as e:
                logger.error(f"Error getting player count: {e}")
                info["player_count"] = "unknown"
        
        return info


class KalmanStateManager(StateManager):
    """
    StateManager extension specifically for Kalman filter integration.
    
    Provides seamless integration between Kalman filter operations and
    the state management system.
    """
    
    def __init__(
        self,
        repository: KalmanStateRepository,
        kalman_model: KalmanPlayerModel,
        **kwargs
    ):
        """
        Initialize Kalman state manager.
        
        Args:
            repository: Kalman state repository
            kalman_model: Kalman player model instance
            **kwargs: Additional StateManager arguments
        """
        super().__init__(repository, **kwargs)
        self.kalman_model = kalman_model
        self.kalman_repository = repository  # Type hint for IDE
    
    async def predict_and_store_state(
        self,
        player_id: int,
        gameweeks_ahead: int = 1,
        position: Optional[str] = None,
        create_snapshot: bool = True
    ) -> bool:
        """
        Predict player state using Kalman filter and store result.
        
        Args:
            player_id: Player ID to predict for
            gameweeks_ahead: Number of gameweeks to predict ahead
            position: Player position for position-specific models
            create_snapshot: Whether to create a snapshot before prediction
            
        Returns:
            True if prediction and storage successful
        """
        try:
            # Get current player state
            current_player_state = await self.get_player_state(player_id)
            if current_player_state is None:
                logger.warning(f"No current state for player {player_id}, cannot predict")
                return False
            
            # Create snapshot before prediction if requested
            if create_snapshot and self.history:
                snapshot = StateSnapshot(
                    snapshot_id=f"pre_prediction_{player_id}_{time.time()}",
                    player_id=player_id,
                    state=current_player_state,
                    timestamp=datetime.now(timezone.utc),
                    metadata={"operation": "kalman_prediction", "gameweeks_ahead": gameweeks_ahead}
                )
                self.history.add_snapshot(snapshot)
            
            # Predict using Kalman model
            predicted_state = self.kalman_model.predict_state(
                current_player_state,
                gameweeks_ahead=gameweeks_ahead,
                position=position
            )
            
            # Store predicted state
            success = await self.update_player_state(
                player_id,
                predicted_state,
                metadata={
                    "operation": "kalman_prediction",
                    "gameweeks_ahead": gameweeks_ahead,
                    "position": position
                },
                create_snapshot=False  # Already created above if needed
            )
            
            if success:
                logger.debug(f"Successfully predicted and stored state for player {player_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to predict and store state for player {player_id}: {e}")
            return False
    
    async def update_with_observation(
        self,
        player_id: int,
        observation: jnp.ndarray,
        observation_noise: Optional[jnp.ndarray] = None,
        position: Optional[str] = None,
        create_snapshot: bool = True
    ) -> bool:
        """
        Update player state with new observation using Kalman filter.
        
        Args:
            player_id: Player ID to update
            observation: New observation vector
            observation_noise: Optional observation noise covariance
            position: Player position for position-specific models
            create_snapshot: Whether to create a snapshot before update
            
        Returns:
            True if update successful
        """
        try:
            # Get current state (should be predicted state)
            current_player_state = await self.get_player_state(player_id)
            if current_player_state is None:
                logger.warning(f"No current state for player {player_id}, cannot update")
                return False
            
            # Create snapshot before update if requested
            if create_snapshot and self.history:
                snapshot = StateSnapshot(
                    snapshot_id=f"pre_update_{player_id}_{time.time()}",
                    player_id=player_id,
                    state=current_player_state,
                    timestamp=datetime.now(timezone.utc),
                    metadata={"operation": "kalman_update", "observation": observation.tolist()}
                )
                self.history.add_snapshot(snapshot)
            
            # Update using Kalman model
            updated_state = self.kalman_model.update_state(
                current_player_state,
                observation,
                observation_noise=observation_noise,
                position=position
            )
            
            # Store updated state
            success = await self.update_player_state(
                player_id,
                updated_state,
                metadata={
                    "operation": "kalman_update",
                    "observation": observation.tolist(),
                    "position": position
                },
                create_snapshot=False  # Already created above if needed
            )
            
            if success:
                logger.debug(f"Successfully updated state for player {player_id} with observation")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update state for player {player_id}: {e}")
            return False
    
    async def batch_predict_and_store(
        self,
        player_ids: List[int],
        gameweeks_ahead: int = 1,
        positions: Optional[Dict[int, str]] = None
    ) -> Dict[int, bool]:
        """
        Batch predict and store states for multiple players.
        
        Args:
            player_ids: List of player IDs to predict for
            gameweeks_ahead: Number of gameweeks to predict ahead
            positions: Optional mapping of player IDs to positions
            
        Returns:
            Dictionary mapping player IDs to success status
        """
        results = {}
        
        # Get current states for all players
        current_states = await self.get_multiple_states(player_ids)
        
        # Process each player
        for player_id in player_ids:
            try:
                current_state = current_states.get(player_id)
                if current_state is None:
                    results[player_id] = False
                    continue
                
                position = positions.get(player_id) if positions else None
                
                # Predict
                predicted_state = self.kalman_model.predict_state(
                    current_state,
                    gameweeks_ahead=gameweeks_ahead,
                    position=position
                )
                
                # Store (will be done in batch later)
                results[player_id] = predicted_state
                
            except Exception as e:
                logger.error(f"Failed to predict state for player {player_id}: {e}")
                results[player_id] = False
        
        # Batch store all successful predictions
        states_to_store = {pid: state for pid, state in results.items() 
                          if isinstance(state, PlayerState)}
        
        if states_to_store:
            store_results = await self.repository.set_multiple_states(states_to_store)
            
            # Update results with storage status
            for player_id in states_to_store:
                results[player_id] = store_results.get(player_id, False)
        
        return results
    
    async def get_kalman_diagnostics(self, player_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Get Kalman-specific diagnostics for players."""
        if player_ids is None:
            player_ids = await self.repository.list_player_ids()
        
        diagnostics = {
            "model_diagnostics": self.kalman_model.get_model_diagnostics(),
            "state_diagnostics": {},
            "filter_states": {}
        }
        
        # Get filter states for diagnostics
        for player_id in player_ids[:10]:  # Limit to first 10 for performance
            try:
                filter_state = await self.kalman_repository.get_filter_state(player_id)
                if filter_state is not None:
                    diagnostics["filter_states"][player_id] = {
                        "condition_number": filter_state.condition_number,
                        "trace_ratio": filter_state.trace_ratio,
                        "log_likelihood": filter_state.log_likelihood,
                        "state_norm": float(jnp.linalg.norm(filter_state.state_mean)),
                        "cov_trace": float(jnp.trace(filter_state.state_cov))
                    }
            except Exception as e:
                logger.warning(f"Failed to get diagnostics for player {player_id}: {e}")
        
        return diagnostics


class KalmanModelPersistence(ModelPersistence):
    """
    ModelPersistence extension for Kalman player models.
    
    Provides specialized checkpoint functionality for KalmanPlayerModel
    with efficient serialization of filter states.
    """
    
    def __init__(
        self,
        kalman_model: KalmanPlayerModel,
        config: Optional[PersistenceConfig] = None,
        **kwargs
    ):
        """
        Initialize Kalman model persistence.
        
        Args:
            kalman_model: Kalman player model to persist
            config: Persistence configuration
            **kwargs: Additional ModelPersistence arguments
        """
        super().__init__(config, **kwargs)
        self.kalman_model = kalman_model
        self.converter = FilterStateConverter()
    
    async def save_kalman_checkpoint(
        self,
        version_id: int,
        checkpoint_name: str,
        description: Optional[str] = None,
        gameweek: Optional[int] = None,
        season: Optional[str] = None,
        include_filter_states: bool = True,
        **kwargs
    ) -> int:
        """
        Save a Kalman model checkpoint with filter states.
        
        Args:
            version_id: Model version ID
            checkpoint_name: Name for the checkpoint
            description: Optional description
            gameweek: Current gameweek
            season: Current season
            include_filter_states: Whether to include individual filter states
            **kwargs: Additional save_checkpoint arguments
            
        Returns:
            Checkpoint ID
        """
        try:
            # Get model diagnostics for metadata
            diagnostics = self.kalman_model.get_model_diagnostics()
            
            # Add Kalman-specific description
            kalman_description = f"Kalman Model ({self.kalman_model.filter_type})"
            if description:
                kalman_description = f"{description} - {kalman_description}"
            
            # Add Kalman-specific tags
            tags = kwargs.get("tags", [])
            tags.extend([
                f"filter_type:{self.kalman_model.filter_type}",
                f"model_mode:{self.kalman_model.model_mode}",
                f"players:{len(self.kalman_model.player_states)}"
            ])
            kwargs["tags"] = tags
            
            # Save main model checkpoint
            checkpoint_id = await self.save_checkpoint(
                model=self.kalman_model,
                version_id=version_id,
                checkpoint_name=checkpoint_name,
                description=kalman_description,
                gameweek=gameweek,
                season=season,
                validation_score=diagnostics.get("model_metrics", {}).get("average_log_likelihood"),
                **kwargs
            )
            
            # Save individual filter states if requested
            if include_filter_states:
                await self._save_kalman_filter_states(checkpoint_id)
            
            logger.info(f"Saved Kalman checkpoint '{checkpoint_name}' with ID {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save Kalman checkpoint: {e}")
            raise ModelPersistenceError(f"Kalman checkpoint save failed: {e}")
    
    async def load_kalman_checkpoint(self, checkpoint_id: int) -> KalmanPlayerModel:
        """
        Load a Kalman model checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Restored KalmanPlayerModel instance
        """
        try:
            # Load base model
            model = await self.load_checkpoint(checkpoint_id)
            
            if not isinstance(model, KalmanPlayerModel):
                raise ModelPersistenceError(f"Checkpoint {checkpoint_id} is not a KalmanPlayerModel")
            
            # Load individual filter states
            await self._load_kalman_filter_states(checkpoint_id, model)
            
            logger.info(f"Loaded Kalman checkpoint {checkpoint_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Kalman checkpoint {checkpoint_id}: {e}")
            raise ModelPersistenceError(f"Kalman checkpoint load failed: {e}")
    
    async def _save_kalman_filter_states(self, checkpoint_id: int) -> None:
        """Save individual Kalman filter states."""
        try:
            # Serialize all filter states
            filter_states_data = {}
            
            for player_id, filter_states in self.kalman_model.filter_states.items():
                player_data = {}
                for filter_name, filter_state in filter_states.items():
                    player_data[filter_name] = self.converter.serialize_filter_state(filter_state)
                filter_states_data[str(player_id)] = player_data
            
            # Save as JSON metadata (could be extended to use dedicated table)
            metadata = {
                "filter_states": filter_states_data,
                "filter_type": self.kalman_model.filter_type,
                "model_mode": self.kalman_model.model_mode,
                "player_count": len(filter_states_data)
            }
            
            # Store in checkpoint metadata or separate storage
            # For now, we'll use the existing ModelState table with special encoding
            await self._store_filter_states_metadata(checkpoint_id, metadata)
            
        except Exception as e:
            logger.warning(f"Failed to save Kalman filter states: {e}")
    
    async def _load_kalman_filter_states(self, checkpoint_id: int, model: KalmanPlayerModel) -> None:
        """Load individual Kalman filter states."""
        try:
            # Load filter states metadata
            metadata = await self._load_filter_states_metadata(checkpoint_id)
            
            if metadata and "filter_states" in metadata:
                filter_states_data = metadata["filter_states"]
                
                # Restore filter states
                for player_id_str, player_data in filter_states_data.items():
                    player_id = int(player_id_str)
                    
                    if player_id not in model.filter_states:
                        model.filter_states[player_id] = {}
                    
                    for filter_name, filter_state_data in player_data.items():
                        filter_state = self.converter.deserialize_filter_state(filter_state_data)
                        model.filter_states[player_id][filter_name] = filter_state
                
                logger.debug(f"Restored filter states for {len(filter_states_data)} players")
            
        except Exception as e:
            logger.warning(f"Failed to load Kalman filter states: {e}")
    
    async def _store_filter_states_metadata(self, checkpoint_id: int, metadata: Dict[str, Any]) -> None:
        """Store filter states metadata."""
        # This is a simplified implementation - in production you might want
        # a dedicated table for filter states or use the existing ModelState table
        
        # For now, store as compressed JSON in a special ModelState entry
        try:
            import gzip
            import json
            
            compressed_data = gzip.compress(json.dumps(metadata).encode('utf-8'))
            
            # Store with special player_id = -1 to indicate metadata
            with session_scope() as session:
                from airsenal.framework.schema import ModelState
                
                metadata_state = ModelState(
                    checkpoint_id=checkpoint_id,
                    player_id=-1,  # Special ID for metadata
                    state_mean=compressed_data,  # Store in state_mean field
                    state_covariance=b"",  # Empty
                    state_dimension=len(metadata.get("filter_states", {})),
                    gameweek=0,
                    season="metadata",
                    last_updated=datetime.now(timezone.utc).isoformat(),
                    serialization_format="kalman_filter_states",
                    compression_used=True,
                    is_valid=True
                )
                
                session.add(metadata_state)
                session.flush()
                
        except Exception as e:
            logger.error(f"Failed to store filter states metadata: {e}")
    
    async def _load_filter_states_metadata(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Load filter states metadata."""
        try:
            import gzip
            import json
            
            with session_scope() as session:
                from airsenal.framework.schema import ModelState
                from sqlalchemy import select
                
                # Look for metadata entry
                metadata_state = session.execute(
                    select(ModelState).where(
                        ModelState.checkpoint_id == checkpoint_id,
                        ModelState.player_id == -1,  # Special metadata ID
                        ModelState.serialization_format == "kalman_filter_states"
                    )
                ).scalar_one_or_none()
                
                if metadata_state and metadata_state.state_mean:
                    # Decompress and load JSON
                    decompressed_data = gzip.decompress(metadata_state.state_mean)
                    metadata = json.loads(decompressed_data.decode('utf-8'))
                    return metadata
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load filter states metadata: {e}")
            return None


# Utility functions for integration

async def create_integrated_kalman_system(
    redis_cache: RedisCache,
    config: Optional[StateSpaceConfig] = None,
    filter_type: str = "extended",
    enable_persistence: bool = True
) -> Tuple[KalmanPlayerModel, KalmanStateManager, Optional[KalmanModelPersistence]]:
    """
    Create a fully integrated Kalman system with state management and persistence.
    
    Args:
        redis_cache: Redis cache for state storage
        config: State space configuration
        filter_type: Type of Kalman filter to use
        enable_persistence: Whether to enable model persistence
        
    Returns:
        Tuple of (model, state_manager, persistence)
    """
    # Create Kalman player model
    if config is None:
        config = StateSpaceConfig()
    
    kalman_model = KalmanPlayerModel(config=config, filter_type=filter_type)
    
    # Create Kalman state repository
    kalman_repository = KalmanStateRepository(redis_cache)
    
    # Create state manager
    state_manager = KalmanStateManager(
        repository=kalman_repository,
        kalman_model=kalman_model,
        enable_history=True,
        enable_validation=True
    )
    
    # Create persistence if requested
    persistence = None
    if enable_persistence:
        persistence = KalmanModelPersistence(kalman_model)
    
    return kalman_model, state_manager, persistence


def convert_player_states_to_filter_states(
    player_states: Dict[int, PlayerState]
) -> Dict[int, FilterState]:
    """Convert dictionary of PlayerStates to FilterStates."""
    converter = FilterStateConverter()
    return {
        player_id: converter.player_to_filter_state(player_state)
        for player_id, player_state in player_states.items()
    }


def convert_filter_states_to_player_states(
    filter_states: Dict[int, FilterState]
) -> Dict[int, PlayerState]:
    """Convert dictionary of FilterStates to PlayerStates."""
    converter = FilterStateConverter()
    return {
        player_id: converter.filter_to_player_state(filter_state)
        for player_id, filter_state in filter_states.items()
    }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of using the integrated Kalman system."""
        # Mock Redis cache (replace with real one)
        redis_cache = Mock()
        
        # Create integrated system
        model, state_manager, persistence = await create_integrated_kalman_system(
            redis_cache=redis_cache,
            filter_type="extended",
            enable_persistence=True
        )
        
        # Initialize a player state
        player_state = model.initialize_state(123, gameweek=1, season="2023")
        
        # Store state
        await state_manager.update_player_state(123, player_state)
        
        # Predict and store
        await state_manager.predict_and_store_state(123, gameweeks_ahead=1)
        
        # Update with observation
        observation = jnp.array([1.0, 0.5, 90.0, 2.0])
        await state_manager.update_with_observation(123, observation)
        
        # Save checkpoint
        if persistence:
            checkpoint_id = await persistence.save_kalman_checkpoint(
                version_id=1,
                checkpoint_name="example_checkpoint",
                gameweek=1,
                season="2023"
            )
            print(f"Saved checkpoint with ID: {checkpoint_id}")
    
    # Run example
    # asyncio.run(example_usage())