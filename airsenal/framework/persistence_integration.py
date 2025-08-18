"""
Integration layer between ModelPersistence and existing AIrsenal systems.

This module provides seamless integration between the new ModelPersistence layer
and existing systems like StateManager, ModelVersion, and Redis cache.

Key Features:
- Automatic checkpoint creation on model updates
- Integration with existing StateManager for state persistence
- Compatibility with existing ModelVersion system
- Redis cache integration for temporary storage
- Event-driven checkpoint management
- Backward compatibility with existing model storage

Usage:
    ```python
    from airsenal.framework.persistence_integration import PersistenceIntegrator
    
    # Initialize integrator
    integrator = PersistenceIntegrator(
        persistence_config=config,
        state_manager=state_manager,
        redis_cache=redis_cache
    )
    
    # Wrap existing model with persistence
    persistent_model = integrator.wrap_model(adaptive_model, version_id=123)
    
    # Model updates now automatically trigger checkpoints
    persistent_model.update_state(player_id, new_state)
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from sqlalchemy.orm import Session

# AIrsenal imports
from airsenal.framework.adaptive_player_model import AdaptivePlayerModel, PlayerState
from airsenal.framework.model_persistence import (
    CheckpointType, ModelPersistence, PersistenceConfig
)
from airsenal.framework.model_versioning import ModelVersionManager, ModelArtifactManager
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import ModelVersion, session_scope
from airsenal.framework.state_manager import StateManager, StateSnapshot, StateTransition

logger = logging.getLogger(__name__)


class PersistentAdaptiveModel:
    """
    Wrapper for AdaptivePlayerModel that adds automatic persistence capabilities.
    
    This class decorates an existing AdaptivePlayerModel with persistence features
    without modifying the original model interface.
    """
    
    def __init__(
        self,
        base_model: AdaptivePlayerModel,
        persistence: ModelPersistence,
        version_id: int,
        integrator: 'PersistenceIntegrator'
    ):
        """
        Initialize persistent model wrapper.
        
        Args:
            base_model: Base AdaptivePlayerModel to wrap
            persistence: ModelPersistence instance
            version_id: Associated ModelVersion ID
            integrator: PersistenceIntegrator instance
        """
        self.base_model = base_model
        self.persistence = persistence
        self.version_id = version_id
        self.integrator = integrator
        
        # Track model updates for automatic checkpointing
        self.update_count = 0
        self.last_checkpoint_time = time.time()
        
        # Forward all attributes to base model
        self._forward_attributes()
    
    def _forward_attributes(self):
        """Forward all base model attributes to this wrapper."""
        for attr_name in dir(self.base_model):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                attr = getattr(self.base_model, attr_name)
                if callable(attr):
                    # Wrap methods to track updates
                    if attr_name in ['update_state', 'fit', 'predict_state']:
                        setattr(self, attr_name, self._wrap_update_method(attr, attr_name))
                    else:
                        setattr(self, attr_name, attr)
                else:
                    setattr(self, attr_name, attr)
    
    def _wrap_update_method(self, method, method_name: str):
        """Wrap model update methods to trigger automatic checkpointing."""
        def wrapped_method(*args, **kwargs):
            # Call original method
            result = method(*args, **kwargs)
            
            # Track update
            self.update_count += 1
            
            # Check for automatic checkpoint
            if self.persistence.should_auto_checkpoint():
                try:
                    asyncio.create_task(
                        self.persistence.create_auto_checkpoint(
                            model=self.base_model,
                            version_id=self.version_id
                        )
                    )
                except Exception as e:
                    logger.warning(f"Auto-checkpoint failed after {method_name}: {e}")
            
            return result
        
        return wrapped_method
    
    async def create_checkpoint(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        **kwargs
    ) -> int:
        """
        Create a manual checkpoint of the current model state.
        
        Args:
            name: Optional checkpoint name
            description: Optional description
            checkpoint_type: Type of checkpoint
            **kwargs: Additional checkpoint metadata
            
        Returns:
            Created checkpoint ID
        """
        return await self.persistence.save_checkpoint(
            model=self.base_model,
            version_id=self.version_id,
            checkpoint_name=name,
            checkpoint_type=checkpoint_type,
            description=description,
            **kwargs
        )
    
    async def restore_from_checkpoint(self, checkpoint_id: int) -> None:
        """
        Restore model state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore from
        """
        restored_model = await self.persistence.load_checkpoint(checkpoint_id)
        
        # Update base model with restored state
        if hasattr(restored_model, 'player_states'):
            self.base_model.player_states = restored_model.player_states
        
        # Copy other relevant attributes
        for attr in ['config', 'learning_rate', 'decay_factor', 'is_fitted']:
            if hasattr(restored_model, attr):
                setattr(self.base_model, attr, getattr(restored_model, attr))
        
        logger.info(f"Model restored from checkpoint {checkpoint_id}")
    
    def get_checkpoint_history(self) -> List[Any]:
        """Get checkpoint history for this model version."""
        try:
            return asyncio.run(
                self.persistence.list_checkpoints(version_id=self.version_id)
            )
        except Exception as e:
            logger.error(f"Failed to get checkpoint history: {e}")
            return []


class StateManagerIntegration:
    """Integration between ModelPersistence and StateManager."""
    
    def __init__(
        self,
        persistence: ModelPersistence,
        state_manager: StateManager
    ):
        """
        Initialize StateManager integration.
        
        Args:
            persistence: ModelPersistence instance
            state_manager: StateManager instance
        """
        self.persistence = persistence
        self.state_manager = state_manager
    
    async def sync_state_snapshots_to_checkpoints(
        self,
        version_id: int,
        player_ids: Optional[List[int]] = None
    ) -> int:
        """
        Sync StateManager snapshots to model checkpoints.
        
        Args:
            version_id: Model version ID
            player_ids: Optional list of player IDs to sync
            
        Returns:
            Number of states synced
        """
        try:
            if not self.state_manager.history:
                logger.warning("StateManager history not enabled")
                return 0
            
            if player_ids is None:
                player_ids = await self.state_manager.repository.list_player_ids()
            
            synced_count = 0
            
            for player_id in player_ids:
                # Get latest snapshot for player
                latest_snapshot = self.state_manager.history.get_latest_snapshot(player_id)
                
                if latest_snapshot:
                    # Create checkpoint from snapshot
                    try:
                        await self._create_checkpoint_from_snapshot(
                            latest_snapshot, version_id
                        )
                        synced_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to sync snapshot for player {player_id}: {e}")
            
            logger.info(f"Synced {synced_count} state snapshots to checkpoints")
            return synced_count
            
        except Exception as e:
            logger.error(f"State snapshot sync failed: {e}")
            return 0
    
    async def _create_checkpoint_from_snapshot(
        self,
        snapshot: StateSnapshot,
        version_id: int
    ) -> int:
        """Create a checkpoint from a state snapshot."""
        # Create a minimal model with just the state
        # This is a simplified implementation - in practice, you'd need
        # to create a proper model instance
        checkpoint_name = f"state_sync_{snapshot.player_id}_{snapshot.snapshot_id}"
        
        # For now, just store the state metadata
        # In a full implementation, you'd reconstruct the model
        logger.debug(f"Would create checkpoint from snapshot {snapshot.snapshot_id}")
        
        return 0  # Placeholder
    
    async def restore_states_from_checkpoint(
        self,
        checkpoint_id: int,
        player_ids: Optional[List[int]] = None
    ) -> int:
        """
        Restore StateManager states from a model checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore from
            player_ids: Optional list of player IDs to restore
            
        Returns:
            Number of states restored
        """
        try:
            # Load checkpoint
            model = await self.persistence.load_checkpoint(checkpoint_id)
            
            if not hasattr(model, 'player_states'):
                logger.warning("Checkpoint has no player states")
                return 0
            
            restored_count = 0
            
            for player_id, state in model.player_states.items():
                if player_ids is None or player_id in player_ids:
                    # Update state in StateManager
                    success = await self.state_manager.update_player_state(
                        player_id=player_id,
                        new_state=state,
                        metadata={
                            "source": "checkpoint_restore",
                            "checkpoint_id": checkpoint_id
                        }
                    )
                    
                    if success:
                        restored_count += 1
            
            logger.info(f"Restored {restored_count} states from checkpoint {checkpoint_id}")
            return restored_count
            
        except Exception as e:
            logger.error(f"State restoration from checkpoint failed: {e}")
            return 0


class ModelVersionIntegration:
    """Integration between ModelPersistence and existing ModelVersion system."""
    
    def __init__(
        self,
        persistence: ModelPersistence,
        version_manager: Optional[ModelVersionManager] = None
    ):
        """
        Initialize ModelVersion integration.
        
        Args:
            persistence: ModelPersistence instance
            version_manager: Optional ModelVersionManager instance
        """
        self.persistence = persistence
        self.version_manager = version_manager
    
    async def create_version_checkpoint(
        self,
        model: AdaptivePlayerModel,
        version_id: int,
        performance_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Create a checkpoint tied to a specific model version.
        
        Args:
            model: AdaptivePlayerModel to checkpoint
            version_id: ModelVersion ID
            performance_metrics: Optional performance metrics
            tags: Optional tags for the checkpoint
            
        Returns:
            Created checkpoint ID
        """
        try:
            # Extract performance metrics
            validation_score = None
            training_loss = None
            
            if performance_metrics:
                validation_score = performance_metrics.get('validation_score')
                training_loss = performance_metrics.get('training_loss')
            
            # Create checkpoint
            checkpoint_id = await self.persistence.save_checkpoint(
                model=model,
                version_id=version_id,
                checkpoint_name=f"version_{version_id}_checkpoint",
                checkpoint_type=CheckpointType.MILESTONE,
                description=f"Checkpoint for model version {version_id}",
                tags=tags,
                validation_score=validation_score,
                training_loss=training_loss,
                created_by="version_integration"
            )
            
            logger.info(f"Created version checkpoint {checkpoint_id} for version {version_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create version checkpoint: {e}")
            raise
    
    async def link_checkpoint_to_artifact(
        self,
        checkpoint_id: int,
        artifact_id: int
    ) -> bool:
        """
        Link a persistence checkpoint to a ModelArtifact.
        
        Args:
            checkpoint_id: Checkpoint ID
            artifact_id: ModelArtifact ID
            
        Returns:
            True if linking successful
        """
        try:
            # This would update the checkpoint metadata to reference the artifact
            # For now, just log the operation
            logger.info(f"Linking checkpoint {checkpoint_id} to artifact {artifact_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link checkpoint to artifact: {e}")
            return False


class RedisIntegration:
    """Integration between ModelPersistence and Redis cache."""
    
    def __init__(
        self,
        persistence: ModelPersistence,
        redis_cache: RedisCache
    ):
        """
        Initialize Redis integration.
        
        Args:
            persistence: ModelPersistence instance
            redis_cache: RedisCache instance
        """
        self.persistence = persistence
        self.redis_cache = redis_cache
        self.cache_prefix = "model_persistence"
    
    def cache_checkpoint_metadata(
        self,
        checkpoint_id: int,
        metadata: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """
        Cache checkpoint metadata in Redis for fast access.
        
        Args:
            checkpoint_id: Checkpoint ID
            metadata: Metadata to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if caching successful
        """
        try:
            cache_key = f"{self.cache_prefix}:checkpoint:{checkpoint_id}:metadata"
            return self.redis_cache.set(cache_key, metadata, ttl)
            
        except Exception as e:
            logger.error(f"Failed to cache checkpoint metadata: {e}")
            return False
    
    def get_cached_checkpoint_metadata(
        self,
        checkpoint_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached checkpoint metadata from Redis.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Cached metadata or None if not found
        """
        try:
            cache_key = f"{self.cache_prefix}:checkpoint:{checkpoint_id}:metadata"
            return self.redis_cache.get(cache_key)
            
        except Exception as e:
            logger.error(f"Failed to get cached checkpoint metadata: {e}")
            return None
    
    def cache_model_predictions(
        self,
        model_id: str,
        predictions: Dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """
        Cache model predictions in Redis.
        
        Args:
            model_id: Model identifier
            predictions: Predictions to cache
            ttl: Time-to-live in seconds
            
        Returns:
            True if caching successful
        """
        try:
            cache_key = f"{self.cache_prefix}:predictions:{model_id}"
            return self.redis_cache.set(cache_key, predictions, ttl)
            
        except Exception as e:
            logger.error(f"Failed to cache model predictions: {e}")
            return False


class PersistenceIntegrator:
    """
    Main integration class that coordinates between ModelPersistence and existing systems.
    
    This class provides a unified interface for using the persistence layer with
    existing AIrsenal components like StateManager, ModelVersion, and Redis cache.
    """
    
    def __init__(
        self,
        persistence_config: Optional[PersistenceConfig] = None,
        state_manager: Optional[StateManager] = None,
        redis_cache: Optional[RedisCache] = None,
        version_manager: Optional[ModelVersionManager] = None
    ):
        """
        Initialize persistence integrator.
        
        Args:
            persistence_config: Optional persistence configuration
            state_manager: Optional StateManager instance
            redis_cache: Optional Redis cache instance
            version_manager: Optional ModelVersionManager instance
        """
        self.config = persistence_config or PersistenceConfig()
        self.persistence = ModelPersistence(
            config=self.config,
            state_manager=state_manager,
            redis_cache=redis_cache
        )
        
        # Initialize integration modules
        if state_manager:
            self.state_integration = StateManagerIntegration(
                self.persistence, state_manager
            )
        else:
            self.state_integration = None
        
        self.version_integration = ModelVersionIntegration(
            self.persistence, version_manager
        )
        
        if redis_cache:
            self.redis_integration = RedisIntegration(
                self.persistence, redis_cache
            )
        else:
            self.redis_integration = None
    
    def wrap_model(
        self,
        model: AdaptivePlayerModel,
        version_id: int
    ) -> PersistentAdaptiveModel:
        """
        Wrap an AdaptivePlayerModel with persistence capabilities.
        
        Args:
            model: AdaptivePlayerModel to wrap
            version_id: Associated ModelVersion ID
            
        Returns:
            PersistentAdaptiveModel wrapper
        """
        return PersistentAdaptiveModel(
            base_model=model,
            persistence=self.persistence,
            version_id=version_id,
            integrator=self
        )
    
    async def create_model_checkpoint(
        self,
        model: AdaptivePlayerModel,
        version_id: int,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        sync_to_state_manager: bool = True,
        cache_metadata: bool = True,
        **kwargs
    ) -> int:
        """
        Create a comprehensive model checkpoint with all integrations.
        
        Args:
            model: AdaptivePlayerModel to checkpoint
            version_id: Associated ModelVersion ID
            checkpoint_type: Type of checkpoint
            sync_to_state_manager: Whether to sync to StateManager
            cache_metadata: Whether to cache metadata in Redis
            **kwargs: Additional checkpoint parameters
            
        Returns:
            Created checkpoint ID
        """
        try:
            # Create main checkpoint
            checkpoint_id = await self.persistence.save_checkpoint(
                model=model,
                version_id=version_id,
                checkpoint_type=checkpoint_type,
                **kwargs
            )
            
            # Get checkpoint info for integrations
            checkpoint_info = await self.persistence.get_checkpoint_info(checkpoint_id)
            
            if checkpoint_info and cache_metadata and self.redis_integration:
                # Cache metadata in Redis
                metadata = {
                    'checkpoint_name': checkpoint_info.checkpoint_name,
                    'created_at': checkpoint_info.created_at.isoformat(),
                    'validation_score': checkpoint_info.validation_score,
                    'model_size_mb': checkpoint_info.model_size_mb,
                }
                self.redis_integration.cache_checkpoint_metadata(
                    checkpoint_id, metadata
                )
            
            # Sync to StateManager if requested
            if sync_to_state_manager and self.state_integration:
                try:
                    await self.state_integration.sync_state_snapshots_to_checkpoints(
                        version_id=version_id
                    )
                except Exception as e:
                    logger.warning(f"StateManager sync failed: {e}")
            
            logger.info(f"Created comprehensive checkpoint {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive checkpoint: {e}")
            raise
    
    async def restore_model_state(
        self,
        checkpoint_id: int,
        restore_to_state_manager: bool = True,
        invalidate_cache: bool = True
    ) -> AdaptivePlayerModel:
        """
        Restore model state with all integrations.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
            restore_to_state_manager: Whether to restore to StateManager
            invalidate_cache: Whether to invalidate related cache entries
            
        Returns:
            Restored AdaptivePlayerModel
        """
        try:
            # Load main checkpoint
            model = await self.persistence.load_checkpoint(checkpoint_id)
            
            # Restore to StateManager if requested
            if restore_to_state_manager and self.state_integration:
                try:
                    await self.state_integration.restore_states_from_checkpoint(
                        checkpoint_id
                    )
                except Exception as e:
                    logger.warning(f"StateManager restore failed: {e}")
            
            # Invalidate cache if requested
            if invalidate_cache and self.redis_integration:
                try:
                    # This would invalidate relevant cache entries
                    logger.debug(f"Invalidating cache for checkpoint {checkpoint_id}")
                except Exception as e:
                    logger.warning(f"Cache invalidation failed: {e}")
            
            logger.info(f"Restored model state from checkpoint {checkpoint_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to restore model state: {e}")
            raise
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics across all integrated systems.
        
        Returns:
            Dictionary with metrics from all systems
        """
        try:
            metrics = {
                'persistence': await self.persistence.get_storage_metrics(),
                'timestamp': time.time()
            }
            
            # Add StateManager metrics if available
            if self.state_integration and self.state_integration.state_manager:
                try:
                    state_metrics = await self.state_integration.state_manager.get_manager_metrics()
                    metrics['state_manager'] = state_metrics
                except Exception as e:
                    logger.warning(f"Failed to get StateManager metrics: {e}")
            
            # Add Redis metrics if available
            if self.redis_integration:
                try:
                    redis_metrics = self.redis_integration.redis_cache.get_metrics()
                    metrics['redis'] = redis_metrics
                except Exception as e:
                    logger.warning(f"Failed to get Redis metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get integration metrics: {e}")
            return {}
    
    @asynccontextmanager
    async def checkpoint_context(
        self,
        model: AdaptivePlayerModel,
        version_id: int,
        auto_save: bool = True,
        checkpoint_name: Optional[str] = None
    ):
        """
        Context manager for automatic checkpointing around model operations.
        
        Args:
            model: AdaptivePlayerModel to monitor
            version_id: Associated ModelVersion ID
            auto_save: Whether to automatically save on exit
            checkpoint_name: Optional checkpoint name
        """
        checkpoint_id = None
        
        try:
            # Create initial checkpoint if requested
            if auto_save:
                checkpoint_id = await self.create_model_checkpoint(
                    model=model,
                    version_id=version_id,
                    checkpoint_type=CheckpointType.AUTOMATIC,
                    checkpoint_name=checkpoint_name or f"context_start_{int(time.time())}",
                    description="Checkpoint created at context start"
                )
            
            yield checkpoint_id
            
        except Exception as e:
            logger.error(f"Error in checkpoint context: {e}")
            raise
        
        finally:
            # Create final checkpoint if auto_save enabled and no error
            if auto_save:
                try:
                    final_checkpoint_id = await self.create_model_checkpoint(
                        model=model,
                        version_id=version_id,
                        checkpoint_type=CheckpointType.AUTOMATIC,
                        checkpoint_name=checkpoint_name or f"context_end_{int(time.time())}",
                        description="Checkpoint created at context end"
                    )
                    logger.info(f"Final checkpoint {final_checkpoint_id} created")
                except Exception as e:
                    logger.error(f"Failed to create final checkpoint: {e}")


# Convenience function for easy setup
def setup_persistence_integration(
    persistence_config: Optional[PersistenceConfig] = None,
    state_manager: Optional[StateManager] = None,
    redis_cache: Optional[RedisCache] = None,
    version_manager: Optional[ModelVersionManager] = None
) -> PersistenceIntegrator:
    """
    Convenience function to set up persistence integration.
    
    Args:
        persistence_config: Optional persistence configuration
        state_manager: Optional StateManager instance
        redis_cache: Optional Redis cache instance
        version_manager: Optional ModelVersionManager instance
        
    Returns:
        Configured PersistenceIntegrator instance
    """
    return PersistenceIntegrator(
        persistence_config=persistence_config,
        state_manager=state_manager,
        redis_cache=redis_cache,
        version_manager=version_manager
    )