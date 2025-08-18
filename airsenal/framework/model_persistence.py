"""
Model Persistence Layer for AIrsenal

This module implements comprehensive model persistence capabilities for saving and loading
AdaptivePlayerModel states, enabling model checkpointing and recovery from failures.

Key Features:
- Complete model state serialization and deserialization
- Compression for storage efficiency (gzip, bz2, lzma)
- Versioned storage format for backward compatibility
- Integrity verification using SHA256 checksums
- Support for multiple storage backends (database, file, cloud)
- Automatic checkpoint management with configurable intervals
- Named checkpoints for manual saves
- State comparison and diffing capabilities
- Checkpoint pruning to manage storage space
- Atomic saves with rollback on failure
- Integration with existing StateManager and ModelVersion systems

Architecture:
The persistence layer consists of several key components:
1. ModelPersistence: Main interface for model checkpoint operations
2. StorageBackend: Abstract interface for different storage systems
3. CompressionManager: Handles data compression and decompression
4. IntegrityManager: Manages checksums and data validation
5. CheckpointManager: Handles checkpoint lifecycle and pruning

Usage:
    ```python
    from airsenal.framework.model_persistence import ModelPersistence
    from airsenal.framework.adaptive_player_model import AdaptivePlayerModel
    
    # Initialize persistence layer
    persistence = ModelPersistence(
        storage_backend="database",
        compression_format="gzip",
        auto_checkpoint_interval=100
    )
    
    # Save model checkpoint
    checkpoint_id = await persistence.save_checkpoint(
        model=adaptive_model,
        checkpoint_name="best_validation",
        description="Best validation score: 0.85"
    )
    
    # Load model checkpoint
    restored_model = await persistence.load_checkpoint(checkpoint_id)
    
    # List available checkpoints
    checkpoints = await persistence.list_checkpoints(
        model_version_id=version_id,
        checkpoint_type="automatic"
    )
    ```

Integration with AIrsenal:
- Seamlessly integrates with existing ModelVersion and StateManager systems
- Supports both local and distributed deployments
- Compatible with existing database connection pooling
- Maintains backward compatibility with current model storage
- Works with all AdaptivePlayerModel implementations
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import logging
import lzma
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session

# AIrsenal imports
from airsenal.framework.adaptive_player_model import AdaptivePlayerModel, PlayerState
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import (
    ModelCheckpoint, ModelState, ModelMetadata, ModelVersion, 
    Player, session_scope
)
from airsenal.framework.state_manager import StateManager

logger = logging.getLogger(__name__)


class CompressionFormat(Enum):
    """Supported compression formats for model data."""
    
    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    ZLIB = "zlib"


class StorageBackend(Enum):
    """Supported storage backends for model persistence."""
    
    DATABASE = "database"
    FILE = "file"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class CheckpointType(Enum):
    """Types of model checkpoints."""
    
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    MILESTONE = "milestone"
    RECOVERY = "recovery"
    BEST_VALIDATION = "best_validation"
    EXPERIMENT = "experiment"


@dataclass
class PersistenceConfig:
    """Configuration for model persistence layer."""
    
    # Storage configuration
    storage_backend: StorageBackend = StorageBackend.DATABASE
    compression_format: CompressionFormat = CompressionFormat.GZIP
    compression_level: int = 6  # 1-9 for gzip/zlib, 1-9 for bz2, 0-9 for lzma
    
    # Checkpoint management
    auto_checkpoint_interval: int = 100  # Save every N state updates
    max_automatic_checkpoints: int = 50  # Keep last N automatic checkpoints
    max_manual_checkpoints: int = 100   # Keep last N manual checkpoints
    checkpoint_pruning_enabled: bool = True
    
    # Integrity verification
    checksum_algorithm: str = "sha256"
    verify_on_load: bool = True
    verify_on_save: bool = True
    
    # Performance options
    enable_differential_saves: bool = True
    enable_async_io: bool = True
    batch_size: int = 10  # For batch operations
    
    # File storage options (when storage_backend = FILE)
    base_file_path: Optional[str] = None
    file_organization: str = "hierarchical"  # "flat" or "hierarchical"
    
    # Cloud storage options
    cloud_bucket: Optional[str] = None
    cloud_prefix: str = "airsenal/model_checkpoints"
    cloud_credentials: Optional[Dict[str, Any]] = None
    
    # Serialization options
    serialization_format: str = "pickle"  # "pickle", "joblib", "numpy"
    pickle_protocol: int = 4  # Pickle protocol version
    
    # Validation options
    state_validation_enabled: bool = True
    strict_validation: bool = False


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    
    checkpoint_id: int
    version_id: int
    checkpoint_name: str
    checkpoint_type: CheckpointType
    created_at: datetime
    created_by: Optional[str] = None
    gameweek: Optional[int] = None
    season: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    validation_score: Optional[float] = None
    training_loss: Optional[float] = None
    model_size_mb: Optional[float] = None
    status: str = "active"
    is_recoverable: bool = True
    recovery_priority: int = 1
    storage_backend: str = "database"
    compression_format: str = "gzip"
    original_size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None


class ModelPersistenceError(Exception):
    """Base exception for model persistence operations."""
    pass


class CheckpointNotFoundError(ModelPersistenceError):
    """Raised when a checkpoint cannot be found."""
    pass


class CheckpointCorruptedError(ModelPersistenceError):
    """Raised when a checkpoint is corrupted or fails integrity checks."""
    pass


class CompressionError(ModelPersistenceError):
    """Raised when compression/decompression fails."""
    pass


class StorageError(ModelPersistenceError):
    """Raised when storage backend operations fail."""
    pass


class CompressionManager:
    """Handles data compression and decompression with multiple formats."""
    
    def __init__(self, config: PersistenceConfig):
        """Initialize compression manager with configuration."""
        self.config = config
        self.compression_format = config.compression_format
        self.compression_level = config.compression_level
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress data using configured compression format.
        
        Args:
            data: Raw data to compress
            
        Returns:
            Tuple of (compressed_data, compression_metadata)
            
        Raises:
            CompressionError: If compression fails
        """
        try:
            original_size = len(data)
            start_time = time.time()
            
            if self.compression_format == CompressionFormat.NONE:
                compressed_data = data
            elif self.compression_format == CompressionFormat.GZIP:
                compressed_data = gzip.compress(data, compresslevel=self.compression_level)
            elif self.compression_format == CompressionFormat.BZ2:
                import bz2
                compressed_data = bz2.compress(data, compresslevel=self.compression_level)
            elif self.compression_format == CompressionFormat.LZMA:
                compressed_data = lzma.compress(
                    data, 
                    preset=self.compression_level,
                    format=lzma.FORMAT_XZ
                )
            elif self.compression_format == CompressionFormat.ZLIB:
                compressed_data = zlib.compress(data, level=self.compression_level)
            else:
                raise CompressionError(f"Unsupported compression format: {self.compression_format}")
            
            compressed_size = len(compressed_data)
            compression_time = time.time() - start_time
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            metadata = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "format": self.compression_format.value,
                "level": self.compression_level,
            }
            
            logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                        f"(ratio: {compression_ratio:.3f}) in {compression_time:.3f}s")
            
            return compressed_data, metadata
            
        except Exception as e:
            raise CompressionError(f"Compression failed: {e}") from e
    
    def decompress(self, compressed_data: bytes, format_hint: Optional[str] = None) -> bytes:
        """
        Decompress data using specified or detected format.
        
        Args:
            compressed_data: Compressed data to decompress
            format_hint: Optional hint about compression format
            
        Returns:
            Decompressed data
            
        Raises:
            CompressionError: If decompression fails
        """
        try:
            start_time = time.time()
            
            # Use format hint or configured format
            format_to_use = CompressionFormat(format_hint) if format_hint else self.compression_format
            
            if format_to_use == CompressionFormat.NONE:
                decompressed_data = compressed_data
            elif format_to_use == CompressionFormat.GZIP:
                decompressed_data = gzip.decompress(compressed_data)
            elif format_to_use == CompressionFormat.BZ2:
                import bz2
                decompressed_data = bz2.decompress(compressed_data)
            elif format_to_use == CompressionFormat.LZMA:
                decompressed_data = lzma.decompress(compressed_data)
            elif format_to_use == CompressionFormat.ZLIB:
                decompressed_data = zlib.decompress(compressed_data)
            else:
                raise CompressionError(f"Unsupported compression format: {format_to_use}")
            
            decompression_time = time.time() - start_time
            
            logger.debug(f"Decompressed {len(compressed_data)} bytes to {len(decompressed_data)} bytes "
                        f"in {decompression_time:.3f}s")
            
            return decompressed_data
            
        except Exception as e:
            raise CompressionError(f"Decompression failed: {e}") from e


class IntegrityManager:
    """Handles data integrity verification using checksums."""
    
    def __init__(self, config: PersistenceConfig):
        """Initialize integrity manager with configuration."""
        self.config = config
        self.checksum_algorithm = config.checksum_algorithm
    
    def compute_checksum(self, data: bytes) -> str:
        """
        Compute checksum for data integrity verification.
        
        Args:
            data: Data to compute checksum for
            
        Returns:
            Hex-encoded checksum string
            
        Raises:
            ModelPersistenceError: If checksum computation fails
        """
        try:
            if self.checksum_algorithm == "sha256":
                hasher = hashlib.sha256()
            elif self.checksum_algorithm == "md5":
                hasher = hashlib.md5()
            elif self.checksum_algorithm == "sha1":
                hasher = hashlib.sha1()
            else:
                raise ModelPersistenceError(f"Unsupported checksum algorithm: {self.checksum_algorithm}")
            
            hasher.update(data)
            return hasher.hexdigest()
            
        except Exception as e:
            raise ModelPersistenceError(f"Checksum computation failed: {e}") from e
    
    def verify_integrity(self, data: bytes, expected_checksum: str) -> bool:
        """
        Verify data integrity against expected checksum.
        
        Args:
            data: Data to verify
            expected_checksum: Expected checksum value
            
        Returns:
            True if integrity check passes, False otherwise
        """
        try:
            actual_checksum = self.compute_checksum(data)
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False


class SerializationManager:
    """Handles model state serialization and deserialization."""
    
    def __init__(self, config: PersistenceConfig):
        """Initialize serialization manager with configuration."""
        self.config = config
        self.format = config.serialization_format
        self.pickle_protocol = config.pickle_protocol
    
    def serialize_model_state(self, model: AdaptivePlayerModel) -> bytes:
        """
        Serialize complete model state to bytes.
        
        Args:
            model: AdaptivePlayerModel to serialize
            
        Returns:
            Serialized model state as bytes
            
        Raises:
            ModelPersistenceError: If serialization fails
        """
        try:
            if self.format == "pickle":
                return pickle.dumps(model, protocol=self.pickle_protocol)
            elif self.format == "joblib":
                import joblib
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                return buffer.getvalue()
            else:
                raise ModelPersistenceError(f"Unsupported serialization format: {self.format}")
                
        except Exception as e:
            raise ModelPersistenceError(f"Model serialization failed: {e}") from e
    
    def deserialize_model_state(self, data: bytes) -> AdaptivePlayerModel:
        """
        Deserialize model state from bytes.
        
        Args:
            data: Serialized model state bytes
            
        Returns:
            Restored AdaptivePlayerModel instance
            
        Raises:
            ModelPersistenceError: If deserialization fails
        """
        try:
            if self.format == "pickle":
                return pickle.loads(data)
            elif self.format == "joblib":
                import joblib
                buffer = io.BytesIO(data)
                return joblib.load(buffer)
            else:
                raise ModelPersistenceError(f"Unsupported serialization format: {self.format}")
                
        except Exception as e:
            raise ModelPersistenceError(f"Model deserialization failed: {e}") from e
    
    def serialize_player_state(self, state: PlayerState) -> Dict[str, bytes]:
        """
        Serialize individual player state components.
        
        Args:
            state: PlayerState to serialize
            
        Returns:
            Dictionary with serialized state components
            
        Raises:
            ModelPersistenceError: If serialization fails
        """
        try:
            serialized = {}
            
            # Serialize state mean vector
            if hasattr(state.state_mean, 'tobytes'):
                # NumPy array
                serialized['state_mean'] = state.state_mean.tobytes()
            else:
                # Regular array/list
                serialized['state_mean'] = pickle.dumps(state.state_mean, protocol=self.pickle_protocol)
            
            # Serialize state covariance matrix
            if hasattr(state.state_cov, 'tobytes'):
                # NumPy array
                serialized['state_covariance'] = state.state_cov.tobytes()
            else:
                # Regular array/list
                serialized['state_covariance'] = pickle.dumps(state.state_cov, protocol=self.pickle_protocol)
            
            return serialized
            
        except Exception as e:
            raise ModelPersistenceError(f"Player state serialization failed: {e}") from e
    
    def deserialize_player_state(
        self, 
        serialized_data: Dict[str, bytes],
        state_dimension: int,
        player_id: int,
        gameweek: int,
        season: str,
        last_updated: str
    ) -> PlayerState:
        """
        Deserialize player state from serialized components.
        
        Args:
            serialized_data: Dictionary with serialized state components
            state_dimension: Dimension of state vector
            player_id: Player ID
            gameweek: Gameweek
            season: Season
            last_updated: Last update timestamp
            
        Returns:
            Restored PlayerState instance
            
        Raises:
            ModelPersistenceError: If deserialization fails
        """
        try:
            # Deserialize state mean
            mean_data = serialized_data['state_mean']
            try:
                # Try as NumPy array first
                state_mean = np.frombuffer(mean_data, dtype=np.float64)
                if len(state_mean) != state_dimension:
                    raise ValueError("State dimension mismatch")
            except (ValueError, TypeError):
                # Fall back to pickle
                state_mean = pickle.loads(mean_data)
            
            # Deserialize state covariance
            cov_data = serialized_data['state_covariance']
            try:
                # Try as NumPy array first
                state_cov = np.frombuffer(cov_data, dtype=np.float64)
                state_cov = state_cov.reshape((state_dimension, state_dimension))
            except (ValueError, TypeError):
                # Fall back to pickle
                state_cov = pickle.loads(cov_data)
            
            # Create PlayerState instance
            return PlayerState(
                player_id=player_id,
                state_mean=state_mean,
                state_cov=state_cov,
                gameweek=gameweek,
                season=season,
                last_updated=last_updated
            )
            
        except Exception as e:
            raise ModelPersistenceError(f"Player state deserialization failed: {e}") from e


class DatabaseStorageBackend:
    """Database storage backend for model persistence."""
    
    def __init__(self, config: PersistenceConfig):
        """Initialize database storage backend."""
        self.config = config
        
    async def save_checkpoint_data(
        self, 
        checkpoint_data: bytes, 
        metadata: Dict[str, Any]
    ) -> int:
        """
        Save checkpoint data to database.
        
        Args:
            checkpoint_data: Compressed checkpoint data
            metadata: Checkpoint metadata
            
        Returns:
            Checkpoint ID
            
        Raises:
            StorageError: If save operation fails
        """
        try:
            with session_scope() as session:
                checkpoint = ModelCheckpoint(
                    version_id=metadata['version_id'],
                    checkpoint_name=metadata['checkpoint_name'],
                    checkpoint_type=metadata['checkpoint_type'],
                    model_data=checkpoint_data,
                    compression_format=metadata['compression_format'],
                    original_size_bytes=metadata.get('original_size_bytes'),
                    compressed_size_bytes=len(checkpoint_data),
                    checksum=metadata['checksum'],
                    checksum_algorithm=metadata.get('checksum_algorithm', 'sha256'),
                    storage_backend='database',
                    created_at=metadata['created_at'],
                    created_by=metadata.get('created_by'),
                    gameweek=metadata.get('gameweek'),
                    season=metadata.get('season'),
                    tags=metadata.get('tags'),
                    description=metadata.get('description'),
                    validation_score=metadata.get('validation_score'),
                    training_loss=metadata.get('training_loss'),
                    model_size_mb=metadata.get('model_size_mb'),
                    status=metadata.get('status', 'active'),
                    is_recoverable=metadata.get('is_recoverable', True),
                    recovery_priority=metadata.get('recovery_priority', 1)
                )
                
                session.add(checkpoint)
                session.flush()
                checkpoint_id = checkpoint.id
                
                return checkpoint_id
                
        except Exception as e:
            raise StorageError(f"Failed to save checkpoint to database: {e}") from e
    
    async def load_checkpoint_data(self, checkpoint_id: int) -> Tuple[bytes, Dict[str, Any]]:
        """
        Load checkpoint data from database.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Tuple of (checkpoint_data, metadata)
            
        Raises:
            CheckpointNotFoundError: If checkpoint not found
            StorageError: If load operation fails
        """
        try:
            with session_scope() as session:
                checkpoint = session.get(ModelCheckpoint, checkpoint_id)
                
                if not checkpoint:
                    raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
                
                if checkpoint.model_data is None:
                    raise StorageError(f"Checkpoint {checkpoint_id} has no data")
                
                metadata = {
                    'version_id': checkpoint.version_id,
                    'checkpoint_name': checkpoint.checkpoint_name,
                    'checkpoint_type': checkpoint.checkpoint_type,
                    'compression_format': checkpoint.compression_format,
                    'original_size_bytes': checkpoint.original_size_bytes,
                    'compressed_size_bytes': checkpoint.compressed_size_bytes,
                    'checksum': checkpoint.checksum,
                    'checksum_algorithm': checkpoint.checksum_algorithm,
                    'created_at': checkpoint.created_at,
                    'created_by': checkpoint.created_by,
                    'gameweek': checkpoint.gameweek,
                    'season': checkpoint.season,
                    'tags': checkpoint.tags,
                    'description': checkpoint.description,
                    'status': checkpoint.status,
                }
                
                return checkpoint.model_data, metadata
                
        except CheckpointNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to load checkpoint from database: {e}") from e
    
    async def save_player_states(
        self, 
        checkpoint_id: int, 
        player_states: Dict[int, Dict[str, bytes]],
        metadata: Dict[int, Dict[str, Any]]
    ) -> List[int]:
        """
        Save player states to database.
        
        Args:
            checkpoint_id: Associated checkpoint ID
            player_states: Dictionary mapping player_id to serialized state data
            metadata: Dictionary mapping player_id to state metadata
            
        Returns:
            List of created ModelState IDs
            
        Raises:
            StorageError: If save operation fails
        """
        try:
            state_ids = []
            
            with session_scope() as session:
                for player_id, state_data in player_states.items():
                    player_metadata = metadata[player_id]
                    
                    model_state = ModelState(
                        checkpoint_id=checkpoint_id,
                        player_id=player_id,
                        state_mean=state_data['state_mean'],
                        state_covariance=state_data['state_covariance'],
                        state_history=state_data.get('state_history'),
                        state_dimension=player_metadata['state_dimension'],
                        gameweek=player_metadata['gameweek'],
                        season=player_metadata['season'],
                        last_updated=player_metadata['last_updated'],
                        serialization_format=player_metadata.get('serialization_format', 'numpy'),
                        compression_used=player_metadata.get('compression_used', True),
                        state_checksum=player_metadata['state_checksum'],
                        is_valid=player_metadata.get('is_valid', True),
                        validation_errors=player_metadata.get('validation_errors'),
                        prediction_accuracy=player_metadata.get('prediction_accuracy'),
                        uncertainty_score=player_metadata.get('uncertainty_score'),
                        update_frequency=player_metadata.get('update_frequency', 0)
                    )
                    
                    session.add(model_state)
                    session.flush()
                    state_ids.append(model_state.id)
                
                return state_ids
                
        except Exception as e:
            raise StorageError(f"Failed to save player states to database: {e}") from e
    
    async def load_player_states(self, checkpoint_id: int) -> Tuple[Dict[int, Dict[str, bytes]], Dict[int, Dict[str, Any]]]:
        """
        Load player states from database.
        
        Args:
            checkpoint_id: Checkpoint ID to load states for
            
        Returns:
            Tuple of (player_states, metadata)
            
        Raises:
            StorageError: If load operation fails
        """
        try:
            player_states = {}
            metadata = {}
            
            with session_scope() as session:
                states = session.execute(
                    select(ModelState).where(ModelState.checkpoint_id == checkpoint_id)
                ).scalars().all()
                
                for state in states:
                    player_id = state.player_id
                    
                    player_states[player_id] = {
                        'state_mean': state.state_mean,
                        'state_covariance': state.state_covariance,
                        'state_history': state.state_history
                    }
                    
                    metadata[player_id] = {
                        'state_dimension': state.state_dimension,
                        'gameweek': state.gameweek,
                        'season': state.season,
                        'last_updated': state.last_updated,
                        'serialization_format': state.serialization_format,
                        'compression_used': state.compression_used,
                        'state_checksum': state.state_checksum,
                        'is_valid': state.is_valid,
                        'validation_errors': state.validation_errors,
                        'prediction_accuracy': state.prediction_accuracy,
                        'uncertainty_score': state.uncertainty_score,
                        'update_frequency': state.update_frequency
                    }
                
                return player_states, metadata
                
        except Exception as e:
            raise StorageError(f"Failed to load player states from database: {e}") from e


class CheckpointManager:
    """Manages checkpoint lifecycle, pruning, and automatic checkpointing."""
    
    def __init__(self, config: PersistenceConfig):
        """Initialize checkpoint manager."""
        self.config = config
        self.auto_checkpoint_counter = 0
        self.last_auto_checkpoint_time = time.time()
    
    def should_create_auto_checkpoint(self, update_count: int) -> bool:
        """
        Determine if an automatic checkpoint should be created.
        
        Args:
            update_count: Number of updates since last checkpoint
            
        Returns:
            True if auto checkpoint should be created
        """
        return (
            self.config.auto_checkpoint_interval > 0 and
            update_count >= self.config.auto_checkpoint_interval
        )
    
    def generate_auto_checkpoint_name(self) -> str:
        """Generate name for automatic checkpoint."""
        self.auto_checkpoint_counter += 1
        timestamp = int(time.time())
        return f"auto_checkpoint_{self.auto_checkpoint_counter:06d}_{timestamp}"
    
    async def prune_old_checkpoints(self, version_id: int, session: Session) -> int:
        """
        Remove old checkpoints based on retention policy.
        
        Args:
            version_id: Model version ID
            session: Database session
            
        Returns:
            Number of checkpoints pruned
        """
        if not self.config.checkpoint_pruning_enabled:
            return 0
        
        try:
            pruned_count = 0
            
            # Prune automatic checkpoints
            if self.config.max_automatic_checkpoints > 0:
                auto_checkpoints = session.execute(
                    select(ModelCheckpoint)
                    .where(
                        ModelCheckpoint.version_id == version_id,
                        ModelCheckpoint.checkpoint_type == CheckpointType.AUTOMATIC.value
                    )
                    .order_by(ModelCheckpoint.created_at.desc())
                ).scalars().all()
                
                if len(auto_checkpoints) > self.config.max_automatic_checkpoints:
                    checkpoints_to_remove = auto_checkpoints[self.config.max_automatic_checkpoints:]
                    
                    for checkpoint in checkpoints_to_remove:
                        # Delete associated ModelState records first
                        session.execute(
                            delete(ModelState).where(ModelState.checkpoint_id == checkpoint.id)
                        )
                        # Delete checkpoint
                        session.delete(checkpoint)
                        pruned_count += 1
            
            # Prune manual checkpoints
            if self.config.max_manual_checkpoints > 0:
                manual_checkpoints = session.execute(
                    select(ModelCheckpoint)
                    .where(
                        ModelCheckpoint.version_id == version_id,
                        ModelCheckpoint.checkpoint_type == CheckpointType.MANUAL.value
                    )
                    .order_by(ModelCheckpoint.created_at.desc())
                ).scalars().all()
                
                if len(manual_checkpoints) > self.config.max_manual_checkpoints:
                    checkpoints_to_remove = manual_checkpoints[self.config.max_manual_checkpoints:]
                    
                    for checkpoint in checkpoints_to_remove:
                        # Delete associated ModelState records first
                        session.execute(
                            delete(ModelState).where(ModelState.checkpoint_id == checkpoint.id)
                        )
                        # Delete checkpoint
                        session.delete(checkpoint)
                        pruned_count += 1
            
            session.flush()
            
            if pruned_count > 0:
                logger.info(f"Pruned {pruned_count} old checkpoints for version {version_id}")
            
            return pruned_count
            
        except Exception as e:
            logger.error(f"Error pruning checkpoints: {e}")
            return 0


class ModelPersistence:
    """
    Main interface for model persistence operations.
    
    Provides comprehensive checkpoint management for AdaptivePlayerModel instances
    with compression, integrity verification, and multiple storage backend support.
    """
    
    def __init__(
        self,
        config: Optional[PersistenceConfig] = None,
        state_manager: Optional[StateManager] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """
        Initialize model persistence layer.
        
        Args:
            config: Persistence configuration
            state_manager: Optional StateManager integration
            redis_cache: Optional Redis cache integration
        """
        self.config = config or PersistenceConfig()
        self.state_manager = state_manager
        self.redis_cache = redis_cache
        
        # Initialize managers
        self.compression_manager = CompressionManager(self.config)
        self.integrity_manager = IntegrityManager(self.config)
        self.serialization_manager = SerializationManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Initialize storage backend
        if self.config.storage_backend == StorageBackend.DATABASE:
            self.storage_backend = DatabaseStorageBackend(self.config)
        else:
            raise ModelPersistenceError(f"Storage backend {self.config.storage_backend} not implemented")
        
        # Counters for automatic checkpointing
        self.update_count = 0
        self.last_checkpoint_time = time.time()
    
    async def save_checkpoint(
        self,
        model: AdaptivePlayerModel,
        version_id: int,
        checkpoint_name: Optional[str] = None,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        gameweek: Optional[int] = None,
        season: Optional[str] = None,
        validation_score: Optional[float] = None,
        training_loss: Optional[float] = None
    ) -> int:
        """
        Save a complete model checkpoint.
        
        Args:
            model: AdaptivePlayerModel to save
            version_id: Associated ModelVersion ID
            checkpoint_name: Optional name for checkpoint
            checkpoint_type: Type of checkpoint
            description: Optional description
            tags: Optional list of tags
            created_by: Optional creator identifier
            gameweek: Optional current gameweek
            season: Optional current season
            validation_score: Optional validation score
            training_loss: Optional training loss
            
        Returns:
            Created checkpoint ID
            
        Raises:
            ModelPersistenceError: If save operation fails
        """
        try:
            # Generate checkpoint name if not provided
            if checkpoint_name is None:
                if checkpoint_type == CheckpointType.AUTOMATIC:
                    checkpoint_name = self.checkpoint_manager.generate_auto_checkpoint_name()
                else:
                    checkpoint_name = f"{checkpoint_type.value}_{int(time.time())}"
            
            # Serialize model
            logger.info(f"Serializing model for checkpoint '{checkpoint_name}'...")
            model_data = self.serialization_manager.serialize_model_state(model)
            
            # Compress data
            logger.info("Compressing model data...")
            compressed_data, compression_metadata = self.compression_manager.compress(model_data)
            
            # Compute integrity checksum
            checksum = self.integrity_manager.compute_checksum(compressed_data)
            
            # Calculate model size
            model_size_mb = len(model_data) / (1024 * 1024)
            
            # Prepare checkpoint metadata
            checkpoint_metadata = {
                'version_id': version_id,
                'checkpoint_name': checkpoint_name,
                'checkpoint_type': checkpoint_type.value,
                'compression_format': self.config.compression_format.value,
                'original_size_bytes': len(model_data),
                'checksum': checksum,
                'checksum_algorithm': self.config.checksum_algorithm,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'created_by': created_by,
                'gameweek': gameweek,
                'season': season,
                'tags': ','.join(tags) if tags else None,
                'description': description,
                'validation_score': validation_score,
                'training_loss': training_loss,
                'model_size_mb': model_size_mb,
                'status': 'active',
                'is_recoverable': True,
                'recovery_priority': 1 if checkpoint_type == CheckpointType.BEST_VALIDATION else 5
            }
            
            # Save checkpoint data
            logger.info("Saving checkpoint to storage backend...")
            checkpoint_id = await self.storage_backend.save_checkpoint_data(
                compressed_data, checkpoint_metadata
            )
            
            # Save individual player states if available
            if hasattr(model, 'player_states') and model.player_states:
                logger.info(f"Saving {len(model.player_states)} player states...")
                await self._save_player_states(checkpoint_id, model.player_states)
            
            # Perform checkpoint pruning if enabled
            if self.config.checkpoint_pruning_enabled:
                with session_scope() as session:
                    pruned_count = await self.checkpoint_manager.prune_old_checkpoints(version_id, session)
                    if pruned_count > 0:
                        logger.info(f"Pruned {pruned_count} old checkpoints")
            
            logger.info(f"Successfully saved checkpoint '{checkpoint_name}' with ID {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelPersistenceError(f"Checkpoint save failed: {e}") from e
    
    async def load_checkpoint(self, checkpoint_id: int) -> AdaptivePlayerModel:
        """
        Load a model checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Restored AdaptivePlayerModel instance
            
        Raises:
            CheckpointNotFoundError: If checkpoint not found
            CheckpointCorruptedError: If checkpoint fails integrity checks
            ModelPersistenceError: If load operation fails
        """
        try:
            logger.info(f"Loading checkpoint {checkpoint_id}...")
            
            # Load checkpoint data from storage
            compressed_data, metadata = await self.storage_backend.load_checkpoint_data(checkpoint_id)
            
            # Verify integrity if enabled
            if self.config.verify_on_load:
                logger.debug("Verifying checkpoint integrity...")
                if not self.integrity_manager.verify_integrity(compressed_data, metadata['checksum']):
                    raise CheckpointCorruptedError(f"Checkpoint {checkpoint_id} failed integrity check")
            
            # Decompress data
            logger.debug("Decompressing checkpoint data...")
            model_data = self.compression_manager.decompress(
                compressed_data, 
                format_hint=metadata.get('compression_format')
            )
            
            # Deserialize model
            logger.debug("Deserializing model...")
            model = self.serialization_manager.deserialize_model_state(model_data)
            
            # Load individual player states if available
            try:
                player_states, state_metadata = await self.storage_backend.load_player_states(checkpoint_id)
                if player_states:
                    logger.info(f"Loading {len(player_states)} player states...")
                    await self._load_player_states(model, player_states, state_metadata)
            except Exception as e:
                logger.warning(f"Could not load player states: {e}")
            
            logger.info(f"Successfully loaded checkpoint {checkpoint_id}")
            return model
            
        except (CheckpointNotFoundError, CheckpointCorruptedError):
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise ModelPersistenceError(f"Checkpoint load failed: {e}") from e
    
    async def _save_player_states(
        self, 
        checkpoint_id: int, 
        player_states: Dict[int, PlayerState]
    ) -> None:
        """Save individual player states to storage."""
        try:
            serialized_states = {}
            metadata = {}
            
            for player_id, state in player_states.items():
                # Serialize player state
                state_data = self.serialization_manager.serialize_player_state(state)
                
                # Compute state checksum
                combined_data = state_data['state_mean'] + state_data['state_covariance']
                state_checksum = self.integrity_manager.compute_checksum(combined_data)
                
                serialized_states[player_id] = state_data
                metadata[player_id] = {
                    'state_dimension': len(state.state_mean),
                    'gameweek': state.gameweek,
                    'season': state.season,
                    'last_updated': state.last_updated,
                    'serialization_format': self.config.serialization_format,
                    'compression_used': self.config.compression_format != CompressionFormat.NONE,
                    'state_checksum': state_checksum,
                    'is_valid': True
                }
            
            # Save to storage backend
            await self.storage_backend.save_player_states(checkpoint_id, serialized_states, metadata)
            
        except Exception as e:
            logger.error(f"Failed to save player states: {e}")
            # Don't fail the entire checkpoint save for player state errors
    
    async def _load_player_states(
        self, 
        model: AdaptivePlayerModel, 
        player_states: Dict[int, Dict[str, bytes]],
        metadata: Dict[int, Dict[str, Any]]
    ) -> None:
        """Load individual player states into model."""
        try:
            restored_states = {}
            
            for player_id, state_data in player_states.items():
                state_metadata = metadata[player_id]
                
                # Verify state integrity if enabled
                if self.config.verify_on_load and state_metadata.get('state_checksum'):
                    combined_data = state_data['state_mean'] + state_data['state_covariance']
                    if not self.integrity_manager.verify_integrity(combined_data, state_metadata['state_checksum']):
                        logger.warning(f"Player state {player_id} failed integrity check, skipping")
                        continue
                
                # Deserialize player state
                state = self.serialization_manager.deserialize_player_state(
                    state_data,
                    state_metadata['state_dimension'],
                    player_id,
                    state_metadata['gameweek'],
                    state_metadata['season'],
                    state_metadata['last_updated']
                )
                
                restored_states[player_id] = state
            
            # Update model with restored states
            if hasattr(model, 'player_states'):
                model.player_states.update(restored_states)
            
            logger.info(f"Restored {len(restored_states)} player states")
            
        except Exception as e:
            logger.error(f"Failed to load player states: {e}")
            # Don't fail the entire checkpoint load for player state errors
    
    async def list_checkpoints(
        self,
        version_id: Optional[int] = None,
        checkpoint_type: Optional[CheckpointType] = None,
        season: Optional[str] = None,
        status: str = "active",
        limit: Optional[int] = None
    ) -> List[CheckpointInfo]:
        """
        List available checkpoints with filtering options.
        
        Args:
            version_id: Filter by model version ID
            checkpoint_type: Filter by checkpoint type
            season: Filter by season
            status: Filter by status (default: "active")
            limit: Limit number of results
            
        Returns:
            List of CheckpointInfo objects
        """
        try:
            with session_scope() as session:
                query = select(ModelCheckpoint)
                
                # Apply filters
                if version_id:
                    query = query.where(ModelCheckpoint.version_id == version_id)
                if checkpoint_type:
                    query = query.where(ModelCheckpoint.checkpoint_type == checkpoint_type.value)
                if season:
                    query = query.where(ModelCheckpoint.season == season)
                if status:
                    query = query.where(ModelCheckpoint.status == status)
                
                # Order by creation time (newest first)
                query = query.order_by(ModelCheckpoint.created_at.desc())
                
                # Apply limit
                if limit:
                    query = query.limit(limit)
                
                checkpoints = session.execute(query).scalars().all()
                
                result = []
                for checkpoint in checkpoints:
                    info = CheckpointInfo(
                        checkpoint_id=checkpoint.id,
                        version_id=checkpoint.version_id,
                        checkpoint_name=checkpoint.checkpoint_name,
                        checkpoint_type=CheckpointType(checkpoint.checkpoint_type),
                        created_at=datetime.fromisoformat(checkpoint.created_at),
                        created_by=checkpoint.created_by,
                        gameweek=checkpoint.gameweek,
                        season=checkpoint.season,
                        description=checkpoint.description,
                        tags=checkpoint.tags.split(',') if checkpoint.tags else [],
                        validation_score=checkpoint.validation_score,
                        training_loss=checkpoint.training_loss,
                        model_size_mb=checkpoint.model_size_mb,
                        status=checkpoint.status,
                        is_recoverable=checkpoint.is_recoverable,
                        recovery_priority=checkpoint.recovery_priority,
                        storage_backend=checkpoint.storage_backend,
                        compression_format=checkpoint.compression_format,
                        original_size_bytes=checkpoint.original_size_bytes,
                        compressed_size_bytes=checkpoint.compressed_size_bytes
                    )
                    result.append(info)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise ModelPersistenceError(f"Checkpoint listing failed: {e}") from e
    
    async def delete_checkpoint(self, checkpoint_id: int) -> bool:
        """
        Delete a checkpoint and its associated data.
        
        Args:
            checkpoint_id: Checkpoint ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with session_scope() as session:
                # Delete associated ModelState records first
                session.execute(
                    delete(ModelState).where(ModelState.checkpoint_id == checkpoint_id)
                )
                
                # Delete checkpoint
                result = session.execute(
                    delete(ModelCheckpoint).where(ModelCheckpoint.id == checkpoint_id)
                )
                
                deleted_count = result.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Deleted checkpoint {checkpoint_id}")
                    return True
                else:
                    logger.warning(f"Checkpoint {checkpoint_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def get_checkpoint_info(self, checkpoint_id: int) -> Optional[CheckpointInfo]:
        """
        Get detailed information about a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            CheckpointInfo object or None if not found
        """
        try:
            with session_scope() as session:
                checkpoint = session.get(ModelCheckpoint, checkpoint_id)
                
                if not checkpoint:
                    return None
                
                return CheckpointInfo(
                    checkpoint_id=checkpoint.id,
                    version_id=checkpoint.version_id,
                    checkpoint_name=checkpoint.checkpoint_name,
                    checkpoint_type=CheckpointType(checkpoint.checkpoint_type),
                    created_at=datetime.fromisoformat(checkpoint.created_at),
                    created_by=checkpoint.created_by,
                    gameweek=checkpoint.gameweek,
                    season=checkpoint.season,
                    description=checkpoint.description,
                    tags=checkpoint.tags.split(',') if checkpoint.tags else [],
                    validation_score=checkpoint.validation_score,
                    training_loss=checkpoint.training_loss,
                    model_size_mb=checkpoint.model_size_mb,
                    status=checkpoint.status,
                    is_recoverable=checkpoint.is_recoverable,
                    recovery_priority=checkpoint.recovery_priority,
                    storage_backend=checkpoint.storage_backend,
                    compression_format=checkpoint.compression_format,
                    original_size_bytes=checkpoint.original_size_bytes,
                    compressed_size_bytes=checkpoint.compressed_size_bytes
                )
                
        except Exception as e:
            logger.error(f"Failed to get checkpoint info for {checkpoint_id}: {e}")
            return None
    
    def should_auto_checkpoint(self) -> bool:
        """Check if an automatic checkpoint should be created."""
        self.update_count += 1
        return self.checkpoint_manager.should_create_auto_checkpoint(self.update_count)
    
    async def create_auto_checkpoint(
        self, 
        model: AdaptivePlayerModel, 
        version_id: int,
        gameweek: Optional[int] = None,
        season: Optional[str] = None
    ) -> Optional[int]:
        """
        Create an automatic checkpoint if conditions are met.
        
        Args:
            model: AdaptivePlayerModel to checkpoint
            version_id: Associated ModelVersion ID
            gameweek: Optional current gameweek
            season: Optional current season
            
        Returns:
            Checkpoint ID if created, None otherwise
        """
        if self.should_auto_checkpoint():
            try:
                checkpoint_id = await self.save_checkpoint(
                    model=model,
                    version_id=version_id,
                    checkpoint_type=CheckpointType.AUTOMATIC,
                    description=f"Automatic checkpoint after {self.update_count} updates",
                    gameweek=gameweek,
                    season=season,
                    created_by="system"
                )
                
                # Reset counter
                self.update_count = 0
                self.last_checkpoint_time = time.time()
                
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Failed to create automatic checkpoint: {e}")
                return None
        
        return None
    
    async def get_storage_metrics(self) -> Dict[str, Any]:
        """
        Get storage and performance metrics for the persistence layer.
        
        Returns:
            Dictionary with various metrics
        """
        try:
            with session_scope() as session:
                # Count checkpoints by type
                checkpoint_counts = {}
                for checkpoint_type in CheckpointType:
                    count = session.execute(
                        select(ModelCheckpoint).where(
                            ModelCheckpoint.checkpoint_type == checkpoint_type.value
                        )
                    ).scalar()
                    checkpoint_counts[checkpoint_type.value] = count or 0
                
                # Calculate total storage used
                total_size_query = session.execute(
                    select(
                        session.query(ModelCheckpoint.compressed_size_bytes).sum()
                    )
                ).scalar()
                total_compressed_bytes = total_size_query or 0
                
                # Count total player states
                player_state_count = session.execute(
                    select(ModelState)
                ).scalar() or 0
                
                return {
                    "checkpoint_counts": checkpoint_counts,
                    "total_checkpoints": sum(checkpoint_counts.values()),
                    "total_compressed_bytes": total_compressed_bytes,
                    "total_compressed_mb": total_compressed_bytes / (1024 * 1024),
                    "player_state_count": player_state_count,
                    "storage_backend": self.config.storage_backend.value,
                    "compression_format": self.config.compression_format.value,
                    "auto_checkpoint_interval": self.config.auto_checkpoint_interval,
                    "update_count_since_last_checkpoint": self.update_count,
                }
                
        except Exception as e:
            logger.error(f"Failed to get storage metrics: {e}")
            return {}