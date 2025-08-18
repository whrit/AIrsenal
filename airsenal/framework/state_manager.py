"""
Model State Management System for AIrsenal

Comprehensive state management system for tracking player abilities over time with
state initialization, updates, history tracking, and rollback capabilities.

This module provides:
- StateManager: Main interface for managing all player states
- StateRepository: Abstract interface for state storage (memory, Redis, database)
- StateHistory: Track state changes over time with timestamps
- StateSnapshot: Immutable state capture at a point in time
- StateTransition: Record of state changes with metadata
- StateValidator: Ensure state consistency and validity

Key Features:
- Thread-safe operations using asyncio and threading locks
- Atomic state updates with rollback on failure
- State versioning and history tracking
- Integration with existing Redis cache and AdaptivePlayerModel
- Distributed state management support
- Efficient state storage with compression
- State diffing for debugging and auditing

Architecture:
The state management system is built on top of the existing PlayerState class
from the AdaptivePlayerModel and integrates with the Redis cache system.
It provides a layered architecture with clear separation between state
persistence, business logic, and validation.

Usage:
    ```python
    from airsenal.framework.state_manager import StateManager
    from airsenal.framework.redis_cache import redis_cache
    
    # Initialize state manager
    state_manager = StateManager(
        repository=RedisStateRepository(redis_cache),
        enable_history=True,
        max_history_size=100
    )
    
    # Initialize states from historical data
    await state_manager.initialize_from_historical_data(
        season="2023", max_gameweek=15
    )
    
    # Update state with new observations
    await state_manager.update_player_state(
        player_id=123,
        new_state=updated_state,
        metadata={"source": "live_data", "confidence": 0.95}
    )
    
    # Query state history
    history = await state_manager.get_state_history(
        player_id=123, from_gameweek=10, to_gameweek=15
    )
    ```
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import zlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from airsenal.framework.adaptive_player_model import PlayerState, StateSpaceConfig
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import Player, PlayerScore

logger = logging.getLogger(__name__)


class StateOperationType(Enum):
    """Types of state operations for tracking and auditing."""
    
    INITIALIZE = "initialize"
    UPDATE = "update"
    PREDICT = "predict"
    ROLLBACK = "rollback"
    SNAPSHOT = "snapshot"
    MERGE = "merge"


class StateValidationError(Exception):
    """Raised when state validation fails."""
    pass


class StateManagerError(Exception):
    """Base exception for state manager operations."""
    pass


class RollbackError(StateManagerError):
    """Raised when state rollback fails."""
    pass


@dataclass(frozen=True)
class StateSnapshot:
    """
    Immutable snapshot of player state at a specific point in time.
    
    Provides a point-in-time view of player state that can be used for:
    - State history tracking
    - Rollback operations
    - State comparison and diffing
    - Audit trails
    """
    
    snapshot_id: str
    player_id: int
    state: PlayerState
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = field(init=False)
    
    def __post_init__(self):
        """Compute checksum for data integrity verification."""
        # Create checksum of state data for integrity checking
        state_data = {
            "player_id": self.player_id,
            "state_mean": self.state.state_mean.tolist() if hasattr(self.state.state_mean, 'tolist') else list(self.state.state_mean),
            "state_cov": self.state.state_cov.tolist() if hasattr(self.state.state_cov, 'tolist') else list(self.state.state_cov),
            "gameweek": self.state.gameweek,
            "season": self.state.season,
        }
        state_str = json.dumps(state_data, sort_keys=True)
        checksum = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        object.__setattr__(self, 'checksum', checksum)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            "snapshot_id": self.snapshot_id,
            "player_id": self.player_id,
            "state": self.state.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSnapshot:
        """Create snapshot from dictionary representation."""
        return cls(
            snapshot_id=data["snapshot_id"],
            player_id=data["player_id"],
            state=PlayerState.from_dict(data["state"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )
    
    def verify_integrity(self) -> bool:
        """Verify snapshot data integrity using checksum."""
        try:
            # Recompute checksum and compare
            temp_snapshot = StateSnapshot(
                snapshot_id=self.snapshot_id,
                player_id=self.player_id,
                state=self.state,
                timestamp=self.timestamp,
                metadata=self.metadata,
            )
            return temp_snapshot.checksum == self.checksum
        except Exception as e:
            logger.error(f"Error verifying snapshot integrity: {e}")
            return False


@dataclass
class StateTransition:
    """
    Record of a state change with complete metadata for auditing and rollback.
    
    Captures all information needed to understand and potentially reverse
    a state change operation.
    """
    
    transition_id: str = field(default_factory=lambda: str(uuid4()))
    player_id: int = 0
    operation_type: StateOperationType = StateOperationType.UPDATE
    previous_state: Optional[PlayerState] = None
    new_state: Optional[PlayerState] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary representation."""
        return {
            "transition_id": self.transition_id,
            "player_id": self.player_id,
            "operation_type": self.operation_type.value,
            "previous_state": self.previous_state.to_dict() if self.previous_state else None,
            "new_state": self.new_state.to_dict() if self.new_state else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateTransition:
        """Create transition from dictionary representation."""
        return cls(
            transition_id=data["transition_id"],
            player_id=data["player_id"],
            operation_type=StateOperationType(data["operation_type"]),
            previous_state=PlayerState.from_dict(data["previous_state"]) if data.get("previous_state") else None,
            new_state=PlayerState.from_dict(data["new_state"]) if data.get("new_state") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


class StateHistory:
    """
    Manages historical state information for players with efficient storage and querying.
    
    Features:
    - Efficient storage using compression
    - Time-based and gameweek-based querying
    - Automatic cleanup of old history
    - Fast state diff computation
    """
    
    def __init__(self, max_history_size: int = 1000, compression_enabled: bool = True):
        """
        Initialize state history tracker.
        
        Args:
            max_history_size: Maximum number of history entries per player
            compression_enabled: Whether to compress stored history data
        """
        self.max_history_size = max_history_size
        self.compression_enabled = compression_enabled
        self._history: Dict[int, List[StateSnapshot]] = {}
        self._transitions: Dict[int, List[StateTransition]] = {}
        self._lock = threading.RLock()
    
    def add_snapshot(self, snapshot: StateSnapshot) -> None:
        """Add a state snapshot to history."""
        with self._lock:
            player_id = snapshot.player_id
            
            if player_id not in self._history:
                self._history[player_id] = []
            
            self._history[player_id].append(snapshot)
            
            # Maintain max history size
            if len(self._history[player_id]) > self.max_history_size:
                self._history[player_id] = self._history[player_id][-self.max_history_size:]
    
    def add_transition(self, transition: StateTransition) -> None:
        """Add a state transition to history."""
        with self._lock:
            player_id = transition.player_id
            
            if player_id not in self._transitions:
                self._transitions[player_id] = []
            
            self._transitions[player_id].append(transition)
            
            # Maintain max history size
            if len(self._transitions[player_id]) > self.max_history_size:
                self._transitions[player_id] = self._transitions[player_id][-self.max_history_size:]
    
    def get_snapshots(
        self,
        player_id: int,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        from_gameweek: Optional[int] = None,
        to_gameweek: Optional[int] = None,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[StateSnapshot]:
        """Get state snapshots with optional filtering."""
        with self._lock:
            if player_id not in self._history:
                return []
            
            snapshots = self._history[player_id].copy()
            
            # Apply filters
            if from_timestamp:
                snapshots = [s for s in snapshots if s.timestamp >= from_timestamp]
            
            if to_timestamp:
                snapshots = [s for s in snapshots if s.timestamp <= to_timestamp]
            
            if season:
                snapshots = [s for s in snapshots if s.state.season == season]
            
            if from_gameweek is not None:
                snapshots = [s for s in snapshots if s.state.gameweek >= from_gameweek]
            
            if to_gameweek is not None:
                snapshots = [s for s in snapshots if s.state.gameweek <= to_gameweek]
            
            # Sort by timestamp
            snapshots.sort(key=lambda s: s.timestamp)
            
            # Apply limit
            if limit:
                snapshots = snapshots[-limit:]
            
            return snapshots
    
    def get_transitions(
        self,
        player_id: int,
        operation_type: Optional[StateOperationType] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[StateTransition]:
        """Get state transitions with optional filtering."""
        with self._lock:
            if player_id not in self._transitions:
                return []
            
            transitions = self._transitions[player_id].copy()
            
            # Apply filters
            if operation_type:
                transitions = [t for t in transitions if t.operation_type == operation_type]
            
            if from_timestamp:
                transitions = [t for t in transitions if t.timestamp >= from_timestamp]
            
            if to_timestamp:
                transitions = [t for t in transitions if t.timestamp <= to_timestamp]
            
            # Sort by timestamp
            transitions.sort(key=lambda t: t.timestamp)
            
            # Apply limit
            if limit:
                transitions = transitions[-limit:]
            
            return transitions
    
    def get_latest_snapshot(self, player_id: int) -> Optional[StateSnapshot]:
        """Get the most recent snapshot for a player."""
        with self._lock:
            if player_id not in self._history or not self._history[player_id]:
                return None
            
            return self._history[player_id][-1]
    
    def compute_state_diff(
        self, 
        player_id: int, 
        from_snapshot_id: str, 
        to_snapshot_id: str
    ) -> Dict[str, Any]:
        """Compute difference between two state snapshots."""
        with self._lock:
            if player_id not in self._history:
                return {}
            
            snapshots = {s.snapshot_id: s for s in self._history[player_id]}
            
            if from_snapshot_id not in snapshots or to_snapshot_id not in snapshots:
                raise ValueError("Snapshot IDs not found in history")
            
            from_snapshot = snapshots[from_snapshot_id]
            to_snapshot = snapshots[to_snapshot_id]
            
            # Compute state differences
            state_mean_diff = np.array(to_snapshot.state.state_mean) - np.array(from_snapshot.state.state_mean)
            state_cov_diff = np.array(to_snapshot.state.state_cov) - np.array(from_snapshot.state.state_cov)
            
            return {
                "from_snapshot": from_snapshot_id,
                "to_snapshot": to_snapshot_id,
                "time_diff": (to_snapshot.timestamp - from_snapshot.timestamp).total_seconds(),
                "gameweek_diff": to_snapshot.state.gameweek - from_snapshot.state.gameweek,
                "state_mean_diff": state_mean_diff.tolist(),
                "state_cov_diff": state_cov_diff.tolist(),
                "state_mean_magnitude": float(np.linalg.norm(state_mean_diff)),
                "metadata_changes": {
                    "from": from_snapshot.metadata,
                    "to": to_snapshot.metadata,
                }
            }
    
    def clear_history(self, player_id: Optional[int] = None) -> None:
        """Clear history for a specific player or all players."""
        with self._lock:
            if player_id is not None:
                self._history.pop(player_id, None)
                self._transitions.pop(player_id, None)
            else:
                self._history.clear()
                self._transitions.clear()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for the history storage."""
        with self._lock:
            total_snapshots = sum(len(snapshots) for snapshots in self._history.values())
            total_transitions = sum(len(transitions) for transitions in self._transitions.values())
            
            # Estimate memory usage (rough calculation)
            avg_snapshot_size = 1024  # bytes - rough estimate
            avg_transition_size = 512  # bytes - rough estimate
            
            estimated_memory = (
                total_snapshots * avg_snapshot_size + 
                total_transitions * avg_transition_size
            )
            
            return {
                "total_snapshots": total_snapshots,
                "total_transitions": total_transitions,
                "total_players": len(self._history),
                "estimated_memory_bytes": estimated_memory,
                "estimated_memory_mb": estimated_memory / (1024 * 1024),
            }


class StateValidator:
    """
    Validates state consistency and integrity with comprehensive checks.
    
    Provides validation for:
    - State vector dimensions and constraints
    - Covariance matrix properties (positive semi-definite)
    - Temporal consistency across state updates
    - Business logic constraints
    """
    
    def __init__(self, config: StateSpaceConfig):
        """
        Initialize state validator with configuration.
        
        Args:
            config: State space configuration for validation rules
        """
        self.config = config
    
    def validate_state(self, state: PlayerState, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate a player state comprehensively.
        
        Args:
            state: Player state to validate
            strict: Whether to apply strict validation rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic structure validation
        if state.player_id <= 0:
            errors.append("Player ID must be positive")
        
        if state.gameweek <= 0:
            errors.append("Gameweek must be positive")
        
        if not state.season:
            errors.append("Season must be specified")
        
        # State vector validation
        try:
            state_mean = np.array(state.state_mean)
            if state_mean.shape[0] != self.config.state_dim:
                errors.append(f"State mean dimension {state_mean.shape[0]} != expected {self.config.state_dim}")
            
            if not np.isfinite(state_mean).all():
                errors.append("State mean contains non-finite values")
                
            if strict and np.abs(state_mean).max() > 10:
                errors.append("State mean values are unusually large (> 10)")
                
        except Exception as e:
            errors.append(f"Error validating state mean: {e}")
        
        # Covariance matrix validation
        try:
            state_cov = np.array(state.state_cov)
            if state_cov.shape != (self.config.state_dim, self.config.state_dim):
                errors.append(f"State covariance shape {state_cov.shape} != expected {(self.config.state_dim, self.config.state_dim)}")
            
            if not np.isfinite(state_cov).all():
                errors.append("State covariance contains non-finite values")
            
            # Check if covariance is symmetric
            if not np.allclose(state_cov, state_cov.T, rtol=1e-10):
                errors.append("State covariance matrix is not symmetric")
            
            # Check if covariance is positive semi-definite
            eigenvals = np.linalg.eigvals(state_cov)
            if np.any(eigenvals < -1e-10):
                errors.append("State covariance matrix is not positive semi-definite")
            
            if strict and np.trace(state_cov) > 100:
                errors.append("State covariance trace is unusually large (> 100)")
                
        except Exception as e:
            errors.append(f"Error validating state covariance: {e}")
        
        return len(errors) == 0, errors
    
    def validate_state_transition(
        self, 
        previous_state: PlayerState, 
        new_state: PlayerState
    ) -> Tuple[bool, List[str]]:
        """
        Validate a state transition for temporal consistency.
        
        Args:
            previous_state: Previous player state
            new_state: New player state
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check that states are for the same player
        if previous_state.player_id != new_state.player_id:
            errors.append("State transition between different players")
        
        # Check temporal progression
        if new_state.gameweek < previous_state.gameweek:
            errors.append("New state gameweek is before previous state gameweek")
        
        # Check for reasonable state changes
        try:
            prev_mean = np.array(previous_state.state_mean)
            new_mean = np.array(new_state.state_mean)
            
            mean_change = np.abs(new_mean - prev_mean).max()
            if mean_change > 5.0:  # Configurable threshold
                errors.append(f"Large state mean change detected: {mean_change}")
            
            prev_cov = np.array(previous_state.state_cov)
            new_cov = np.array(new_state.state_cov)
            
            # Check for reasonable covariance changes
            cov_change = np.abs(new_cov - prev_cov).max()
            if cov_change > 10.0:  # Configurable threshold
                errors.append(f"Large state covariance change detected: {cov_change}")
                
        except Exception as e:
            errors.append(f"Error validating state transition: {e}")
        
        return len(errors) == 0, errors
    
    def validate_batch_states(
        self, 
        states: List[PlayerState], 
        allow_failures: bool = True
    ) -> Tuple[List[bool], Dict[int, List[str]]]:
        """
        Validate a batch of states efficiently.
        
        Args:
            states: List of player states to validate
            allow_failures: Whether to continue validation after failures
            
        Returns:
            Tuple of (list_of_validity_flags, dict_of_errors_by_index)
        """
        results = []
        all_errors = {}
        
        for i, state in enumerate(states):
            try:
                is_valid, errors = self.validate_state(state)
                results.append(is_valid)
                if not is_valid:
                    all_errors[i] = errors
                    if not allow_failures:
                        break
            except Exception as e:
                results.append(False)
                all_errors[i] = [f"Validation exception: {e}"]
                if not allow_failures:
                    break
        
        return results, all_errors


class StateRepository(ABC):
    """
    Abstract base class for state storage backends.
    
    Defines the interface for state persistence across different storage systems:
    - Memory-based storage for testing and development
    - Redis-based storage for distributed systems
    - Database-based storage for persistence
    """
    
    @abstractmethod
    async def get_state(self, player_id: int) -> Optional[PlayerState]:
        """Get current state for a player."""
        pass
    
    @abstractmethod
    async def set_state(self, player_id: int, state: PlayerState) -> bool:
        """Set current state for a player."""
        pass
    
    @abstractmethod
    async def delete_state(self, player_id: int) -> bool:
        """Delete state for a player."""
        pass
    
    @abstractmethod
    async def get_multiple_states(self, player_ids: List[int]) -> Dict[int, Optional[PlayerState]]:
        """Get states for multiple players efficiently."""
        pass
    
    @abstractmethod
    async def set_multiple_states(self, states: Dict[int, PlayerState]) -> Dict[int, bool]:
        """Set states for multiple players efficiently."""
        pass
    
    @abstractmethod
    async def list_player_ids(self) -> List[int]:
        """List all player IDs with stored states."""
        pass
    
    @abstractmethod
    async def clear_all_states(self) -> bool:
        """Clear all stored states."""
        pass
    
    @abstractmethod
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage backend."""
        pass


class MemoryStateRepository(StateRepository):
    """
    In-memory state repository for testing and development.
    
    Features:
    - Fast access for development and testing
    - Thread-safe operations
    - Optional persistence to disk
    """
    
    def __init__(self):
        """Initialize memory repository."""
        self._states: Dict[int, PlayerState] = {}
        self._lock = asyncio.RLock()
    
    async def get_state(self, player_id: int) -> Optional[PlayerState]:
        """Get current state for a player."""
        async with self._lock:
            state = self._states.get(player_id)
            return state.copy() if state else None
    
    async def set_state(self, player_id: int, state: PlayerState) -> bool:
        """Set current state for a player."""
        async with self._lock:
            self._states[player_id] = state.copy()
            return True
    
    async def delete_state(self, player_id: int) -> bool:
        """Delete state for a player."""
        async with self._lock:
            if player_id in self._states:
                del self._states[player_id]
                return True
            return False
    
    async def get_multiple_states(self, player_ids: List[int]) -> Dict[int, Optional[PlayerState]]:
        """Get states for multiple players efficiently."""
        async with self._lock:
            result = {}
            for player_id in player_ids:
                state = self._states.get(player_id)
                result[player_id] = state.copy() if state else None
            return result
    
    async def set_multiple_states(self, states: Dict[int, PlayerState]) -> Dict[int, bool]:
        """Set states for multiple players efficiently."""
        async with self._lock:
            result = {}
            for player_id, state in states.items():
                self._states[player_id] = state.copy()
                result[player_id] = True
            return result
    
    async def list_player_ids(self) -> List[int]:
        """List all player IDs with stored states."""
        async with self._lock:
            return list(self._states.keys())
    
    async def clear_all_states(self) -> bool:
        """Clear all stored states."""
        async with self._lock:
            self._states.clear()
            return True
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage backend."""
        async with self._lock:
            return {
                "type": "memory",
                "player_count": len(self._states),
                "memory_usage_estimate": len(self._states) * 1024,  # rough estimate
            }


class RedisStateRepository(StateRepository):
    """
    Redis-based state repository for distributed systems.
    
    Features:
    - Distributed state management
    - Integration with existing Redis cache
    - Compression and serialization
    - Atomic operations
    """
    
    def __init__(
        self, 
        redis_cache: RedisCache, 
        key_prefix: str = "state",
        ttl: Optional[int] = None
    ):
        """
        Initialize Redis repository.
        
        Args:
            redis_cache: Redis cache instance
            key_prefix: Prefix for state keys
            ttl: Time-to-live for state entries (None for no expiration)
        """
        self.redis_cache = redis_cache
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._lock = asyncio.Lock()
    
    def _get_state_key(self, player_id: int) -> str:
        """Generate Redis key for player state."""
        return f"{self.key_prefix}:player:{player_id}"
    
    async def get_state(self, player_id: int) -> Optional[PlayerState]:
        """Get current state for a player."""
        key = self._get_state_key(player_id)
        data = self.redis_cache.get(key)
        
        if data is None:
            return None
        
        try:
            return PlayerState.from_dict(data)
        except Exception as e:
            logger.error(f"Error deserializing state for player {player_id}: {e}")
            return None
    
    async def set_state(self, player_id: int, state: PlayerState) -> bool:
        """Set current state for a player."""
        key = self._get_state_key(player_id)
        data = state.to_dict()
        
        return self.redis_cache.set(key, data, self.ttl)
    
    async def delete_state(self, player_id: int) -> bool:
        """Delete state for a player."""
        key = self._get_state_key(player_id)
        return self.redis_cache.delete(key)
    
    async def get_multiple_states(self, player_ids: List[int]) -> Dict[int, Optional[PlayerState]]:
        """Get states for multiple players efficiently."""
        if not player_ids:
            return {}
        
        keys = [self._get_state_key(player_id) for player_id in player_ids]
        raw_data = self.redis_cache.mget(keys)
        
        result = {}
        for i, player_id in enumerate(player_ids):
            if i < len(raw_data) and raw_data[i] is not None:
                try:
                    result[player_id] = PlayerState.from_dict(raw_data[i])
                except Exception as e:
                    logger.error(f"Error deserializing state for player {player_id}: {e}")
                    result[player_id] = None
            else:
                result[player_id] = None
        
        return result
    
    async def set_multiple_states(self, states: Dict[int, PlayerState]) -> Dict[int, bool]:
        """Set states for multiple players efficiently."""
        if not states:
            return {}
        
        # Prepare data for batch set
        key_value_pairs = {}
        for player_id, state in states.items():
            key = self._get_state_key(player_id)
            key_value_pairs[key] = state.to_dict()
        
        # Use batch set operation
        success = self.redis_cache.mset(key_value_pairs, self.ttl)
        
        # Return individual results (Redis mset is atomic, so all succeed or all fail)
        return {player_id: success for player_id in states.keys()}
    
    async def list_player_ids(self) -> List[int]:
        """List all player IDs with stored states."""
        # This is an expensive operation in Redis - use scan for large datasets
        pattern = f"{self.key_prefix}:player:*"
        
        if not self.redis_cache.is_available():
            return []
        
        try:
            with self.redis_cache.connection_manager.get_connection() as conn:
                keys = conn.keys(pattern)
                
                player_ids = []
                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    # Extract player ID from key
                    parts = key_str.split(':')
                    if len(parts) >= 3:
                        try:
                            player_id = int(parts[-1])
                            player_ids.append(player_id)
                        except ValueError:
                            continue
                
                return sorted(player_ids)
                
        except Exception as e:
            logger.error(f"Error listing player IDs from Redis: {e}")
            return []
    
    async def clear_all_states(self) -> bool:
        """Clear all stored states."""
        pattern = f"{self.key_prefix}:player:*"
        deleted_count = self.redis_cache.delete_pattern(pattern)
        return deleted_count > 0
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage backend."""
        info = {
            "type": "redis",
            "available": self.redis_cache.is_available(),
            "key_prefix": self.key_prefix,
            "ttl": self.ttl,
        }
        
        if self.redis_cache.is_available():
            info.update(self.redis_cache.get_metrics())
            
            # Get approximate count of state keys
            try:
                player_ids = await self.list_player_ids()
                info["player_count"] = len(player_ids)
            except Exception as e:
                logger.error(f"Error getting player count: {e}")
                info["player_count"] = "unknown"
        
        return info


class StateManager:
    """
    Main interface for managing all player states with comprehensive features.
    
    Provides:
    - Thread-safe state management
    - Atomic updates with rollback capabilities
    - State history tracking and querying
    - Integration with multiple storage backends
    - Batch operations for efficiency
    - State validation and consistency checks
    """
    
    def __init__(
        self,
        repository: StateRepository,
        config: Optional[StateSpaceConfig] = None,
        enable_history: bool = True,
        max_history_size: int = 1000,
        enable_validation: bool = True,
        auto_snapshot: bool = True,
    ):
        """
        Initialize state manager with configuration.
        
        Args:
            repository: State storage backend
            config: State space configuration for validation
            enable_history: Whether to track state history
            max_history_size: Maximum history entries per player
            enable_validation: Whether to validate states
            auto_snapshot: Whether to automatically create snapshots
        """
        self.repository = repository
        self.config = config or StateSpaceConfig()
        self.enable_history = enable_history
        self.enable_validation = enable_validation
        self.auto_snapshot = auto_snapshot
        
        # Initialize components
        if enable_history:
            self.history = StateHistory(max_history_size)
        else:
            self.history = None
        
        if enable_validation:
            self.validator = StateValidator(self.config)
        else:
            self.validator = None
        
        # Thread safety
        self._lock = asyncio.RLock()
        self._operation_counter = 0
        
        # Metrics
        self._metrics = {
            "operations_count": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "rollbacks_count": 0,
            "validation_errors": 0,
        }
    
    async def get_player_state(self, player_id: int) -> Optional[PlayerState]:
        """
        Get current state for a player.
        
        Args:
            player_id: Player ID to get state for
            
        Returns:
            Player state or None if not found
        """
        return await self.repository.get_state(player_id)
    
    async def update_player_state(
        self,
        player_id: int,
        new_state: PlayerState,
        metadata: Optional[Dict[str, Any]] = None,
        create_snapshot: bool = True,
    ) -> bool:
        """
        Update player state with rollback capability.
        
        Args:
            player_id: Player ID to update
            new_state: New player state
            metadata: Optional metadata for the operation
            create_snapshot: Whether to create a snapshot before update
            
        Returns:
            True if update succeeded, False otherwise
        """
        async with self._lock:
            self._operation_counter += 1
            operation_id = f"update_{self._operation_counter}_{time.time()}"
            
            # Get current state for rollback
            previous_state = await self.repository.get_state(player_id)
            
            # Create transition record
            transition = StateTransition(
                player_id=player_id,
                operation_type=StateOperationType.UPDATE,
                previous_state=previous_state,
                new_state=new_state,
                metadata=metadata or {},
            )
            
            try:
                # Validate new state
                if self.enable_validation and self.validator:
                    is_valid, errors = self.validator.validate_state(new_state)
                    if not is_valid:
                        raise StateValidationError(f"State validation failed: {errors}")
                    
                    # Validate transition if we have previous state
                    if previous_state:
                        is_valid, errors = self.validator.validate_state_transition(previous_state, new_state)
                        if not is_valid:
                            raise StateValidationError(f"State transition validation failed: {errors}")
                
                # Create snapshot before update
                if create_snapshot and self.auto_snapshot and self.history and previous_state:
                    snapshot = StateSnapshot(
                        snapshot_id=f"pre_update_{operation_id}",
                        player_id=player_id,
                        state=previous_state,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"operation_id": operation_id, "type": "pre_update"},
                    )
                    self.history.add_snapshot(snapshot)
                
                # Perform the update
                success = await self.repository.set_state(player_id, new_state)
                
                if not success:
                    raise StateManagerError("Repository update failed")
                
                # Record successful transition
                transition.success = True
                if self.history:
                    self.history.add_transition(transition)
                
                # Create post-update snapshot
                if create_snapshot and self.auto_snapshot and self.history:
                    snapshot = StateSnapshot(
                        snapshot_id=f"post_update_{operation_id}",
                        player_id=player_id,
                        state=new_state,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"operation_id": operation_id, "type": "post_update"},
                    )
                    self.history.add_snapshot(snapshot)
                
                self._metrics["operations_count"] += 1
                self._metrics["successful_operations"] += 1
                
                return True
                
            except Exception as e:
                # Record failed transition
                transition.success = False
                transition.error_message = str(e)
                
                if self.history:
                    self.history.add_transition(transition)
                
                self._metrics["operations_count"] += 1
                self._metrics["failed_operations"] += 1
                
                if isinstance(e, StateValidationError):
                    self._metrics["validation_errors"] += 1
                
                logger.error(f"Failed to update state for player {player_id}: {e}")
                return False
    
    async def rollback_player_state(
        self,
        player_id: int,
        to_snapshot_id: Optional[str] = None,
        to_timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Rollback player state to a previous snapshot.
        
        Args:
            player_id: Player ID to rollback
            to_snapshot_id: Specific snapshot ID to rollback to
            to_timestamp: Timestamp to rollback to (uses nearest snapshot)
            
        Returns:
            True if rollback succeeded, False otherwise
        """
        if not self.history:
            raise RollbackError("History tracking is disabled")
        
        async with self._lock:
            try:
                # Find target snapshot
                target_snapshot = None
                
                if to_snapshot_id:
                    snapshots = self.history.get_snapshots(player_id)
                    target_snapshot = next(
                        (s for s in snapshots if s.snapshot_id == to_snapshot_id), 
                        None
                    )
                elif to_timestamp:
                    snapshots = self.history.get_snapshots(
                        player_id, to_timestamp=to_timestamp
                    )
                    if snapshots:
                        target_snapshot = snapshots[-1]  # Most recent before timestamp
                else:
                    # Rollback to previous snapshot
                    snapshots = self.history.get_snapshots(player_id, limit=2)
                    if len(snapshots) >= 2:
                        target_snapshot = snapshots[-2]  # Second most recent
                
                if not target_snapshot:
                    raise RollbackError("No suitable snapshot found for rollback")
                
                # Verify snapshot integrity
                if not target_snapshot.verify_integrity():
                    raise RollbackError("Target snapshot failed integrity check")
                
                # Perform rollback
                success = await self.repository.set_state(player_id, target_snapshot.state)
                
                if not success:
                    raise RollbackError("Repository rollback failed")
                
                # Record rollback transition
                transition = StateTransition(
                    player_id=player_id,
                    operation_type=StateOperationType.ROLLBACK,
                    new_state=target_snapshot.state,
                    metadata={
                        "target_snapshot_id": target_snapshot.snapshot_id,
                        "rollback_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    success=True,
                )
                
                self.history.add_transition(transition)
                self._metrics["rollbacks_count"] += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to rollback state for player {player_id}: {e}")
                return False
    
    async def initialize_from_historical_data(
        self,
        dbsession: Session,
        season: str,
        max_gameweek: int,
        min_games_threshold: int = 5,
    ) -> Dict[str, Any]:
        """
        Initialize states from historical PlayerScore data.
        
        Args:
            dbsession: Database session
            season: Season to initialize from
            max_gameweek: Maximum gameweek to consider
            min_games_threshold: Minimum games played to create state
            
        Returns:
            Dictionary with initialization results
        """
        logger.info(f"Initializing states from historical data: {season}, GW <= {max_gameweek}")
        
        try:
            # Query historical data
            query = (
                select(PlayerScore)
                .join(Player)
                .where(
                    PlayerScore.fixture.has(season=season),
                    PlayerScore.fixture.has(gameweek<=max_gameweek),
                )
                .order_by(PlayerScore.player_id, PlayerScore.fixture_id)
            )
            
            results = dbsession.execute(query).scalars().all()
            
            # Group by player
            player_data = {}
            for score in results:
                if score.player_id not in player_data:
                    player_data[score.player_id] = []
                player_data[score.player_id].append(score)
            
            # Initialize states
            initialized_count = 0
            skipped_count = 0
            error_count = 0
            
            for player_id, scores in player_data.items():
                if len(scores) < min_games_threshold:
                    skipped_count += 1
                    continue
                
                try:
                    # Compute initial state from historical performance
                    observations = np.array([
                        [score.goals, score.assists, score.minutes, score.bonus]
                        for score in scores
                    ])
                    
                    # Simple initialization: mean performance as state
                    mean_performance = np.mean(observations, axis=0)
                    state_cov = np.cov(observations.T) + np.eye(len(mean_performance)) * 0.01
                    
                    # Create initial state
                    initial_state = PlayerState(
                        player_id=player_id,
                        state_mean=mean_performance,
                        state_cov=state_cov,
                        gameweek=max_gameweek,
                        season=season,
                        last_updated=datetime.now(timezone.utc).isoformat(),
                    )
                    
                    # Store state
                    success = await self.repository.set_state(player_id, initial_state)
                    
                    if success:
                        # Create initial snapshot
                        if self.history:
                            snapshot = StateSnapshot(
                                snapshot_id=f"init_{player_id}_{season}",
                                player_id=player_id,
                                state=initial_state,
                                timestamp=datetime.now(timezone.utc),
                                metadata={
                                    "type": "initialization",
                                    "season": season,
                                    "games_count": len(scores),
                                },
                            )
                            self.history.add_snapshot(snapshot)
                        
                        initialized_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Error initializing state for player {player_id}: {e}")
                    error_count += 1
            
            results = {
                "initialized_count": initialized_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "total_players": len(player_data),
                "season": season,
                "max_gameweek": max_gameweek,
            }
            
            logger.info(f"State initialization completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to initialize states from historical data: {e}")
            raise StateManagerError(f"Initialization failed: {e}")
    
    async def get_state_history(
        self,
        player_id: int,
        from_gameweek: Optional[int] = None,
        to_gameweek: Optional[int] = None,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[StateSnapshot]:
        """Get state history for a player with filtering."""
        if not self.history:
            return []
        
        return self.history.get_snapshots(
            player_id=player_id,
            from_gameweek=from_gameweek,
            to_gameweek=to_gameweek,
            season=season,
            limit=limit,
        )
    
    async def get_multiple_states(self, player_ids: List[int]) -> Dict[int, Optional[PlayerState]]:
        """Get states for multiple players efficiently."""
        return await self.repository.get_multiple_states(player_ids)
    
    async def create_state_snapshot(
        self,
        player_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a manual state snapshot."""
        if not self.history:
            return None
        
        current_state = await self.repository.get_state(player_id)
        if not current_state:
            return None
        
        snapshot = StateSnapshot(
            snapshot_id=f"manual_{player_id}_{int(time.time())}",
            player_id=player_id,
            state=current_state,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        
        self.history.add_snapshot(snapshot)
        return snapshot.snapshot_id
    
    async def get_manager_metrics(self) -> Dict[str, Any]:
        """Get comprehensive state manager metrics."""
        metrics = self._metrics.copy()
        
        # Add repository info
        metrics["repository"] = await self.repository.get_storage_info()
        
        # Add history info
        if self.history:
            metrics["history"] = self.history.get_memory_usage()
        
        # Add configuration info
        metrics["config"] = {
            "enable_history": self.enable_history,
            "enable_validation": self.enable_validation,
            "auto_snapshot": self.auto_snapshot,
            "state_dim": self.config.state_dim,
            "obs_dim": self.config.obs_dim,
        }
        
        return metrics
    
    async def cleanup_old_history(
        self,
        older_than_days: int = 30,
        keep_snapshots_per_player: int = 10,
    ) -> int:
        """Clean up old history entries to manage memory."""
        if not self.history:
            return 0
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cleaned_count = 0
        
        # Get all player IDs
        player_ids = await self.repository.list_player_ids()
        
        for player_id in player_ids:
            snapshots = self.history.get_snapshots(player_id)
            
            # Keep recent snapshots and a minimum number
            snapshots_to_keep = []
            snapshots_to_remove = []
            
            # Sort by timestamp (newest first)
            snapshots.sort(key=lambda s: s.timestamp, reverse=True)
            
            for i, snapshot in enumerate(snapshots):
                if i < keep_snapshots_per_player or snapshot.timestamp >= cutoff_time:
                    snapshots_to_keep.append(snapshot)
                else:
                    snapshots_to_remove.append(snapshot)
            
            # Update history for this player
            if snapshots_to_remove:
                self.history._history[player_id] = snapshots_to_keep
                cleaned_count += len(snapshots_to_remove)
        
        logger.info(f"Cleaned up {cleaned_count} old history entries")
        return cleaned_count
    
    async def validate_all_states(self) -> Dict[str, Any]:
        """Validate all stored states and return summary."""
        if not self.validator:
            return {"error": "Validation is disabled"}
        
        player_ids = await self.repository.list_player_ids()
        states = await self.repository.get_multiple_states(player_ids)
        
        valid_count = 0
        invalid_count = 0
        error_details = {}
        
        for player_id, state in states.items():
            if state is None:
                continue
            
            is_valid, errors = self.validator.validate_state(state)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                error_details[player_id] = errors
        
        return {
            "total_states": len([s for s in states.values() if s is not None]),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "error_details": error_details,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        }