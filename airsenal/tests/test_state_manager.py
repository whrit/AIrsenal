"""
Comprehensive tests for the Model State Management System.

Test Coverage:
- StateSnapshot creation and integrity verification
- StateTransition recording and serialization
- StateHistory tracking and querying
- StateValidator validation rules and error handling
- StateRepository implementations (Memory and Redis)
- StateManager core functionality and edge cases
- Concurrent state updates and thread safety
- Rollback mechanism and error recovery
- State initialization from historical data
- Integration with AdaptivePlayerModel
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.adaptive_player_model import PlayerState, StateSpaceConfig
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.state_manager import (
    MemoryStateRepository,
    RedisStateRepository,
    StateHistory,
    StateManager,
    StateManagerError,
    StateOperationType,
    StateSnapshot,
    StateTransition,
    StateValidationError,
    StateValidator,
)


class TestStateSnapshot:
    """Test StateSnapshot functionality."""
    
    def test_state_snapshot_creation(self):
        """Test basic state snapshot creation."""
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        snapshot = StateSnapshot(
            snapshot_id="test_snapshot_1",
            player_id=123,
            state=state,
            timestamp=datetime.now(timezone.utc),
            metadata={"test": "value"},
        )
        
        assert snapshot.snapshot_id == "test_snapshot_1"
        assert snapshot.player_id == 123
        assert snapshot.state.player_id == 123
        assert "test" in snapshot.metadata
        assert isinstance(snapshot.checksum, str)
        assert len(snapshot.checksum) == 16  # SHA256 truncated to 16 chars
    
    def test_state_snapshot_serialization(self):
        """Test snapshot serialization and deserialization."""
        state = PlayerState(
            player_id=456,
            state_mean=np.array([4.0, 5.0, 6.0]),
            state_cov=np.eye(3) * 2,
            gameweek=15,
            season="2023",
        )
        
        snapshot = StateSnapshot(
            snapshot_id="test_snapshot_2",
            player_id=456,
            state=state,
            timestamp=datetime.now(timezone.utc),
            metadata={"source": "test"},
        )
        
        # Test to_dict
        snapshot_dict = snapshot.to_dict()
        assert "snapshot_id" in snapshot_dict
        assert "checksum" in snapshot_dict
        assert snapshot_dict["player_id"] == 456
        
        # Test from_dict
        restored_snapshot = StateSnapshot.from_dict(snapshot_dict)
        assert restored_snapshot.snapshot_id == snapshot.snapshot_id
        assert restored_snapshot.player_id == snapshot.player_id
        assert restored_snapshot.checksum == snapshot.checksum
        np.testing.assert_array_equal(
            restored_snapshot.state.state_mean, snapshot.state.state_mean
        )
    
    def test_state_snapshot_integrity_verification(self):
        """Test snapshot integrity verification."""
        state = PlayerState(
            player_id=789,
            state_mean=np.array([7.0, 8.0, 9.0]),
            state_cov=np.eye(3) * 3,
            gameweek=20,
            season="2023",
        )
        
        snapshot = StateSnapshot(
            snapshot_id="test_snapshot_3",
            player_id=789,
            state=state,
            timestamp=datetime.now(timezone.utc),
        )
        
        # Should verify correctly
        assert snapshot.verify_integrity()
        
        # Test with corrupted data
        corrupted_snapshot = StateSnapshot(
            snapshot_id=snapshot.snapshot_id,
            player_id=999,  # Changed player ID
            state=snapshot.state,
            timestamp=snapshot.timestamp,
        )
        
        # Should fail verification due to different checksum
        assert not corrupted_snapshot.verify_integrity()


class TestStateTransition:
    """Test StateTransition functionality."""
    
    def test_state_transition_creation(self):
        """Test basic state transition creation."""
        previous_state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        new_state = PlayerState(
            player_id=123,
            state_mean=np.array([1.1, 2.1, 3.1]),
            state_cov=np.eye(3) * 1.1,
            gameweek=11,
            season="2023",
        )
        
        transition = StateTransition(
            player_id=123,
            operation_type=StateOperationType.UPDATE,
            previous_state=previous_state,
            new_state=new_state,
            metadata={"confidence": 0.95},
        )
        
        assert transition.player_id == 123
        assert transition.operation_type == StateOperationType.UPDATE
        assert transition.previous_state is not None
        assert transition.new_state is not None
        assert transition.success is True
        assert "confidence" in transition.metadata
    
    def test_state_transition_serialization(self):
        """Test transition serialization and deserialization."""
        transition = StateTransition(
            player_id=456,
            operation_type=StateOperationType.ROLLBACK,
            success=False,
            error_message="Test error",
            metadata={"reason": "validation_failed"},
        )
        
        # Test to_dict
        transition_dict = transition.to_dict()
        assert transition_dict["player_id"] == 456
        assert transition_dict["operation_type"] == "rollback"
        assert transition_dict["success"] is False
        assert transition_dict["error_message"] == "Test error"
        
        # Test from_dict
        restored_transition = StateTransition.from_dict(transition_dict)
        assert restored_transition.player_id == transition.player_id
        assert restored_transition.operation_type == transition.operation_type
        assert restored_transition.success == transition.success
        assert restored_transition.error_message == transition.error_message


class TestStateHistory:
    """Test StateHistory functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.history = StateHistory(max_history_size=10)
        self.player_id = 123
        
        # Create test snapshots
        self.snapshots = []
        for i in range(5):
            state = PlayerState(
                player_id=self.player_id,
                state_mean=np.array([float(i), float(i+1), float(i+2)]),
                state_cov=np.eye(3) * (i + 1),
                gameweek=10 + i,
                season="2023",
            )
            
            snapshot = StateSnapshot(
                snapshot_id=f"snapshot_{i}",
                player_id=self.player_id,
                state=state,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                metadata={"index": i},
            )
            
            self.snapshots.append(snapshot)
            self.history.add_snapshot(snapshot)
    
    def test_add_and_retrieve_snapshots(self):
        """Test adding and retrieving snapshots."""
        snapshots = self.history.get_snapshots(self.player_id)
        assert len(snapshots) == 5
        
        # Check ordering (should be sorted by timestamp)
        for i in range(len(snapshots) - 1):
            assert snapshots[i].timestamp <= snapshots[i + 1].timestamp
    
    def test_snapshot_filtering(self):
        """Test snapshot filtering by various criteria."""
        # Filter by gameweek range
        snapshots = self.history.get_snapshots(
            self.player_id, from_gameweek=12, to_gameweek=13
        )
        assert len(snapshots) == 2
        assert all(12 <= s.state.gameweek <= 13 for s in snapshots)
        
        # Filter by season
        snapshots = self.history.get_snapshots(self.player_id, season="2023")
        assert len(snapshots) == 5
        
        # Filter with limit
        snapshots = self.history.get_snapshots(self.player_id, limit=3)
        assert len(snapshots) == 3
    
    def test_latest_snapshot(self):
        """Test getting the latest snapshot."""
        latest = self.history.get_latest_snapshot(self.player_id)
        assert latest is not None
        assert latest.metadata["index"] == 4  # Last added
    
    def test_state_diff_computation(self):
        """Test state difference computation."""
        diff = self.history.compute_state_diff(
            self.player_id, "snapshot_0", "snapshot_4"
        )
        
        assert "state_mean_diff" in diff
        assert "state_cov_diff" in diff
        assert "time_diff" in diff
        assert "gameweek_diff" in diff
        assert diff["gameweek_diff"] == 4
        
        # Check magnitude calculation
        assert "state_mean_magnitude" in diff
        assert isinstance(diff["state_mean_magnitude"], float)
    
    def test_history_size_limit(self):
        """Test that history respects size limits."""
        history = StateHistory(max_history_size=3)
        
        # Add more snapshots than the limit
        for i in range(5):
            state = PlayerState(
                player_id=999,
                state_mean=np.array([float(i)]),
                state_cov=np.array([[1.0]]),
                gameweek=i,
                season="2023",
            )
            
            snapshot = StateSnapshot(
                snapshot_id=f"test_{i}",
                player_id=999,
                state=state,
                timestamp=datetime.now(timezone.utc),
            )
            
            history.add_snapshot(snapshot)
        
        # Should only keep the last 3
        snapshots = history.get_snapshots(999)
        assert len(snapshots) == 3
        assert snapshots[0].snapshot_id == "test_2"  # Should start from index 2
    
    def test_memory_usage_calculation(self):
        """Test memory usage estimation."""
        usage = self.history.get_memory_usage()
        
        assert "total_snapshots" in usage
        assert "total_transitions" in usage
        assert "total_players" in usage
        assert "estimated_memory_bytes" in usage
        assert usage["total_snapshots"] == 5
        assert usage["total_players"] == 1


class TestStateValidator:
    """Test StateValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StateSpaceConfig(state_dim=3, obs_dim=4)
        self.validator = StateValidator(self.config)
    
    def test_valid_state_validation(self):
        """Test validation of a valid state."""
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        is_valid, errors = self.validator.validate_state(state)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_state_validation(self):
        """Test validation of invalid states."""
        # Test with wrong state dimension
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0]),  # Wrong dimension
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        is_valid, errors = self.validator.validate_state(state)
        assert not is_valid
        assert any("dimension" in error.lower() for error in errors)
        
        # Test with non-finite values
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, np.inf, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        is_valid, errors = self.validator.validate_state(state)
        assert not is_valid
        assert any("finite" in error.lower() for error in errors)
        
        # Test with non-positive definite covariance
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            gameweek=10,
            season="2023",
        )
        
        is_valid, errors = self.validator.validate_state(state)
        assert not is_valid
        assert any("positive semi-definite" in error.lower() for error in errors)
    
    def test_state_transition_validation(self):
        """Test validation of state transitions."""
        previous_state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Valid transition
        new_state = PlayerState(
            player_id=123,
            state_mean=np.array([1.1, 2.1, 3.1]),
            state_cov=np.eye(3) * 1.1,
            gameweek=11,
            season="2023",
        )
        
        is_valid, errors = self.validator.validate_state_transition(previous_state, new_state)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid transition (different players)
        new_state.player_id = 456
        is_valid, errors = self.validator.validate_state_transition(previous_state, new_state)
        assert not is_valid
        assert any("different players" in error.lower() for error in errors)
        
        # Invalid transition (gameweek going backwards)
        new_state.player_id = 123
        new_state.gameweek = 9
        is_valid, errors = self.validator.validate_state_transition(previous_state, new_state)
        assert not is_valid
        assert any("before previous" in error.lower() for error in errors)
    
    def test_batch_validation(self):
        """Test batch validation of multiple states."""
        states = []
        
        # Create mix of valid and invalid states
        for i in range(5):
            if i == 2:
                # Invalid state with wrong dimension
                state = PlayerState(
                    player_id=i,
                    state_mean=np.array([1.0, 2.0]),  # Wrong dimension
                    state_cov=np.eye(3),
                    gameweek=10,
                    season="2023",
                )
            else:
                # Valid state
                state = PlayerState(
                    player_id=i,
                    state_mean=np.array([1.0, 2.0, 3.0]),
                    state_cov=np.eye(3),
                    gameweek=10,
                    season="2023",
                )
            states.append(state)
        
        results, errors = self.validator.validate_batch_states(states)
        
        assert len(results) == 5
        assert results[0] is True  # Valid
        assert results[1] is True  # Valid
        assert results[2] is False  # Invalid
        assert results[3] is True  # Valid
        assert results[4] is True  # Valid
        
        assert 2 in errors  # Index 2 should have errors
        assert len(errors[2]) > 0


class TestMemoryStateRepository:
    """Test MemoryStateRepository functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """Test basic repository operations."""
        repo = MemoryStateRepository()
        
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Test set and get
        success = await repo.set_state(123, state)
        assert success
        
        retrieved_state = await repo.get_state(123)
        assert retrieved_state is not None
        assert retrieved_state.player_id == 123
        np.testing.assert_array_equal(retrieved_state.state_mean, state.state_mean)
        
        # Test non-existent player
        retrieved_state = await repo.get_state(999)
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch operations."""
        repo = MemoryStateRepository()
        
        # Create multiple states
        states = {}
        for i in range(5):
            states[i] = PlayerState(
                player_id=i,
                state_mean=np.array([float(i), float(i+1), float(i+2)]),
                state_cov=np.eye(3) * (i + 1),
                gameweek=10,
                season="2023",
            )
        
        # Test batch set
        results = await repo.set_multiple_states(states)
        assert all(results.values())
        
        # Test batch get
        retrieved_states = await repo.get_multiple_states(list(states.keys()))
        assert len(retrieved_states) == 5
        
        for player_id, state in retrieved_states.items():
            assert state is not None
            assert state.player_id == player_id
    
    @pytest.mark.asyncio
    async def test_delete_operations(self):
        """Test delete operations."""
        repo = MemoryStateRepository()
        
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Set and then delete
        await repo.set_state(123, state)
        success = await repo.delete_state(123)
        assert success
        
        # Should be gone
        retrieved_state = await repo.get_state(123)
        assert retrieved_state is None
        
        # Delete non-existent
        success = await repo.delete_state(999)
        assert not success
    
    @pytest.mark.asyncio
    async def test_list_and_clear(self):
        """Test listing and clearing operations."""
        repo = MemoryStateRepository()
        
        # Add some states
        for i in range(3):
            state = PlayerState(
                player_id=i,
                state_mean=np.array([float(i)]),
                state_cov=np.array([[1.0]]),
                gameweek=10,
                season="2023",
            )
            await repo.set_state(i, state)
        
        # Test list
        player_ids = await repo.list_player_ids()
        assert len(player_ids) == 3
        assert set(player_ids) == {0, 1, 2}
        
        # Test clear
        success = await repo.clear_all_states()
        assert success
        
        player_ids = await repo.list_player_ids()
        assert len(player_ids) == 0


@pytest.mark.skipif(
    not hasattr(pytest, 'redis_available'), 
    reason="Redis not available for testing"
)
class TestRedisStateRepository:
    """Test RedisStateRepository functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Redis cache for testing
        self.mock_redis = MagicMock()
        self.repo = RedisStateRepository(
            redis_cache=self.mock_redis,
            key_prefix="test_state",
            ttl=3600
        )
    
    @pytest.mark.asyncio
    async def test_key_generation(self):
        """Test Redis key generation."""
        key = self.repo._get_state_key(123)
        assert key == "test_state:player:123"
    
    @pytest.mark.asyncio
    async def test_state_operations_with_mock(self):
        """Test state operations with mocked Redis."""
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Mock successful set
        self.mock_redis.set.return_value = True
        success = await self.repo.set_state(123, state)
        assert success
        
        # Mock successful get
        self.mock_redis.get.return_value = state.to_dict()
        retrieved_state = await self.repo.get_state(123)
        assert retrieved_state is not None
        assert retrieved_state.player_id == 123
        
        # Mock get for non-existent state
        self.mock_redis.get.return_value = None
        retrieved_state = await self.repo.get_state(999)
        assert retrieved_state is None


class TestStateManager:
    """Test StateManager functionality."""
    
    @pytest.mark.asyncio
    async def setup_method(self):
        """Set up test fixtures."""
        self.config = StateSpaceConfig(state_dim=3, obs_dim=4)
        self.repository = MemoryStateRepository()
        self.state_manager = StateManager(
            repository=self.repository,
            config=self.config,
            enable_history=True,
            enable_validation=True,
            auto_snapshot=True,
        )
    
    @pytest.mark.asyncio
    async def test_basic_state_operations(self):
        """Test basic state management operations."""
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Test update
        success = await self.state_manager.update_player_state(123, state)
        assert success
        
        # Test get
        retrieved_state = await self.state_manager.get_player_state(123)
        assert retrieved_state is not None
        assert retrieved_state.player_id == 123
    
    @pytest.mark.asyncio
    async def test_state_validation_integration(self):
        """Test state validation integration."""
        # Invalid state (wrong dimension)
        invalid_state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0]),  # Wrong dimension
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        # Should fail validation
        success = await self.state_manager.update_player_state(123, invalid_state)
        assert not success
        
        # Check that state was not stored
        retrieved_state = await self.state_manager.get_player_state(123)
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_history_tracking(self):
        """Test state history tracking."""
        state1 = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        state2 = PlayerState(
            player_id=123,
            state_mean=np.array([1.1, 2.1, 3.1]),
            state_cov=np.eye(3) * 1.1,
            gameweek=11,
            season="2023",
        )
        
        # Update states
        await self.state_manager.update_player_state(123, state1)
        await self.state_manager.update_player_state(123, state2)
        
        # Check history
        history = await self.state_manager.get_state_history(123)
        assert len(history) >= 2  # Should have snapshots from updates
    
    @pytest.mark.asyncio
    async def test_rollback_functionality(self):
        """Test state rollback functionality."""
        state1 = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        state2 = PlayerState(
            player_id=123,
            state_mean=np.array([2.0, 3.0, 4.0]),
            state_cov=np.eye(3) * 2,
            gameweek=11,
            season="2023",
        )
        
        # Update states
        await self.state_manager.update_player_state(123, state1)
        snapshot_id = await self.state_manager.create_state_snapshot(123)
        await self.state_manager.update_player_state(123, state2)
        
        # Current state should be state2
        current_state = await self.state_manager.get_player_state(123)
        np.testing.assert_array_almost_equal(current_state.state_mean, state2.state_mean)
        
        # Rollback to snapshot
        success = await self.state_manager.rollback_player_state(123, snapshot_id)
        assert success
        
        # Should be back to state1
        rolled_back_state = await self.state_manager.get_player_state(123)
        np.testing.assert_array_almost_equal(rolled_back_state.state_mean, state1.state_mean)
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test concurrent state updates."""
        async def update_state(player_id, value):
            state = PlayerState(
                player_id=player_id,
                state_mean=np.array([value, value + 1, value + 2]),
                state_cov=np.eye(3),
                gameweek=10,
                season="2023",
            )
            return await self.state_manager.update_player_state(player_id, state)
        
        # Run concurrent updates
        tasks = [update_state(i, float(i)) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        
        # Check that all states were stored
        for i in range(10):
            state = await self.state_manager.get_player_state(i)
            assert state is not None
            assert state.player_id == i
    
    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch state operations."""
        player_ids = [1, 2, 3, 4, 5]
        
        # Create states for batch test
        states = {}
        for player_id in player_ids:
            states[player_id] = PlayerState(
                player_id=player_id,
                state_mean=np.array([float(player_id), float(player_id + 1), float(player_id + 2)]),
                state_cov=np.eye(3),
                gameweek=10,
                season="2023",
            )
            # Add states individually first
            await self.state_manager.update_player_state(player_id, states[player_id])
        
        # Test batch get
        batch_states = await self.state_manager.get_multiple_states(player_ids)
        assert len(batch_states) == len(player_ids)
        
        for player_id in player_ids:
            assert batch_states[player_id] is not None
            assert batch_states[player_id].player_id == player_id
    
    @pytest.mark.asyncio
    async def test_manager_metrics(self):
        """Test state manager metrics collection."""
        # Perform some operations to generate metrics
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.eye(3),
            gameweek=10,
            season="2023",
        )
        
        await self.state_manager.update_player_state(123, state)
        
        metrics = await self.state_manager.get_manager_metrics()
        
        assert "operations_count" in metrics
        assert "successful_operations" in metrics
        assert "repository" in metrics
        assert "config" in metrics
        
        if self.state_manager.enable_history:
            assert "history" in metrics
    
    @pytest.mark.asyncio
    async def test_state_validation_comprehensive(self):
        """Test comprehensive state validation."""
        # Add some states
        for i in range(3):
            state = PlayerState(
                player_id=i,
                state_mean=np.array([float(i), float(i+1), float(i+2)]),
                state_cov=np.eye(3),
                gameweek=10,
                season="2023",
            )
            await self.state_manager.update_player_state(i, state)
        
        validation_results = await self.state_manager.validate_all_states()
        
        assert "total_states" in validation_results
        assert "valid_count" in validation_results
        assert "invalid_count" in validation_results
        assert validation_results["total_states"] == 3
        assert validation_results["valid_count"] == 3
        assert validation_results["invalid_count"] == 0


class TestStateManagerIntegration:
    """Test StateManager integration with external systems."""
    
    @pytest.mark.asyncio
    async def test_initialization_from_mock_data(self):
        """Test state initialization from mock historical data."""
        # Create mock database session and data
        mock_session = MagicMock()
        
        # Mock PlayerScore data
        mock_scores = []
        for i in range(10):  # 10 games for player 123
            mock_score = MagicMock()
            mock_score.player_id = 123
            mock_score.goals = i % 3  # Varying goals
            mock_score.assists = i % 2  # Varying assists
            mock_score.minutes = 90 if i % 4 != 0 else 0  # Mostly full games
            mock_score.bonus = i % 4  # Varying bonus
            mock_scores.append(mock_score)
        
        # Mock database query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_scores
        mock_session.execute.return_value = mock_result
        
        # Create state manager
        repository = MemoryStateRepository()
        state_manager = StateManager(
            repository=repository,
            enable_history=True,
            enable_validation=True,
        )
        
        # Test initialization
        results = await state_manager.initialize_from_historical_data(
            dbsession=mock_session,
            season="2023",
            max_gameweek=15,
            min_games_threshold=5,
        )
        
        assert results["initialized_count"] == 1  # One player initialized
        assert results["total_players"] == 1
        
        # Check that state was created
        state = await state_manager.get_player_state(123)
        assert state is not None
        assert state.player_id == 123
        assert state.season == "2023"
        assert state.gameweek == 15


# Utility functions for testing
def create_test_state(player_id: int, gameweek: int = 10, season: str = "2023") -> PlayerState:
    """Create a test PlayerState."""
    return PlayerState(
        player_id=player_id,
        state_mean=np.random.randn(3),
        state_cov=np.eye(3) + np.random.randn(3, 3) * 0.1,
        gameweek=gameweek,
        season=season,
    )


def create_test_config() -> StateSpaceConfig:
    """Create a test StateSpaceConfig."""
    return StateSpaceConfig(
        state_dim=3,
        obs_dim=4,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
    )


# Performance and stress tests
class TestStateManagerPerformance:
    """Performance and stress tests for StateManager."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_scale_operations(self):
        """Test state manager with large number of players."""
        repository = MemoryStateRepository()
        state_manager = StateManager(
            repository=repository,
            enable_history=True,
            enable_validation=True,
        )
        
        # Create states for 1000 players
        n_players = 1000
        start_time = time.time()
        
        for player_id in range(n_players):
            state = create_test_state(player_id)
            await state_manager.update_player_state(player_id, state)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"Created {n_players} states in {elapsed:.2f} seconds")
        print(f"Rate: {n_players / elapsed:.2f} states/second")
        
        # Verify all states were created
        metrics = await state_manager.get_manager_metrics()
        assert metrics["repository"]["player_count"] == n_players
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_history_memory_usage(self):
        """Test memory usage with extensive history."""
        repository = MemoryStateRepository()
        state_manager = StateManager(
            repository=repository,
            enable_history=True,
            max_history_size=100,
        )
        
        player_id = 123
        
        # Generate many state updates
        for i in range(200):  # More than max_history_size
            state = PlayerState(
                player_id=player_id,
                state_mean=np.random.randn(3),
                state_cov=np.eye(3),
                gameweek=i,
                season="2023",
            )
            await state_manager.update_player_state(player_id, state)
        
        # Check that history size is limited
        history = await state_manager.get_state_history(player_id)
        assert len(history) <= 100  # Should be limited by max_history_size
        
        # Check memory usage
        metrics = await state_manager.get_manager_metrics()
        if "history" in metrics:
            memory_mb = metrics["history"]["estimated_memory_mb"]
            print(f"Estimated history memory usage: {memory_mb:.2f} MB")
            assert memory_mb < 100  # Should be reasonable


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "-x"])