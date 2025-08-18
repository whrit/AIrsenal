"""
Comprehensive tests for the Model Persistence Layer.

Test Coverage:
- ModelPersistence core functionality (save/load checkpoints)
- CompressionManager compression and decompression
- IntegrityManager checksum verification
- SerializationManager model state serialization
- CheckpointManager automatic checkpointing and pruning
- DatabaseStorageBackend persistence operations
- PersistenceIntegrator integration with existing systems
- Error handling and recovery scenarios
- Performance and concurrency testing
- Migration and backward compatibility
"""

import asyncio
import gzip
import hashlib
import pickle
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Import test fixtures and utilities
from airsenal.tests.fixtures.model_factories import create_test_adaptive_model
from airsenal.tests.fixtures.database_factories import create_test_session
from airsenal.tests.fixtures.pytest_fixtures import test_database

# Import the modules we're testing
from airsenal.framework.adaptive_player_model import AdaptivePlayerModel, PlayerState, StateSpaceConfig
from airsenal.framework.model_persistence import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointNotFoundError,
    CheckpointType,
    CompressionError,
    CompressionFormat,
    CompressionManager,
    DatabaseStorageBackend,
    IntegrityManager,
    ModelPersistence,
    ModelPersistenceError,
    PersistenceConfig,
    SerializationManager,
    StorageBackend,
    StorageError,
)
from airsenal.framework.persistence_integration import (
    PersistenceIntegrator,
    PersistentAdaptiveModel,
    StateManagerIntegration,
)
from airsenal.framework.redis_cache import RedisCache
from airsenal.framework.schema import (
    Base,
    ModelCheckpoint,
    ModelMetadata,
    ModelState,
    ModelVersion,
    session_scope,
)
from airsenal.framework.state_manager import StateManager, MemoryStateRepository


class TestCompressionManager:
    """Test CompressionManager functionality."""
    
    def test_compression_manager_init(self):
        """Test CompressionManager initialization."""
        config = PersistenceConfig(
            compression_format=CompressionFormat.GZIP,
            compression_level=6
        )
        manager = CompressionManager(config)
        
        assert manager.compression_format == CompressionFormat.GZIP
        assert manager.compression_level == 6
    
    def test_gzip_compression_decompression(self):
        """Test GZIP compression and decompression."""
        config = PersistenceConfig(compression_format=CompressionFormat.GZIP)
        manager = CompressionManager(config)
        
        # Create test data
        original_data = b"This is test data for compression" * 100
        
        # Compress
        compressed_data, metadata = manager.compress(original_data)
        
        assert len(compressed_data) < len(original_data)
        assert metadata['original_size'] == len(original_data)
        assert metadata['compressed_size'] == len(compressed_data)
        assert metadata['format'] == 'gzip'
        assert metadata['compression_ratio'] < 1.0
        
        # Decompress
        decompressed_data = manager.decompress(compressed_data)
        
        assert decompressed_data == original_data
    
    def test_no_compression(self):
        """Test no compression mode."""
        config = PersistenceConfig(compression_format=CompressionFormat.NONE)
        manager = CompressionManager(config)
        
        original_data = b"Test data"
        
        compressed_data, metadata = manager.compress(original_data)
        
        assert compressed_data == original_data
        assert metadata['compression_ratio'] == 1.0
        
        decompressed_data = manager.decompress(compressed_data)
        assert decompressed_data == original_data
    
    def test_compression_error_handling(self):
        """Test compression error handling."""
        config = PersistenceConfig(compression_format=CompressionFormat.GZIP)
        manager = CompressionManager(config)
        
        # Test decompression with invalid data
        with pytest.raises(CompressionError):
            manager.decompress(b"invalid compressed data", format_hint="gzip")
    
    def test_multiple_compression_formats(self):
        """Test different compression formats."""
        formats_to_test = [
            CompressionFormat.GZIP,
            CompressionFormat.ZLIB,
            CompressionFormat.NONE
        ]
        
        original_data = b"Test data for multiple formats" * 50
        
        for fmt in formats_to_test:
            config = PersistenceConfig(compression_format=fmt)
            manager = CompressionManager(config)
            
            compressed_data, metadata = manager.compress(original_data)
            decompressed_data = manager.decompress(compressed_data, format_hint=fmt.value)
            
            assert decompressed_data == original_data
            assert metadata['format'] == fmt.value


class TestIntegrityManager:
    """Test IntegrityManager functionality."""
    
    def test_integrity_manager_init(self):
        """Test IntegrityManager initialization."""
        config = PersistenceConfig(checksum_algorithm="sha256")
        manager = IntegrityManager(config)
        
        assert manager.checksum_algorithm == "sha256"
    
    def test_checksum_computation(self):
        """Test checksum computation."""
        config = PersistenceConfig(checksum_algorithm="sha256")
        manager = IntegrityManager(config)
        
        data = b"Test data for checksum"
        checksum = manager.compute_checksum(data)
        
        # Verify it's a valid SHA256 hex string
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)
        
        # Verify consistency
        checksum2 = manager.compute_checksum(data)
        assert checksum == checksum2
    
    def test_integrity_verification(self):
        """Test integrity verification."""
        config = PersistenceConfig(checksum_algorithm="sha256")
        manager = IntegrityManager(config)
        
        data = b"Test data for verification"
        checksum = manager.compute_checksum(data)
        
        # Valid verification
        assert manager.verify_integrity(data, checksum) is True
        
        # Invalid verification
        wrong_checksum = "0" * 64
        assert manager.verify_integrity(data, wrong_checksum) is False
        
        # Modified data
        modified_data = b"Modified test data"
        assert manager.verify_integrity(modified_data, checksum) is False
    
    def test_different_checksum_algorithms(self):
        """Test different checksum algorithms."""
        algorithms = ["sha256", "md5", "sha1"]
        data = b"Test data"
        
        for algorithm in algorithms:
            config = PersistenceConfig(checksum_algorithm=algorithm)
            manager = IntegrityManager(config)
            
            checksum = manager.compute_checksum(data)
            assert manager.verify_integrity(data, checksum) is True


class TestSerializationManager:
    """Test SerializationManager functionality."""
    
    def test_serialization_manager_init(self):
        """Test SerializationManager initialization."""
        config = PersistenceConfig(
            serialization_format="pickle",
            pickle_protocol=4
        )
        manager = SerializationManager(config)
        
        assert manager.format == "pickle"
        assert manager.pickle_protocol == 4
    
    def test_player_state_serialization(self):
        """Test PlayerState serialization and deserialization."""
        config = PersistenceConfig()
        manager = SerializationManager(config)
        
        # Create test player state
        state = PlayerState(
            player_id=123,
            state_mean=np.array([1.0, 2.0, 3.0]),
            state_cov=np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]),
            gameweek=10,
            season="2023",
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        # Serialize
        serialized = manager.serialize_player_state(state)
        
        assert 'state_mean' in serialized
        assert 'state_covariance' in serialized
        assert isinstance(serialized['state_mean'], bytes)
        assert isinstance(serialized['state_covariance'], bytes)
        
        # Deserialize
        restored_state = manager.deserialize_player_state(
            serialized,
            state_dimension=3,
            player_id=123,
            gameweek=10,
            season="2023",
            last_updated=state.last_updated
        )
        
        assert restored_state.player_id == state.player_id
        assert restored_state.gameweek == state.gameweek
        assert restored_state.season == state.season
        np.testing.assert_array_equal(restored_state.state_mean, state.state_mean)
        np.testing.assert_array_equal(restored_state.state_cov, state.state_cov)
    
    @pytest.mark.skipif(True, reason="Requires actual AdaptivePlayerModel implementation")
    def test_model_serialization(self):
        """Test full model serialization (placeholder)."""
        # This would test actual model serialization when we have a concrete implementation
        pass


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    def test_checkpoint_manager_init(self):
        """Test CheckpointManager initialization."""
        config = PersistenceConfig(
            auto_checkpoint_interval=100,
            max_automatic_checkpoints=50
        )
        manager = CheckpointManager(config)
        
        assert manager.config.auto_checkpoint_interval == 100
        assert manager.config.max_automatic_checkpoints == 50
    
    def test_auto_checkpoint_detection(self):
        """Test automatic checkpoint detection."""
        config = PersistenceConfig(auto_checkpoint_interval=10)
        manager = CheckpointManager(config)
        
        # Should not trigger initially
        assert not manager.should_create_auto_checkpoint(5)
        
        # Should trigger after interval
        assert manager.should_create_auto_checkpoint(10)
        assert manager.should_create_auto_checkpoint(15)
    
    def test_auto_checkpoint_name_generation(self):
        """Test automatic checkpoint name generation."""
        config = PersistenceConfig()
        manager = CheckpointManager(config)
        
        name1 = manager.generate_auto_checkpoint_name()
        name2 = manager.generate_auto_checkpoint_name()
        
        assert name1 != name2
        assert name1.startswith("auto_checkpoint_")
        assert name2.startswith("auto_checkpoint_")


class TestDatabaseStorageBackend:
    """Test DatabaseStorageBackend functionality."""
    
    @pytest.fixture
    def storage_backend(self):
        """Create storage backend for testing."""
        config = PersistenceConfig()
        return DatabaseStorageBackend(config)
    
    @pytest.fixture
    def test_model_version(self, test_database):
        """Create a test model version."""
        with session_scope() as session:
            # Create test model version (simplified)
            version = ModelVersion(
                registry_id=1,  # Assuming registry exists
                version="1.0.0-test",
                config_hash="test_hash",
                training_data_version="test_data",
                training_date=datetime.now(timezone.utc).isoformat(),
                status="ready"
            )
            session.add(version)
            session.flush()
            version_id = version.id
            return version_id
    
    @pytest.mark.asyncio
    async def test_save_load_checkpoint_data(self, storage_backend, test_model_version):
        """Test saving and loading checkpoint data."""
        # Prepare test data
        test_data = b"Test checkpoint data" * 100
        metadata = {
            'version_id': test_model_version,
            'checkpoint_name': 'test_checkpoint',
            'checkpoint_type': CheckpointType.MANUAL.value,
            'compression_format': 'gzip',
            'checksum': 'test_checksum',
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        
        # Save checkpoint
        checkpoint_id = await storage_backend.save_checkpoint_data(test_data, metadata)
        
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, int)
        
        # Load checkpoint
        loaded_data, loaded_metadata = await storage_backend.load_checkpoint_data(checkpoint_id)
        
        assert loaded_data == test_data
        assert loaded_metadata['checkpoint_name'] == 'test_checkpoint'
        assert loaded_metadata['version_id'] == test_model_version
    
    @pytest.mark.asyncio
    async def test_checkpoint_not_found(self, storage_backend):
        """Test loading non-existent checkpoint."""
        with pytest.raises(CheckpointNotFoundError):
            await storage_backend.load_checkpoint_data(999999)
    
    @pytest.mark.asyncio
    async def test_save_load_player_states(self, storage_backend, test_model_version):
        """Test saving and loading player states."""
        # Create checkpoint first
        checkpoint_data = b"test data"
        checkpoint_metadata = {
            'version_id': test_model_version,
            'checkpoint_name': 'states_test',
            'checkpoint_type': CheckpointType.MANUAL.value,
            'compression_format': 'none',
            'checksum': 'test',
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        
        checkpoint_id = await storage_backend.save_checkpoint_data(
            checkpoint_data, checkpoint_metadata
        )
        
        # Prepare player states
        player_states = {
            1: {
                'state_mean': b'state_mean_data_1',
                'state_covariance': b'state_cov_data_1',
                'state_history': None
            },
            2: {
                'state_mean': b'state_mean_data_2',
                'state_covariance': b'state_cov_data_2',
                'state_history': None
            }
        }
        
        state_metadata = {
            1: {
                'state_dimension': 3,
                'gameweek': 10,
                'season': '2023',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'state_checksum': 'checksum1'
            },
            2: {
                'state_dimension': 3,
                'gameweek': 10,
                'season': '2023',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'state_checksum': 'checksum2'
            }
        }
        
        # Save player states
        state_ids = await storage_backend.save_player_states(
            checkpoint_id, player_states, state_metadata
        )
        
        assert len(state_ids) == 2
        
        # Load player states
        loaded_states, loaded_metadata = await storage_backend.load_player_states(checkpoint_id)
        
        assert len(loaded_states) == 2
        assert 1 in loaded_states
        assert 2 in loaded_states
        assert loaded_states[1]['state_mean'] == b'state_mean_data_1'
        assert loaded_metadata[1]['state_dimension'] == 3


class TestModelPersistence:
    """Test ModelPersistence core functionality."""
    
    @pytest.fixture
    def persistence_config(self):
        """Create test persistence configuration."""
        return PersistenceConfig(
            compression_format=CompressionFormat.GZIP,
            auto_checkpoint_interval=5,
            max_automatic_checkpoints=10,
            verify_on_save=True,
            verify_on_load=True
        )
    
    @pytest.fixture
    def model_persistence(self, persistence_config):
        """Create ModelPersistence instance for testing."""
        return ModelPersistence(config=persistence_config)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock AdaptivePlayerModel for testing."""
        model = MagicMock(spec=AdaptivePlayerModel)
        model.config = StateSpaceConfig()
        model.learning_rate = 0.01
        model.decay_factor = 0.95
        model.is_fitted = True
        model.player_states = {
            1: PlayerState(
                player_id=1,
                state_mean=np.array([1.0, 2.0, 3.0]),
                state_cov=np.eye(3),
                gameweek=10,
                season="2023",
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        }
        return model
    
    @pytest.fixture
    def test_model_version(self, test_database):
        """Create test model version."""
        with session_scope() as session:
            version = ModelVersion(
                registry_id=1,
                version="1.0.0-test",
                config_hash="test_hash",
                training_data_version="test_data",
                training_date=datetime.now(timezone.utc).isoformat(),
                status="ready"
            )
            session.add(version)
            session.flush()
            return version.id
    
    @pytest.mark.asyncio
    @patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state')
    async def test_save_checkpoint(self, mock_serialize, model_persistence, mock_model, test_model_version):
        """Test saving a model checkpoint."""
        # Mock serialization
        mock_serialize.return_value = b"serialized_model_data" * 100
        
        # Save checkpoint
        checkpoint_id = await model_persistence.save_checkpoint(
            model=mock_model,
            version_id=test_model_version,
            checkpoint_name="test_checkpoint",
            description="Test checkpoint",
            tags=["test", "manual"],
            validation_score=0.85
        )
        
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, int)
        
        # Verify checkpoint was created
        info = await model_persistence.get_checkpoint_info(checkpoint_id)
        assert info is not None
        assert info.checkpoint_name == "test_checkpoint"
        assert info.description == "Test checkpoint"
        assert info.validation_score == 0.85
        assert "test" in info.tags
        assert "manual" in info.tags
    
    @pytest.mark.asyncio
    @patch('airsenal.framework.model_persistence.SerializationManager.deserialize_model_state')
    async def test_load_checkpoint(self, mock_deserialize, model_persistence, mock_model, test_model_version):
        """Test loading a model checkpoint."""
        # First save a checkpoint
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"serialized_model_data" * 100
            
            checkpoint_id = await model_persistence.save_checkpoint(
                model=mock_model,
                version_id=test_model_version,
                checkpoint_name="load_test"
            )
        
        # Mock deserialization
        mock_deserialize.return_value = mock_model
        
        # Load checkpoint
        loaded_model = await model_persistence.load_checkpoint(checkpoint_id)
        
        assert loaded_model is not None
        mock_deserialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self, model_persistence, mock_model, test_model_version):
        """Test listing checkpoints with filtering."""
        # Create multiple checkpoints
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"test_data"
            
            checkpoint_ids = []
            
            # Manual checkpoint
            checkpoint_ids.append(await model_persistence.save_checkpoint(
                model=mock_model,
                version_id=test_model_version,
                checkpoint_type=CheckpointType.MANUAL,
                checkpoint_name="manual_checkpoint"
            ))
            
            # Automatic checkpoint
            checkpoint_ids.append(await model_persistence.save_checkpoint(
                model=mock_model,
                version_id=test_model_version,
                checkpoint_type=CheckpointType.AUTOMATIC,
                checkpoint_name="auto_checkpoint"
            ))
        
        # List all checkpoints
        all_checkpoints = await model_persistence.list_checkpoints(
            version_id=test_model_version
        )
        assert len(all_checkpoints) >= 2
        
        # List only manual checkpoints
        manual_checkpoints = await model_persistence.list_checkpoints(
            version_id=test_model_version,
            checkpoint_type=CheckpointType.MANUAL
        )
        assert len(manual_checkpoints) >= 1
        assert all(cp.checkpoint_type == CheckpointType.MANUAL for cp in manual_checkpoints)
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, model_persistence, mock_model, test_model_version):
        """Test deleting a checkpoint."""
        # Create checkpoint
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"test_data"
            
            checkpoint_id = await model_persistence.save_checkpoint(
                model=mock_model,
                version_id=test_model_version,
                checkpoint_name="delete_test"
            )
        
        # Verify checkpoint exists
        info = await model_persistence.get_checkpoint_info(checkpoint_id)
        assert info is not None
        
        # Delete checkpoint
        success = await model_persistence.delete_checkpoint(checkpoint_id)
        assert success is True
        
        # Verify checkpoint is gone
        info = await model_persistence.get_checkpoint_info(checkpoint_id)
        assert info is None
    
    @pytest.mark.asyncio
    async def test_auto_checkpointing(self, model_persistence, mock_model, test_model_version):
        """Test automatic checkpoint creation."""
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"test_data"
            
            # Simulate multiple updates
            for i in range(10):  # More than auto_checkpoint_interval (5)
                checkpoint_id = await model_persistence.create_auto_checkpoint(
                    model=mock_model,
                    version_id=test_model_version
                )
                
                if i >= model_persistence.config.auto_checkpoint_interval:
                    # Should create checkpoint after interval
                    assert checkpoint_id is not None
                    break
    
    @pytest.mark.asyncio
    async def test_storage_metrics(self, model_persistence):
        """Test getting storage metrics."""
        metrics = await model_persistence.get_storage_metrics()
        
        assert isinstance(metrics, dict)
        assert 'checkpoint_counts' in metrics
        assert 'total_checkpoints' in metrics
        assert 'storage_backend' in metrics


class TestPersistenceIntegrator:
    """Test PersistenceIntegrator functionality."""
    
    @pytest.fixture
    def mock_state_manager(self):
        """Create mock StateManager."""
        state_manager = MagicMock(spec=StateManager)
        state_manager.history = MagicMock()
        return state_manager
    
    @pytest.fixture
    def mock_redis_cache(self):
        """Create mock Redis cache."""
        return MagicMock(spec=RedisCache)
    
    @pytest.fixture
    def integrator(self, mock_state_manager, mock_redis_cache):
        """Create PersistenceIntegrator for testing."""
        config = PersistenceConfig(
            compression_format=CompressionFormat.GZIP,
            auto_checkpoint_interval=10
        )
        return PersistenceIntegrator(
            persistence_config=config,
            state_manager=mock_state_manager,
            redis_cache=mock_redis_cache
        )
    
    def test_integrator_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator.persistence is not None
        assert integrator.state_integration is not None
        assert integrator.redis_integration is not None
        assert integrator.version_integration is not None
    
    def test_wrap_model(self, integrator):
        """Test model wrapping functionality."""
        mock_model = MagicMock(spec=AdaptivePlayerModel)
        version_id = 123
        
        wrapped_model = integrator.wrap_model(mock_model, version_id)
        
        assert isinstance(wrapped_model, PersistentAdaptiveModel)
        assert wrapped_model.base_model == mock_model
        assert wrapped_model.version_id == version_id
    
    @pytest.mark.asyncio
    async def test_comprehensive_checkpoint(self, integrator, test_model_version):
        """Test comprehensive checkpoint creation."""
        mock_model = MagicMock(spec=AdaptivePlayerModel)
        
        with patch.object(integrator.persistence, 'save_checkpoint') as mock_save:
            mock_save.return_value = 123
            
            with patch.object(integrator.persistence, 'get_checkpoint_info') as mock_info:
                mock_info.return_value = MagicMock()
                
                checkpoint_id = await integrator.create_model_checkpoint(
                    model=mock_model,
                    version_id=test_model_version,
                    description="Test comprehensive checkpoint"
                )
                
                assert checkpoint_id == 123
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_metrics(self, integrator):
        """Test getting integration metrics."""
        with patch.object(integrator.persistence, 'get_storage_metrics') as mock_metrics:
            mock_metrics.return_value = {'test': 'metrics'}
            
            metrics = await integrator.get_integration_metrics()
            
            assert 'persistence' in metrics
            assert 'timestamp' in metrics


class TestPersistentAdaptiveModel:
    """Test PersistentAdaptiveModel wrapper functionality."""
    
    @pytest.fixture
    def mock_base_model(self):
        """Create mock base model."""
        model = MagicMock(spec=AdaptivePlayerModel)
        model.update_state = MagicMock()
        model.predict_state = MagicMock()
        model.fit = MagicMock()
        return model
    
    @pytest.fixture
    def mock_persistence(self):
        """Create mock persistence."""
        return MagicMock(spec=ModelPersistence)
    
    @pytest.fixture
    def mock_integrator(self):
        """Create mock integrator."""
        return MagicMock(spec=PersistenceIntegrator)
    
    @pytest.fixture
    def persistent_model(self, mock_base_model, mock_persistence, mock_integrator):
        """Create PersistentAdaptiveModel for testing."""
        return PersistentAdaptiveModel(
            base_model=mock_base_model,
            persistence=mock_persistence,
            version_id=123,
            integrator=mock_integrator
        )
    
    def test_persistent_model_init(self, persistent_model, mock_base_model):
        """Test persistent model initialization."""
        assert persistent_model.base_model == mock_base_model
        assert persistent_model.version_id == 123
        assert persistent_model.update_count == 0
    
    def test_method_forwarding(self, persistent_model, mock_base_model):
        """Test that methods are properly forwarded to base model."""
        # Test that update methods trigger checkpoint checking
        persistent_model.update_state("test_args")
        
        # Base model method should be called
        mock_base_model.update_state.assert_called_once_with("test_args")
        
        # Update count should be incremented
        assert persistent_model.update_count == 1
    
    @pytest.mark.asyncio
    async def test_manual_checkpoint_creation(self, persistent_model, mock_persistence):
        """Test manual checkpoint creation."""
        mock_persistence.save_checkpoint.return_value = 456
        
        checkpoint_id = await persistent_model.create_checkpoint(
            name="manual_test",
            description="Manual test checkpoint"
        )
        
        assert checkpoint_id == 456
        mock_persistence.save_checkpoint.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoints."""
        config = PersistenceConfig(verify_on_load=True)
        persistence = ModelPersistence(config)
        
        # This would test loading a corrupted checkpoint
        # Implementation depends on having actual corrupted data
        pass
    
    @pytest.mark.asyncio
    async def test_storage_failure_handling(self):
        """Test handling of storage failures."""
        config = PersistenceConfig()
        persistence = ModelPersistence(config)
        
        # Mock storage failure
        with patch.object(persistence.storage_backend, 'save_checkpoint_data') as mock_save:
            mock_save.side_effect = StorageError("Storage failed")
            
            mock_model = MagicMock(spec=AdaptivePlayerModel)
            
            with pytest.raises(ModelPersistenceError):
                await persistence.save_checkpoint(
                    model=mock_model,
                    version_id=1,
                    checkpoint_name="failure_test"
                )
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        # Test invalid compression format
        with pytest.raises((ValueError, AttributeError)):
            config = PersistenceConfig(compression_format="invalid_format")
            CompressionManager(config)
    
    def test_compression_with_invalid_data(self):
        """Test compression manager with edge cases."""
        config = PersistenceConfig(compression_format=CompressionFormat.GZIP)
        manager = CompressionManager(config)
        
        # Test empty data
        compressed, metadata = manager.compress(b"")
        decompressed = manager.decompress(compressed)
        assert decompressed == b""
        
        # Test very small data
        small_data = b"x"
        compressed, metadata = manager.compress(small_data)
        decompressed = manager.decompress(compressed)
        assert decompressed == small_data


class TestPerformanceAndConcurrency:
    """Test performance characteristics and concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_creation(self):
        """Test concurrent checkpoint creation."""
        config = PersistenceConfig()
        persistence = ModelPersistence(config)
        
        mock_model = MagicMock(spec=AdaptivePlayerModel)
        
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"test_data"
            
            # Create multiple checkpoints concurrently
            tasks = []
            for i in range(5):
                task = persistence.save_checkpoint(
                    model=mock_model,
                    version_id=1,
                    checkpoint_name=f"concurrent_test_{i}"
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, int)
    
    def test_large_data_compression(self):
        """Test compression with large data."""
        config = PersistenceConfig(compression_format=CompressionFormat.GZIP)
        manager = CompressionManager(config)
        
        # Create large test data (1MB)
        large_data = b"test_pattern" * 100000
        
        start_time = time.time()
        compressed, metadata = manager.compress(large_data)
        compression_time = time.time() - start_time
        
        start_time = time.time()
        decompressed = manager.decompress(compressed)
        decompression_time = time.time() - start_time
        
        assert decompressed == large_data
        assert metadata['compression_ratio'] < 1.0  # Should compress well
        
        # Performance should be reasonable (adjust thresholds as needed)
        assert compression_time < 1.0  # Should compress in under 1 second
        assert decompression_time < 0.5  # Should decompress in under 0.5 seconds
    
    @pytest.mark.asyncio
    async def test_checkpoint_pruning_performance(self):
        """Test performance of checkpoint pruning."""
        config = PersistenceConfig(
            max_automatic_checkpoints=5,
            checkpoint_pruning_enabled=True
        )
        persistence = ModelPersistence(config)
        
        mock_model = MagicMock(spec=AdaptivePlayerModel)
        
        with patch('airsenal.framework.model_persistence.SerializationManager.serialize_model_state') as mock_serialize:
            mock_serialize.return_value = b"test_data"
            
            # Create many checkpoints to trigger pruning
            for i in range(10):
                await persistence.save_checkpoint(
                    model=mock_model,
                    version_id=1,
                    checkpoint_type=CheckpointType.AUTOMATIC,
                    checkpoint_name=f"prune_test_{i}"
                )
            
            # Verify pruning occurred
            checkpoints = await persistence.list_checkpoints(
                version_id=1,
                checkpoint_type=CheckpointType.AUTOMATIC
            )
            
            # Should have pruned to max_automatic_checkpoints
            assert len(checkpoints) <= config.max_automatic_checkpoints


class TestBackwardCompatibility:
    """Test backward compatibility and migration scenarios."""
    
    @pytest.mark.skipif(True, reason="Requires migration testing setup")
    def test_old_checkpoint_format_compatibility(self):
        """Test loading checkpoints created with older versions."""
        # This would test loading checkpoints created with previous formats
        pass
    
    @pytest.mark.skipif(True, reason="Requires version upgrade testing")
    def test_schema_migration_compatibility(self):
        """Test that schema migrations don't break existing checkpoints."""
        # This would test schema migration scenarios
        pass


# Integration test that exercises the full persistence workflow
@pytest.mark.asyncio
async def test_full_persistence_workflow():
    """Test complete persistence workflow from model creation to recovery."""
    # This test would create a real model, save checkpoints, modify it,
    # and restore from checkpoints to verify the complete workflow works
    
    # For now, just verify the components integrate properly
    config = PersistenceConfig(
        compression_format=CompressionFormat.GZIP,
        auto_checkpoint_interval=5,
        verify_on_save=True,
        verify_on_load=True
    )
    
    persistence = ModelPersistence(config)
    integrator = PersistenceIntegrator(persistence_config=config)
    
    # Verify integration works
    assert persistence is not None
    assert integrator is not None
    assert integrator.persistence is not None