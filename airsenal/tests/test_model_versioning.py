"""
Test suite for model versioning system

Tests cover:
- Model registration and metadata tracking
- Artifact storage and retrieval
- Model comparison and performance tracking
- A/B testing framework
- Integration with prediction pipeline
"""

import json
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.ab_testing import ABTestManager, ExperimentConfig, ExperimentStatus
from airsenal.framework.model_versioning import (
    ModelVersionManager,
    ModelVersioningConfig,
    ModelArtifactManager,
    ModelNotFoundError,
    ModelStorageError,
    register_model,
    load_model,
)
from airsenal.framework.player_model import ConjugatePlayerModel, NumpyroPlayerModel
from airsenal.framework.schema import (
    Base,
    ModelRegistry,
    ModelVersion,
    ModelArtifact,
    ModelPerformance,
    ModelExperiment,
)
from airsenal.framework.versioned_prediction_utils import VersionedPredictionManager


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for model storage"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_storage_dir):
    """Create test configuration"""
    config = ModelVersioningConfig()
    config.model_storage_dir = temp_storage_dir
    return config


@pytest.fixture
def test_db_session():
    """Create in-memory test database session"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def mock_player_model():
    """Create mock player model for testing"""
    model = ConjugatePlayerModel()
    model.player_ids = np.array([1, 2, 3])
    model.prior = np.array([1.0, 0.5, 2.0])
    model.posterior = np.array([2.0, 1.0, 3.0])
    model.mean_probabilities = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
    return model


@pytest.fixture
def mock_numpyro_model():
    """Create mock NumPyro model for testing"""
    model = NumpyroPlayerModel()
    model.player_ids = np.array([1, 2, 3])
    # Mock JAX arrays
    model.samples = {
        "probs": np.random.random((1000, 3, 3)),  # MCMC samples
    }
    return model


class TestModelVersioningConfig:
    """Test model versioning configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ModelVersioningConfig()
        
        assert config.enable_compression is True
        assert config.default_compression == "gzip"
        assert config.max_versions_per_model == 50
        assert config.enable_s3_storage is False
        assert config.checksum_algorithm == "sha256"
    
    def test_ensure_storage_dir(self, temp_storage_dir):
        """Test storage directory creation"""
        config = ModelVersioningConfig()
        config.model_storage_dir = temp_storage_dir / "models"
        
        # Directory doesn't exist initially
        assert not config.model_storage_dir.exists()
        
        # Should create directory
        result_dir = config.ensure_storage_dir()
        assert config.model_storage_dir.exists()
        assert result_dir == config.model_storage_dir


class TestModelArtifactManager:
    """Test model artifact serialization and storage"""
    
    def test_serialize_conjugate_model(self, mock_player_model):
        """Test serialization of conjugate player model"""
        config = ModelVersioningConfig()
        manager = ModelArtifactManager(config)
        
        serialized = manager.serialize_conjugate_model(mock_player_model)
        
        assert serialized["model_type"] == "ConjugatePlayerModel"
        assert np.array_equal(serialized["player_ids"], mock_player_model.player_ids)
        assert np.array_equal(serialized["prior"], mock_player_model.prior)
        assert np.array_equal(serialized["posterior"], mock_player_model.posterior)
    
    def test_deserialize_conjugate_model(self, mock_player_model):
        """Test deserialization of conjugate player model"""
        config = ModelVersioningConfig()
        manager = ModelArtifactManager(config)
        
        # Serialize then deserialize
        serialized = manager.serialize_conjugate_model(mock_player_model)
        deserialized = manager.deserialize_conjugate_model(serialized)
        
        assert np.array_equal(deserialized.player_ids, mock_player_model.player_ids)
        assert np.array_equal(deserialized.prior, mock_player_model.prior)
        assert np.array_equal(deserialized.posterior, mock_player_model.posterior)
        assert np.array_equal(deserialized.mean_probabilities, mock_player_model.mean_probabilities)
    
    def test_serialize_numpyro_model(self, mock_numpyro_model):
        """Test serialization of NumPyro model"""
        config = ModelVersioningConfig()
        manager = ModelArtifactManager(config)
        
        serialized = manager.serialize_jax_model(mock_numpyro_model)
        
        assert serialized["model_type"] == "NumpyroPlayerModel"
        assert np.array_equal(serialized["player_ids"], mock_numpyro_model.player_ids)
        assert "samples" in serialized
        assert "jax_version" in serialized
    
    def test_save_and_load_artifact(self, test_config, test_db_session, mock_player_model):
        """Test saving and loading model artifacts"""
        manager = ModelArtifactManager(test_config)
        
        # Save artifact
        artifact = manager.save_artifact(
            mock_player_model, version_id=1, artifact_type="full_model", dbsession=test_db_session
        )
        
        assert artifact.version_id == 1
        assert artifact.artifact_type == "full_model"
        assert artifact.file_path is not None
        assert Path(artifact.file_path).exists()
        assert artifact.checksum is not None
        
        # Load artifact
        loaded_model = manager.load_artifact(artifact)
        
        assert isinstance(loaded_model, ConjugatePlayerModel)
        assert np.array_equal(loaded_model.player_ids, mock_player_model.player_ids)
    
    def test_checksum_verification(self, test_config, test_db_session, mock_player_model):
        """Test checksum verification during loading"""
        manager = ModelArtifactManager(test_config)
        
        # Save artifact
        artifact = manager.save_artifact(
            mock_player_model, version_id=1, dbsession=test_db_session
        )
        
        # Corrupt the checksum
        artifact.checksum = "invalid_checksum"
        
        # Should raise error due to checksum mismatch
        with pytest.raises(ModelStorageError, match="Checksum mismatch"):
            manager.load_artifact(artifact)


class TestModelVersionManager:
    """Test model version management"""
    
    def test_register_model_type(self, test_config, test_db_session):
        """Test registering a new model type"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        registry = manager.register_model_type(
            model_name="test_model",
            model_type="TestModel",
            description="Test model for unit tests",
        )
        
        assert registry.model_name == "test_model"
        assert registry.model_type == "TestModel"
        assert registry.description == "Test model for unit tests"
        assert registry.is_active is True
        
        # Test duplicate registration
        registry2 = manager.register_model_type(
            model_name="test_model",
            model_type="TestModel",
        )
        
        assert registry.id == registry2.id  # Should return existing registry
    
    def test_register_model_version(self, test_config, test_db_session, mock_player_model):
        """Test registering a model version"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        training_params = {"param1": "value1", "param2": 42}
        performance_metrics = {"mae": 0.15, "rmse": 0.25}
        
        model_version = manager.register_model_version(
            model=mock_player_model,
            model_name="test_player_model",
            version="1.0.0",
            training_params=training_params,
            performance_metrics=performance_metrics,
            notes="Test model version",
            tags=["test", "player"],
        )
        
        assert model_version.version == "1.0.0"
        assert model_version.validation_mae == 0.15
        assert model_version.validation_rmse == 0.25
        assert model_version.status == "ready"
        assert model_version.notes == "Test model version"
        assert model_version.tags == "test,player"
        
        # Check that artifact was created
        artifacts = test_db_session.query(ModelArtifact).filter_by(
            version_id=model_version.id
        ).all()
        assert len(artifacts) == 1
        assert artifacts[0].artifact_type == "full_model"
    
    def test_load_model_version(self, test_config, test_db_session, mock_player_model):
        """Test loading a specific model version"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register model
        model_version = manager.register_model_version(
            model=mock_player_model,
            model_name="test_model",
            version="1.0.0",
        )
        
        # Load model
        loaded_model = manager.load_model_version("test_model", "1.0.0")
        
        assert isinstance(loaded_model, ConjugatePlayerModel)
        assert np.array_equal(loaded_model.player_ids, mock_player_model.player_ids)
    
    def test_load_latest_model(self, test_config, test_db_session, mock_player_model):
        """Test loading the latest model version"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register multiple versions
        manager.register_model_version(mock_player_model, "test_model", "1.0.0")
        manager.register_model_version(mock_player_model, "test_model", "1.1.0")
        manager.register_model_version(mock_player_model, "test_model", "2.0.0")
        
        # Load latest (should be 2.0.0)
        loaded_model = manager.load_model_version("test_model")
        
        # Verify it's the latest version by checking the database
        latest_version = test_db_session.query(ModelVersion).join(ModelRegistry).filter(
            ModelRegistry.model_name == "test_model"
        ).order_by(ModelVersion.training_date.desc()).first()
        
        assert latest_version.version == "2.0.0"
    
    def test_model_not_found(self, test_config, test_db_session):
        """Test error handling for non-existent models"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        with pytest.raises(ModelNotFoundError, match="not found in registry"):
            manager.load_model_version("non_existent_model")
    
    def test_list_model_versions(self, test_config, test_db_session, mock_player_model):
        """Test listing model versions"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register multiple models and versions
        manager.register_model_version(mock_player_model, "model_a", "1.0.0")
        manager.register_model_version(mock_player_model, "model_a", "1.1.0")
        manager.register_model_version(mock_player_model, "model_b", "1.0.0")
        
        # List all versions
        df_all = manager.list_model_versions()
        assert len(df_all) == 3
        
        # List versions for specific model
        df_model_a = manager.list_model_versions("model_a")
        assert len(df_model_a) == 2
        assert all(df_model_a["model_name"] == "model_a")
    
    def test_compare_models(self, test_config, test_db_session, mock_player_model):
        """Test model comparison"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register versions with different performance
        manager.register_model_version(
            mock_player_model, "test_model", "1.0.0",
            performance_metrics={"mae": 0.20, "rmse": 0.30}
        )
        manager.register_model_version(
            mock_player_model, "test_model", "1.1.0",
            performance_metrics={"mae": 0.15, "rmse": 0.25}
        )
        
        # Compare versions
        comparison = manager.compare_models("test_model", ["1.0.0", "1.1.0"])
        
        assert len(comparison) == 2
        assert comparison.loc[0, "version"] == "1.0.0"
        assert comparison.loc[0, "validation_mae"] == 0.20
        assert comparison.loc[1, "version"] == "1.1.0"
        assert comparison.loc[1, "validation_mae"] == 0.15
    
    def test_set_production_model(self, test_config, test_db_session, mock_player_model):
        """Test setting production model"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register versions
        v1 = manager.register_model_version(mock_player_model, "test_model", "1.0.0")
        v2 = manager.register_model_version(mock_player_model, "test_model", "1.1.0")
        
        # Set v1.1.0 as production
        prod_version = manager.set_production_model("test_model", "1.1.0")
        
        assert prod_version.version == "1.1.0"
        assert prod_version.is_production is True
        assert prod_version.deployment_date is not None
        
        # Check that v1.0.0 is no longer production
        v1_updated = test_db_session.query(ModelVersion).filter_by(id=v1.id).first()
        assert v1_updated.is_production is False
    
    def test_record_performance(self, test_config, test_db_session, mock_player_model):
        """Test recording performance metrics"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register model
        model_version = manager.register_model_version(mock_player_model, "test_model", "1.0.0")
        
        # Record performance
        metrics = {
            "mae": 0.15,
            "rmse": 0.25,
            "prediction_correlation": 0.85,
            "custom_metric": 0.95,
        }
        
        performance = manager.record_performance(
            "test_model", "1.0.0", metrics, "validation"
        )
        
        assert performance.mae == 0.15
        assert performance.rmse == 0.25
        assert performance.prediction_correlation == 0.85
        assert performance.dataset_type == "validation"
        
        # Check additional metrics are stored as JSON
        additional = json.loads(performance.additional_metrics)
        assert additional["custom_metric"] == 0.95
    
    def test_cleanup_old_versions(self, test_config, test_db_session, mock_player_model):
        """Test cleanup of old model versions"""
        manager = ModelVersionManager(test_config, test_db_session)
        
        # Register 6 versions
        versions = []
        for i in range(6):
            version = manager.register_model_version(
                mock_player_model, "test_model", f"1.{i}.0"
            )
            versions.append(version)
        
        # Set one as production
        manager.set_production_model("test_model", "1.2.0")
        
        # Cleanup keeping latest 3 + production
        deleted_count = manager.cleanup_old_versions("test_model", keep_latest=3, keep_production=True)
        
        # Should delete 2 versions (keep latest 3 + production = 4 total)
        assert deleted_count == 2
        
        # Check remaining versions
        remaining = test_db_session.query(ModelVersion).join(ModelRegistry).filter(
            ModelRegistry.model_name == "test_model"
        ).all()
        assert len(remaining) == 4


class TestABTestManager:
    """Test A/B testing framework"""
    
    def test_create_experiment(self, test_db_session, mock_player_model):
        """Test creating an A/B test experiment"""
        # Setup model versions
        manager = ModelVersionManager(dbsession=test_db_session)
        manager.register_model_version(mock_player_model, "test_model", "1.0.0")
        manager.register_model_version(mock_player_model, "test_model", "1.1.0")
        
        # Create experiment
        ab_manager = ABTestManager(test_db_session)
        config = ExperimentConfig(
            experiment_name="test_experiment",
            model_name="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0",
            traffic_split=0.6,
        )
        
        experiment = ab_manager.create_experiment(config)
        
        assert experiment.experiment_name == "test_experiment"
        assert experiment.traffic_split == 0.6
        assert experiment.status == ExperimentStatus.PLANNED.value
    
    def test_start_stop_experiment(self, test_db_session, mock_player_model):
        """Test starting and stopping experiments"""
        # Setup
        manager = ModelVersionManager(dbsession=test_db_session)
        manager.register_model_version(mock_player_model, "test_model", "1.0.0")
        manager.register_model_version(mock_player_model, "test_model", "1.1.0")
        
        ab_manager = ABTestManager(test_db_session)
        config = ExperimentConfig(
            experiment_name="test_experiment",
            model_name="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0",
        )
        
        experiment = ab_manager.create_experiment(config)
        
        # Start experiment
        started_exp = ab_manager.start_experiment(experiment.id)
        assert started_exp.status == ExperimentStatus.RUNNING.value
        
        # Stop experiment
        stopped_exp = ab_manager.stop_experiment(experiment.id, "Test complete")
        assert stopped_exp.status == ExperimentStatus.STOPPED.value
        assert "Test complete" in stopped_exp.notes
    
    @patch('airsenal.framework.versioned_prediction_utils.VersionedPredictionManager.run_versioned_predictions')
    def test_run_experiment_iteration(self, mock_run_predictions, test_db_session, mock_player_model):
        """Test running an experiment iteration"""
        # Setup
        manager = ModelVersionManager(dbsession=test_db_session)
        manager.register_model_version(mock_player_model, "player_model_fwd", "1.0.0")
        manager.register_model_version(mock_player_model, "player_model_fwd", "1.1.0")
        
        ab_manager = ABTestManager(test_db_session)
        config = ExperimentConfig(
            experiment_name="test_experiment",
            model_name="player_model_fwd",
            control_version="1.0.0",
            treatment_version="1.1.0",
        )
        
        experiment = ab_manager.create_experiment(config)
        ab_manager.start_experiment(experiment.id)
        
        # Mock prediction returns
        mock_run_predictions.side_effect = ["control_tag_123", "treatment_tag_456"]
        
        # Run iteration
        results = ab_manager.run_experiment_iteration(experiment.id, [1, 2, 3])
        
        assert results["control_tag"] == "control_tag_123"
        assert results["treatment_tag"] == "treatment_tag_456"
        assert results["experiment_id"] == experiment.id
        
        # Check that predictions were called with correct parameters
        assert mock_run_predictions.call_count == 2
    
    def test_analyze_experiment_results(self, test_db_session, mock_player_model):
        """Test analyzing experiment results"""
        # This test would require mock prediction data
        # For now, test the basic structure
        ab_manager = ABTestManager(test_db_session)
        
        # Mock some prediction data
        with patch.object(ab_manager, '_get_predictions_by_tags') as mock_get_predictions:
            # Create mock dataframes
            control_data = pd.DataFrame({
                'predicted_points': np.random.normal(5.0, 1.0, 100),
                'player_id': range(100),
                'fixture_id': range(100),
            })
            treatment_data = pd.DataFrame({
                'predicted_points': np.random.normal(5.2, 1.0, 100),
                'player_id': range(100),
                'fixture_id': range(100),
            })
            
            mock_get_predictions.side_effect = [control_data, treatment_data]
            
            # Analyze results
            analysis = ab_manager.analyze_experiment_results(
                1, ["control_tag"], ["treatment_tag"]
            )
            
            assert "control_sample_size" in analysis
            assert "treatment_sample_size" in analysis
            assert "t_test_pvalue" in analysis
            assert "cohens_d" in analysis
            assert "recommendation" in analysis


class TestVersionedPredictionManager:
    """Test versioned prediction utilities"""
    
    @patch('airsenal.framework.prediction_utils.process_player_data')
    def test_fit_and_register_player_model(self, mock_process_data, test_db_session):
        """Test fitting and registering a player model"""
        # Mock training data
        mock_data = {
            "player_ids": np.array([1, 2, 3]),
            "nplayer": 3,
            "nmatch": 10,
            "minutes": np.random.randint(0, 90, (3, 10)),
            "y": np.random.randint(0, 3, (3, 10, 3)),
            "alpha": np.array([1.0, 0.5, 2.0]),
        }
        mock_process_data.return_value = mock_data
        
        manager = VersionedPredictionManager(dbsession=test_db_session)
        
        # Fit and register model
        model = manager.fit_and_register_player_model(
            position="FWD",
            season="2324",
            gameweek=10,
            version="1.0.0",
            notes="Test forward model",
        )
        
        assert isinstance(model, ConjugatePlayerModel)
        
        # Check that model was registered
        registry = test_db_session.query(ModelRegistry).filter_by(
            model_name="player_model_fwd"
        ).first()
        assert registry is not None
        
        version_record = test_db_session.query(ModelVersion).filter_by(
            registry_id=registry.id, version="1.0.0"
        ).first()
        assert version_record is not None
        assert version_record.notes == "Test forward model"


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_register_model_function(self, mock_player_model):
        """Test convenience register_model function"""
        with patch('airsenal.framework.model_versioning.ModelVersionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            register_model(mock_player_model, "test_model", "1.0.0", notes="Test")
            
            mock_manager.register_model_version.assert_called_once_with(
                mock_player_model, "test_model", "1.0.0", notes="Test"
            )
    
    def test_load_model_function(self):
        """Test convenience load_model function"""
        with patch('airsenal.framework.model_versioning.ModelVersionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.load_model_version.return_value = "mock_model"
            
            result = load_model("test_model", "1.0.0")
            
            assert result == "mock_model"
            mock_manager.load_model_version.assert_called_once_with("test_model", "1.0.0")


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self, test_config, test_db_session, mock_player_model):
        """Test complete workflow from model registration to A/B testing"""
        # 1. Register models
        manager = ModelVersionManager(test_config, test_db_session)
        
        v1 = manager.register_model_version(
            mock_player_model, "test_model", "1.0.0",
            performance_metrics={"mae": 0.20}
        )
        v2 = manager.register_model_version(
            mock_player_model, "test_model", "1.1.0",
            performance_metrics={"mae": 0.18}
        )
        
        # 2. Set production model
        manager.set_production_model("test_model", "1.0.0")
        
        # 3. Create A/B test
        ab_manager = ABTestManager(test_db_session)
        config = ExperimentConfig(
            experiment_name="v1_vs_v11",
            model_name="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0",
        )
        experiment = ab_manager.create_experiment(config)
        
        # 4. Load models
        loaded_v1 = manager.load_model_version("test_model", "1.0.0")
        loaded_v2 = manager.load_model_version("test_model", "1.1.0")
        prod_model = manager.get_production_model("test_model")
        
        # 5. Compare models
        comparison = manager.compare_models("test_model", ["1.0.0", "1.1.0"], ["validation_mae"])
        
        # Verify the workflow
        assert len(comparison) == 2
        assert comparison.loc[1, "validation_mae"] < comparison.loc[0, "validation_mae"]  # v1.1.0 is better
        assert experiment.status == ExperimentStatus.PLANNED.value
        assert isinstance(loaded_v1, ConjugatePlayerModel)
        assert isinstance(prod_model, ConjugatePlayerModel)


if __name__ == "__main__":
    pytest.main([__file__])