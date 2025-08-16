"""
Model Versioning System for AIrsenal

This module provides comprehensive model versioning capabilities including:
- Model registration and metadata tracking
- Artifact storage and retrieval (JAX/NumPyro model serialization)
- Performance monitoring and comparison
- A/B testing framework
- Model lifecycle management

The system is designed to work with AIrsenal's existing prediction models:
- Player models (NumpyroPlayerModel, ConjugatePlayerModel)
- Team models (ExtendedDixonColesMatchPredictor, etc.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from airsenal.framework.env import (
    AIRSENAL_HOME,
    AIRSENAL_MODEL_COMPRESSION,
    AIRSENAL_MODEL_ENABLE_S3,
    AIRSENAL_MODEL_MAX_VERSIONS,
    AIRSENAL_MODEL_S3_BUCKET,
    AIRSENAL_MODEL_STORAGE_DIR,
)
from airsenal.framework.player_model import (
    ConjugatePlayerModel,
    NumpyroPlayerModel,
)
from airsenal.framework.schema import (
    ModelArtifact,
    ModelPerformance,
    ModelRegistry,
    ModelVersion,
    session,
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelVersioningError(Exception):
    """Base exception for model versioning errors"""


class ModelNotFoundError(ModelVersioningError):
    """Raised when a requested model version is not found"""


class ModelStorageError(ModelVersioningError):
    """Raised when there are issues with model storage/retrieval"""


class ModelVersioningConfig:
    """Configuration class for model versioning system"""

    def __init__(self):
        # Use environment variables or defaults
        if AIRSENAL_MODEL_STORAGE_DIR:
            self.model_storage_dir = Path(AIRSENAL_MODEL_STORAGE_DIR)
        else:
            self.model_storage_dir = AIRSENAL_HOME / "models"

        self.enable_compression = True
        self.default_compression = AIRSENAL_MODEL_COMPRESSION or "gzip"
        self.max_versions_per_model = AIRSENAL_MODEL_MAX_VERSIONS or 50
        self.enable_s3_storage = AIRSENAL_MODEL_ENABLE_S3 or False
        self.s3_bucket = AIRSENAL_MODEL_S3_BUCKET
        self.checksum_algorithm = "sha256"

    def ensure_storage_dir(self) -> Path:
        """Ensure model storage directory exists"""
        self.model_storage_dir.mkdir(parents=True, exist_ok=True)
        return self.model_storage_dir


class ModelArtifactManager:
    """Handles serialization and storage of model artifacts"""

    def __init__(self, config: ModelVersioningConfig):
        self.config = config

    def serialize_jax_model(self, model: NumpyroPlayerModel) -> dict[str, Any]:
        """Serialize JAX/NumPyro model to dictionary"""
        if model.samples is None or model.player_ids is None:
            raise ModelStorageError("Model must be fitted before serialization")

        # Convert JAX arrays to numpy for serialization
        serialized_samples = {}
        for key, value in model.samples.items():
            if isinstance(value, jnp.ndarray):
                serialized_samples[key] = np.array(value)
            else:
                serialized_samples[key] = value

        return {
            "model_type": "NumpyroPlayerModel",
            "player_ids": np.array(model.player_ids),
            "samples": serialized_samples,
            "jax_version": jax.__version__,
        }

    def deserialize_jax_model(self, data: dict[str, Any]) -> NumpyroPlayerModel:
        """Deserialize JAX/NumPyro model from dictionary"""
        model = NumpyroPlayerModel()
        model.player_ids = data["player_ids"]

        # Convert numpy arrays back to JAX arrays
        model.samples = {}
        for key, value in data["samples"].items():
            if isinstance(value, np.ndarray):
                model.samples[key] = jnp.array(value)
            else:
                model.samples[key] = value

        return model

    def serialize_conjugate_model(self, model: ConjugatePlayerModel) -> dict[str, Any]:
        """Serialize conjugate player model to dictionary"""
        if model.player_ids is None:
            raise ModelStorageError("Model must be fitted before serialization")

        return {
            "model_type": "ConjugatePlayerModel",
            "player_ids": np.array(model.player_ids),
            "prior": np.array(model.prior) if model.prior is not None else None,
            "posterior": np.array(model.posterior) if model.posterior is not None else None,
            "mean_probabilities": np.array(model.mean_probabilities) if model.mean_probabilities is not None else None,
        }

    def deserialize_conjugate_model(self, data: dict[str, Any]) -> ConjugatePlayerModel:
        """Deserialize conjugate player model from dictionary"""
        model = ConjugatePlayerModel()
        model.player_ids = data["player_ids"]
        model.prior = data["prior"]
        model.posterior = data["posterior"]
        model.mean_probabilities = data["mean_probabilities"]
        return model

    def serialize_model(self, model: Any) -> dict[str, Any]:
        """Generic model serialization dispatcher"""
        if isinstance(model, NumpyroPlayerModel):
            return self.serialize_jax_model(model)
        if isinstance(model, ConjugatePlayerModel):
            return self.serialize_conjugate_model(model)
        # For other models (like BPL team models), use pickle-based serialization
        return {
            "model_type": type(model).__name__,
            "model_data": model,
            "serialization_method": "pickle"
        }

    def deserialize_model(self, data: dict[str, Any]) -> Any:
        """Generic model deserialization dispatcher"""
        model_type = data["model_type"]

        if model_type == "NumpyroPlayerModel":
            return self.deserialize_jax_model(data)
        if model_type == "ConjugatePlayerModel":
            return self.deserialize_conjugate_model(data)
        # For pickle-serialized models
        return data["model_data"]

    def save_artifact(
        self,
        model: Any,
        version_id: int,
        artifact_type: str = "full_model",
        dbsession: Session = session,
    ) -> ModelArtifact:
        """Save model artifact to storage and create database record"""

        # Serialize model
        model_data = self.serialize_model(model)

        # Create file path
        storage_dir = self.config.ensure_storage_dir()
        filename = f"model_v{version_id}_{artifact_type}.pkl"
        file_path = storage_dir / filename

        # Save to file
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)

            # Calculate checksum
            checksum = self._calculate_checksum(file_path)
            file_size = file_path.stat().st_size

            # Create database record
            artifact = ModelArtifact(
                version_id=version_id,
                artifact_type=artifact_type,
                file_path=str(file_path),
                file_size_bytes=file_size,
                checksum=checksum,
                serialization_format="pickle",
                created_at=datetime.now().isoformat(),
            )

            dbsession.add(artifact)
            dbsession.commit()

            logger.info(f"Saved model artifact {artifact_type} for version {version_id}")
            return artifact

        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")
            if file_path.exists():
                file_path.unlink()  # Clean up partial file
            raise ModelStorageError(f"Failed to save model artifact: {e}")

    def load_artifact(self, artifact: ModelArtifact) -> Any:
        """Load model artifact from storage"""
        file_path = Path(artifact.file_path)

        if not file_path.exists():
            raise ModelStorageError(f"Artifact file not found: {file_path}")

        # Verify checksum if available
        if artifact.checksum:
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != artifact.checksum:
                raise ModelStorageError(f"Checksum mismatch for artifact {artifact.id}")

        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            return self.deserialize_model(model_data)

        except Exception as e:
            logger.error(f"Failed to load artifact {artifact.id}: {e}")
            raise ModelStorageError(f"Failed to load model artifact: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file integrity verification"""
        hash_algo = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        return hash_algo.hexdigest()


class ModelVersionManager:
    """Main interface for model versioning operations"""

    def __init__(self, config: ModelVersioningConfig | None = None, dbsession: Session = session):
        self.config = config or ModelVersioningConfig()
        self.artifact_manager = ModelArtifactManager(self.config)
        self.dbsession = dbsession

    def register_model_type(
        self,
        model_name: str,
        model_type: str,
        description: str | None = None,
        created_by: str = "airsenal",
    ) -> ModelRegistry:
        """Register a new model type in the registry"""

        # Check if already exists
        existing = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, model_type=model_type
        ).first()

        if existing:
            logger.info(f"Model type {model_name}:{model_type} already registered")
            return existing

        registry = ModelRegistry(
            model_name=model_name,
            model_type=model_type,
            description=description,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
        )

        self.dbsession.add(registry)
        self.dbsession.commit()

        logger.info(f"Registered new model type: {model_name}:{model_type}")
        return registry

    def register_model_version(
        self,
        model: Any,
        model_name: str,
        version: str,
        training_params: dict[str, Any] | None = None,
        feature_set: str | None = None,
        training_duration_seconds: int | None = None,
        performance_metrics: dict[str, float] | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> ModelVersion:
        """Register a new model version with artifacts and metadata"""

        # Get or create model registry
        model_type = type(model).__name__
        registry = self.register_model_type(model_name, model_type)

        # Create configuration hash
        config_data = {
            "training_params": training_params or {},
            "feature_set": feature_set,
            "model_type": model_type,
        }
        config_hash = hashlib.sha256(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Create training data version hash (simplified)
        training_data_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model version record
        model_version = ModelVersion(
            registry_id=registry.id,
            version=version,
            config_hash=config_hash,
            training_data_version=training_data_version,
            training_params=json.dumps(training_params) if training_params else None,
            feature_set=feature_set,
            training_date=datetime.now().isoformat(),
            training_duration_seconds=training_duration_seconds,
            status="ready",
            notes=notes,
            tags=",".join(tags) if tags else None,
        )

        # Add performance metrics if provided
        if performance_metrics:
            model_version.validation_mae = performance_metrics.get("mae")
            model_version.validation_rmse = performance_metrics.get("rmse")
            model_version.validation_accuracy = performance_metrics.get("accuracy")
            model_version.cross_validation_score = performance_metrics.get("cv_score")

        self.dbsession.add(model_version)
        self.dbsession.commit()

        # Save model artifact
        try:
            self.artifact_manager.save_artifact(
                model, model_version.id, "full_model", self.dbsession
            )
            logger.info(f"Registered model version {model_name} v{version}")
            return model_version

        except Exception as e:
            # Clean up the model version record if artifact saving fails
            self.dbsession.delete(model_version)
            self.dbsession.commit()
            raise ModelStorageError(f"Failed to register model version: {e}")

    def load_model_version(
        self,
        model_name: str,
        version: str | None = None,
    ) -> Any:
        """Load a specific model version or the latest if version not specified"""

        try:
            registry = self.dbsession.query(ModelRegistry).filter_by(
                model_name=model_name, is_active=True
            ).one()
        except NoResultFound:
            raise ModelNotFoundError(f"Model '{model_name}' not found in registry")

        # Get model version
        version_query = self.dbsession.query(ModelVersion).filter_by(registry_id=registry.id)

        if version:
            try:
                model_version = version_query.filter_by(version=version).one()
            except NoResultFound:
                raise ModelNotFoundError(f"Version '{version}' not found for model '{model_name}'")
        else:
            # Get latest version
            model_version = version_query.filter_by(status="ready").order_by(
                ModelVersion.training_date.desc()
            ).first()

            if not model_version:
                raise ModelNotFoundError(f"No ready versions found for model '{model_name}'")

        # Load artifact
        artifact = self.dbsession.query(ModelArtifact).filter_by(
            version_id=model_version.id, artifact_type="full_model"
        ).first()

        if not artifact:
            raise ModelStorageError(f"No artifacts found for model version {model_version.id}")

        model = self.artifact_manager.load_artifact(artifact)
        logger.info(f"Loaded model {model_name} v{model_version.version}")
        return model

    def list_model_versions(
        self,
        model_name: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List available model versions"""

        query = self.dbsession.query(
            ModelRegistry.model_name,
            ModelRegistry.model_type,
            ModelVersion.version,
            ModelVersion.status,
            ModelVersion.training_date,
            ModelVersion.validation_mae,
            ModelVersion.validation_rmse,
            ModelVersion.is_production,
            ModelVersion.notes,
        ).join(ModelVersion)

        if model_name:
            query = query.filter(ModelRegistry.model_name == model_name)

        if limit:
            query = query.limit(limit)

        query = query.order_by(ModelVersion.training_date.desc())

        df = pd.read_sql(query.statement, self.dbsession.bind)
        return df

    def compare_models(
        self,
        model_name: str,
        versions: list[str],
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare performance metrics across model versions"""

        if metrics is None:
            metrics = ["validation_mae", "validation_rmse", "validation_accuracy"]

        registry = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, is_active=True
        ).first()

        if not registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        # Build comparison data
        comparison_data = []
        for version in versions:
            model_version = self.dbsession.query(ModelVersion).filter_by(
                registry_id=registry.id, version=version
            ).first()

            if model_version:
                row = {"version": version}
                for metric in metrics:
                    row[metric] = getattr(model_version, metric, None)
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def set_production_model(
        self,
        model_name: str,
        version: str,
    ) -> ModelVersion:
        """Set a specific model version as the production model"""

        registry = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, is_active=True
        ).first()

        if not registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        # Unset current production model
        current_prod = self.dbsession.query(ModelVersion).filter_by(
            registry_id=registry.id, is_production=True
        ).first()

        if current_prod:
            current_prod.is_production = False

        # Set new production model
        new_prod = self.dbsession.query(ModelVersion).filter_by(
            registry_id=registry.id, version=version
        ).first()

        if not new_prod:
            raise ModelNotFoundError(f"Version '{version}' not found for model '{model_name}'")

        new_prod.is_production = True
        new_prod.deployment_date = datetime.now().isoformat()

        self.dbsession.commit()

        logger.info(f"Set {model_name} v{version} as production model")
        return new_prod

    def get_production_model(self, model_name: str) -> Any:
        """Get the current production model"""

        registry = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, is_active=True
        ).first()

        if not registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        prod_version = self.dbsession.query(ModelVersion).filter_by(
            registry_id=registry.id, is_production=True
        ).first()

        if not prod_version:
            # Fall back to latest ready version
            prod_version = self.dbsession.query(ModelVersion).filter_by(
                registry_id=registry.id, status="ready"
            ).order_by(ModelVersion.training_date.desc()).first()

            if not prod_version:
                raise ModelNotFoundError(f"No production or ready model found for '{model_name}'")

        # Load the model
        artifact = self.dbsession.query(ModelArtifact).filter_by(
            version_id=prod_version.id, artifact_type="full_model"
        ).first()

        if not artifact:
            raise ModelStorageError(f"No artifacts found for production model {prod_version.id}")

        return self.artifact_manager.load_artifact(artifact)

    def record_performance(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        dataset_type: str = "validation",
        time_period_start: str | None = None,
        time_period_end: str | None = None,
    ) -> ModelPerformance:
        """Record performance metrics for a model version"""

        registry = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, is_active=True
        ).first()

        if not registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        model_version = self.dbsession.query(ModelVersion).filter_by(
            registry_id=registry.id, version=version
        ).first()

        if not model_version:
            raise ModelNotFoundError(f"Version '{version}' not found for model '{model_name}'")

        performance = ModelPerformance(
            version_id=model_version.id,
            evaluation_date=datetime.now().isoformat(),
            dataset_type=dataset_type,
            time_period_start=time_period_start,
            time_period_end=time_period_end,
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            prediction_correlation=metrics.get("prediction_correlation"),
            top_transfer_accuracy=metrics.get("top_transfer_accuracy"),
            points_captured=metrics.get("points_captured"),
            additional_metrics=json.dumps({k: v for k, v in metrics.items()
                                         if k not in ["mae", "rmse", "accuracy", "precision",
                                                     "recall", "f1_score", "prediction_correlation",
                                                     "top_transfer_accuracy", "points_captured"]})
        )

        self.dbsession.add(performance)
        self.dbsession.commit()

        logger.info(f"Recorded performance for {model_name} v{version}")
        return performance

    def cleanup_old_versions(
        self,
        model_name: str,
        keep_latest: int = 5,
        keep_production: bool = True,
    ) -> int:
        """Clean up old model versions and artifacts"""

        registry = self.dbsession.query(ModelRegistry).filter_by(
            model_name=model_name, is_active=True
        ).first()

        if not registry:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        # Get versions to keep
        versions_query = self.dbsession.query(ModelVersion).filter_by(registry_id=registry.id)

        keep_versions = set()

        # Keep latest versions
        latest_versions = versions_query.order_by(
            ModelVersion.training_date.desc()
        ).limit(keep_latest).all()
        keep_versions.update(v.id for v in latest_versions)

        # Keep production version
        if keep_production:
            prod_version = versions_query.filter_by(is_production=True).first()
            if prod_version:
                keep_versions.add(prod_version.id)

        # Find versions to delete
        all_versions = versions_query.all()
        delete_versions = [v for v in all_versions if v.id not in keep_versions]

        deleted_count = 0
        for version in delete_versions:
            # Delete artifacts
            artifacts = self.dbsession.query(ModelArtifact).filter_by(
                version_id=version.id
            ).all()

            for artifact in artifacts:
                if artifact.file_path and Path(artifact.file_path).exists():
                    Path(artifact.file_path).unlink()
                self.dbsession.delete(artifact)

            # Delete performance records
            performances = self.dbsession.query(ModelPerformance).filter_by(
                version_id=version.id
            ).all()
            for perf in performances:
                self.dbsession.delete(perf)

            # Delete version
            self.dbsession.delete(version)
            deleted_count += 1

        self.dbsession.commit()

        logger.info(f"Cleaned up {deleted_count} old versions for {model_name}")
        return deleted_count


# Convenience functions for easy access
def register_model(
    model: Any,
    model_name: str,
    version: str,
    **kwargs
) -> ModelVersion:
    """Convenience function to register a model version"""
    manager = ModelVersionManager()
    return manager.register_model_version(model, model_name, version, **kwargs)


def load_model(
    model_name: str,
    version: str | None = None,
) -> Any:
    """Convenience function to load a model version"""
    manager = ModelVersionManager()
    return manager.load_model_version(model_name, version)


def get_latest_model(model_name: str) -> Any:
    """Convenience function to get the latest model version"""
    return load_model(model_name)


def get_production_model(model_name: str) -> Any:
    """Convenience function to get the production model"""
    manager = ModelVersionManager()
    return manager.get_production_model(model_name)


def list_models(model_name: str | None = None) -> pd.DataFrame:
    """Convenience function to list model versions"""
    manager = ModelVersionManager()
    return manager.list_model_versions(model_name)


def compare_model_versions(
    model_name: str,
    versions: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Convenience function to compare model versions"""
    manager = ModelVersionManager()
    return manager.compare_models(model_name, versions, metrics)
