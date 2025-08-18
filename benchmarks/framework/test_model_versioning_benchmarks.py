"""
Model Versioning Performance Benchmarks

Tests performance of the Model Versioning system including:
- Model registration and metadata operations
- Model serialization and deserialization
- Artifact storage and retrieval
- Version comparison and rollback
- Model loading times
"""

import contextlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from airsenal.framework.model_versioning import ModelVersionManager
from benchmarks.utils import BenchmarkRunner


class ModelVersioningBenchmarks:
    """Model versioning performance benchmark suite."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.model_manager = runner.model_manager
        self.config = runner.config

        # Test models and metadata
        self.test_models = self._create_test_models()
        self.test_metadata = {
            "name": "benchmark_test_model",
            "description": "Test model for benchmarking",
            "model_type": "player_prediction",
            "framework": "jax_numpyro",
            "version": "v1.0.0",
            "tags": ["benchmark", "test"],
            "hyperparameters": {
                "learning_rate": 0.01,
                "num_samples": 1000,
                "num_warmup": 500,
            },
        }

    def _create_test_models(self):
        """Create test models for benchmarking."""
        # Create mock model states/artifacts
        test_models = {}

        # Small model (simulated)
        test_models["small"] = {
            "weights": np.random.normal(0, 1, (10, 5)),
            "biases": np.random.normal(0, 0.1, 5),
            "metadata": {"parameters": 55, "size_mb": 0.001},
        }

        # Medium model
        test_models["medium"] = {
            "weights": np.random.normal(0, 1, (100, 50)),
            "biases": np.random.normal(0, 0.1, 50),
            "metadata": {"parameters": 5050, "size_mb": 0.04},
        }

        # Large model
        test_models["large"] = {
            "weights": np.random.normal(0, 1, (1000, 500)),
            "biases": np.random.normal(0, 0.1, 500),
            "metadata": {"parameters": 500500, "size_mb": 4.0},
        }

        return test_models

    def run_all(self):
        """Run all model versioning benchmarks."""
        if not self.model_manager:
            print("Model manager not available - skipping benchmarks")
            return

        self.benchmark_model_registration()
        self.benchmark_model_serialization()
        self.benchmark_model_loading()
        self.benchmark_artifact_storage()
        self.benchmark_version_operations()
        self.benchmark_metadata_queries()
        self.benchmark_model_comparison()

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_model_registration(self):
        """Benchmark model registration performance."""
        with self.runner.benchmark_context("model_register"):
            for i in range(10):
                metadata = self.test_metadata.copy()
                metadata["name"] = f"benchmark_model_{i}"
                metadata["version"] = f"v1.{i}.0"

                try:
                    self.model_manager.register_model(
                        name=metadata["name"],
                        version=metadata["version"],
                        model_type=metadata["model_type"],
                        framework=metadata["framework"],
                        description=metadata["description"],
                        tags=metadata["tags"],
                        hyperparameters=metadata["hyperparameters"],
                    )
                except Exception:
                    pass  # Model might already exist

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_model_serialization(self):
        """Benchmark model serialization performance."""
        for model_size in ["small", "medium", "large"]:
            model_data = self.test_models[model_size]

            with self.runner.benchmark_context(f"model_serialize_{model_size}"):
                # Simulate model serialization
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                    # Pickle serialization (JAX models use this)
                    import pickle

                    pickle.dump(model_data, f)
                    temp_file = f.name

                # Clean up
                Path(temp_file).unlink(missing_ok=True)

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_model_loading(self):
        """Benchmark model loading performance."""
        # First, save models to load
        saved_models = {}
        for model_size in ["small", "medium", "large"]:
            model_data = self.test_models[model_size]

            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                import pickle

                pickle.dump(model_data, f)
                saved_models[model_size] = f.name

        # Benchmark loading
        for model_size in ["small", "medium", "large"]:
            with self.runner.benchmark_context(f"model_load_{model_size}"):
                with open(saved_models[model_size], "rb") as f:
                    import pickle

                    pickle.load(f)

        # Clean up
        for temp_file in saved_models.values():
            Path(temp_file).unlink(missing_ok=True)

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_artifact_storage(self):
        """Benchmark model artifact storage operations."""
        model_data = self.test_models["medium"]

        # Benchmark artifact saving
        with self.runner.benchmark_context("model_artifact_save"):
            try:
                artifact_id = self.model_manager.save_model_artifact(
                    model_id=1,  # Assume model exists
                    artifact_data=model_data,
                    artifact_type="weights",
                    compression=True,
                )
            except Exception:
                artifact_id = None

        # Benchmark artifact loading
        if artifact_id:
            with self.runner.benchmark_context("model_artifact_load"):
                with contextlib.suppress(Exception):
                    self.model_manager.load_model_artifact(artifact_id)

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_version_operations(self):
        """Benchmark version management operations."""
        # Benchmark version listing
        with self.runner.benchmark_context("model_version_list"):
            with contextlib.suppress(Exception):
                self.model_manager.list_model_versions(
                    model_name="benchmark_model_0", limit=100
                )

        # Benchmark version comparison
        with self.runner.benchmark_context("model_version_compare"):
            with contextlib.suppress(Exception):
                self.model_manager.compare_model_versions(
                    model_name="benchmark_model_0", version1="v1.0.0", version2="v1.1.0"
                )

        # Benchmark model rollback
        with self.runner.benchmark_context("model_version_rollback"):
            with contextlib.suppress(Exception):
                self.model_manager.rollback_model(
                    model_name="benchmark_model_0", target_version="v1.0.0"
                )

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_metadata_queries(self):
        """Benchmark metadata query operations."""
        # Benchmark model search by tags
        with self.runner.benchmark_context("model_search_tags"):
            with contextlib.suppress(Exception):
                self.model_manager.search_models(
                    tags=["benchmark"], model_type="player_prediction"
                )

        # Benchmark performance metrics query
        with self.runner.benchmark_context("model_performance_query"):
            with contextlib.suppress(Exception):
                self.model_manager.get_model_performance(
                    model_name="benchmark_model_0",
                    version="v1.0.0",
                    metric_names=["accuracy", "rmse", "mae"],
                )

        # Benchmark model lineage query
        with self.runner.benchmark_context("model_lineage_query"):
            with contextlib.suppress(Exception):
                self.model_manager.get_model_lineage(model_name="benchmark_model_0")

    @pytest.mark.benchmark(group="model_versioning")
    def benchmark_model_comparison(self):
        """Benchmark model comparison operations."""
        model1 = self.test_models["medium"]
        model2 = self.test_models["large"]

        with self.runner.benchmark_context("model_weights_comparison"):
            # Simulate model weight comparison
            if "weights" in model1 and "weights" in model2:
                # Compare compatible shapes only
                min_shape = min(model1["weights"].shape[0], model2["weights"].shape[0])

                diff = np.abs(
                    model1["weights"][
                        :min_shape,
                        : min(model1["weights"].shape[1], model2["weights"].shape[1]),
                    ]
                    - model2["weights"][
                        :min_shape,
                        : min(model1["weights"].shape[1], model2["weights"].shape[1]),
                    ]
                )

                {
                    "mean_diff": np.mean(diff),
                    "max_diff": np.max(diff),
                    "std_diff": np.std(diff),
                }

    def benchmark_model_export_import(self):
        """Benchmark model export/import operations."""
        self.test_models["medium"]

        # Benchmark model export
        with self.runner.benchmark_context("model_export"):
            try:
                export_path = self.model_manager.export_model(
                    model_name="benchmark_model_0",
                    version="v1.0.0",
                    format="onnx",  # or "pickle", "jax"
                )
            except Exception:
                export_path = None

        # Benchmark model import
        if export_path and Path(export_path).exists():
            with self.runner.benchmark_context("model_import"):
                with contextlib.suppress(Exception):
                    self.model_manager.import_model(
                        import_path=export_path, model_name="benchmark_imported_model"
                    )

            # Clean up
            Path(export_path).unlink(missing_ok=True)

    def stress_test_concurrent_operations(self):
        """Stress test concurrent model operations."""
        import concurrent.futures

        def model_operations(thread_id: int):
            """Perform model operations in separate thread."""
            try:
                # Register model
                model_name = f"stress_model_{thread_id}"
                self.model_manager.register_model(
                    name=model_name,
                    version="v1.0.0",
                    model_type="player_prediction",
                    framework="jax_numpyro",
                )

                # Save artifact
                self.model_manager.save_model_artifact(
                    model_id=thread_id,
                    artifact_data=self.test_models["small"],
                    artifact_type="weights",
                )

                # Query model
                self.model_manager.list_model_versions(model_name)

                return 1
            except Exception:
                return 0

        with self.runner.benchmark_context("model_concurrent_stress"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(model_operations, i) for i in range(50)]

                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

        # Log success rate
        success_rate = sum(results) / len(results)
        self.runner.results[-1]["success_rate"] = success_rate


# Standalone pytest functions for pytest-benchmark
@pytest.mark.benchmark(group="model_versioning_single")
def test_model_registration(benchmark):
    """Pytest-benchmark test for model registration."""
    model_manager = ModelVersionManager()

    def register_model():
        return model_manager.register_model(
            name="pytest_benchmark_model",
            version="v1.0.0",
            model_type="player_prediction",
            framework="jax_numpyro",
            description="Benchmark test model",
        )

    try:
        benchmark(register_model)
    except Exception:
        # Model might already exist
        pass


@pytest.mark.benchmark(group="model_versioning_single")
def test_model_serialization(benchmark):
    """Pytest-benchmark test for model serialization."""
    import pickle
    import tempfile

    # Create test model data
    model_data = {
        "weights": np.random.normal(0, 1, (100, 50)),
        "biases": np.random.normal(0, 0.1, 50),
        "metadata": {"parameters": 5050},
    }

    def serialize_model():
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(model_data, f)
            temp_file = f.name

        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        return temp_file

    benchmark(serialize_model)


@pytest.mark.benchmark(group="model_versioning_batch")
def test_version_operations(benchmark):
    """Pytest-benchmark test for version operations."""
    model_manager = ModelVersionManager()

    def version_operations():
        try:
            # List versions
            model_manager.list_model_versions(
                model_name="pytest_benchmark_model", limit=50
            )

            # Search models
            models = model_manager.search_models(model_type="player_prediction")

            return len(models) if models else 0
        except Exception:
            return 0

    benchmark(version_operations)
