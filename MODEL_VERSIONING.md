# AIrsenal Model Versioning System

This document describes the comprehensive model versioning system implemented for AIrsenal, providing capabilities for model registry, artifact storage, performance tracking, A/B testing, and production deployment.

## Overview

The model versioning system provides:

- **Model Registry**: Track different types of models with metadata
- **Version Management**: Semantic versioning for model iterations
- **Artifact Storage**: Efficient serialization and storage of JAX/NumPyro models
- **Performance Tracking**: Monitor model performance over time
- **A/B Testing**: Compare model versions with statistical significance
- **Production Deployment**: Manage which models are used in production

## Architecture

### Core Components

1. **Database Schema**: Extended schema with model registry tables
2. **Model Versioning Module**: Core API for model management
3. **Artifact Manager**: Handles model serialization and storage
4. **A/B Testing Framework**: Statistical comparison of model versions
5. **Versioned Prediction Utils**: Integration with existing prediction pipeline
6. **CLI Commands**: Command-line interface for model management

### Database Tables

- `model_registry`: Model types and configurations
- `model_version`: Specific model versions with metadata
- `model_artifact`: Storage metadata for model files
- `model_performance`: Performance metrics tracking
- `model_experiment`: A/B testing experiments

## Quick Start

### 1. Environment Setup

Set environment variables for model storage:

```bash
export AIRSENAL_MODEL_STORAGE_DIR="/path/to/model/storage"
export AIRSENAL_MODEL_COMPRESSION="gzip"
export AIRSENAL_MODEL_MAX_VERSIONS=50
```

### 2. Train and Register a Model

```python
from airsenal.framework.model_versioning import register_model
from airsenal.framework.player_model import ConjugatePlayerModel

# Train your model
model = ConjugatePlayerModel()
# ... training code ...

# Register the model
model_version = register_model(
    model=model,
    model_name="player_model_fwd",
    version="1.0.0",
    training_params={"position": "FWD", "season": "2324"},
    performance_metrics={"mae": 0.15, "rmse": 0.25},
    notes="Initial forward player model"
)
```

### 3. Load a Model

```python
from airsenal.framework.model_versioning import load_model, get_production_model

# Load specific version
model = load_model("player_model_fwd", "1.0.0")

# Load latest version
latest_model = load_model("player_model_fwd")

# Load production model
prod_model = get_production_model("player_model_fwd")
```

### 4. Run Versioned Predictions

```python
from airsenal.framework.versioned_prediction_utils import VersionedPredictionManager

manager = VersionedPredictionManager()

# Run predictions with specific model versions
tag = manager.run_versioned_predictions(
    gw_range=[1, 2, 3],
    season="2425",
    player_model_versions={"fwd": "1.0.0", "mid": "1.1.0"},
    team_model_version="2.0.0"
)
```

## CLI Usage

### Model Management

List available models:
```bash
python -m airsenal.scripts.model_management model list
python -m airsenal.scripts.model_management model list --model-name player_model_fwd
```

Compare model versions:
```bash
python -m airsenal.scripts.model_management model compare player_model_fwd 1.0.0 1.1.0 2.0.0
```

Set production model:
```bash
python -m airsenal.scripts.model_management model set-production player_model_fwd 1.1.0 --confirm
```

Train and register new model:
```bash
python -m airsenal.scripts.model_management model train-and-register \
    player_model_fwd player --position FWD --version 1.2.0 --notes "Updated with new features"
```

### A/B Testing

Create an experiment:
```bash
python -m airsenal.scripts.model_management experiment create \
    fwd_v1_vs_v2 player_model_fwd 1.0.0 1.1.0 \
    --traffic-split 0.6 --description "Test improved FWD model"
```

Run experiment iteration:
```bash
python -m airsenal.scripts.model_management experiment run 1 --weeks-ahead 3
```

Analyze results:
```bash
python -m airsenal.scripts.model_management experiment analyze 1 \
    control_tag_123 --treatment-tags treatment_tag_456 \
    --output-file experiment_results.json
```

### Enhanced Pipeline

Run pipeline with versioned models:
```bash
python -m airsenal.scripts.airsenal_run_versioned_pipeline \
    --weeks_ahead 3 --use-versioned-models --use-production-models
```

Run pipeline with A/B test:
```bash
python -m airsenal.scripts.airsenal_run_versioned_pipeline \
    --weeks_ahead 3 --ab-test 1.0.0 1.1.0 --experiment-name weekly_test
```

## Advanced Usage

### Custom Model Registration

```python
from airsenal.framework.model_versioning import ModelVersionManager
from datetime import datetime

manager = ModelVersionManager()

# Register with detailed metadata
model_version = manager.register_model_version(
    model=my_model,
    model_name="custom_player_model",
    version="1.0.0-beta",
    training_params={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "features": ["minutes", "goals", "assists", "bonus"]
    },
    feature_set="Extended player features with team context",
    training_duration_seconds=3600,
    performance_metrics={
        "mae": 0.145,
        "rmse": 0.234,
        "prediction_correlation": 0.87,
        "top_transfer_accuracy": 0.72
    },
    notes="Beta version with new feature engineering",
    tags=["beta", "enhanced_features", "experimental"]
)
```

### Performance Tracking

```python
# Record additional performance metrics
manager.record_performance(
    model_name="player_model_fwd",
    version="1.0.0",
    metrics={
        "mae": 0.142,
        "prediction_correlation": 0.89,
        "points_captured": 0.85,
        "gameweek_consistency": 0.78
    },
    dataset_type="production",
    time_period_start="2024-01-01T00:00:00",
    time_period_end="2024-01-31T23:59:59"
)
```

### A/B Testing with Custom Analysis

```python
from airsenal.framework.ab_testing import ABTestManager, ExperimentConfig

ab_manager = ABTestManager()

# Create sophisticated experiment
config = ExperimentConfig(
    experiment_name="player_model_improvement",
    model_name="player_model_mid",
    control_version="1.0.0",
    treatment_version="1.1.0",
    traffic_split=0.5,
    min_sample_size=200,
    max_duration_days=7,
    significance_level=0.01,
    minimum_effect_size=0.05,
    early_stopping=True,
    success_metrics=["mae", "prediction_correlation", "points_captured"]
)

experiment = ab_manager.create_experiment(config)
ab_manager.start_experiment(experiment.id)

# Run multiple iterations and analyze
for week in range(3):
    results = ab_manager.run_experiment_iteration(experiment.id, [week+1], "2425")
    
    # Analyze after sufficient data
    if week >= 1:
        analysis = ab_manager.analyze_experiment_results(
            experiment.id,
            [results["control_tag"]],
            [results["treatment_tag"]]
        )
        
        # Check for early stopping
        should_stop, reason = ab_manager.check_early_stopping(experiment.id, analysis, config)
        if should_stop:
            ab_manager.stop_experiment(experiment.id, reason)
            break
```

### Model Cleanup and Maintenance

```python
# Cleanup old versions
deleted_count = manager.cleanup_old_versions(
    model_name="player_model_fwd",
    keep_latest=5,
    keep_production=True
)

# Get model history
history = manager.list_model_versions("player_model_fwd", limit=10)
print(history)

# Compare multiple versions
comparison = manager.compare_models(
    "player_model_fwd",
    ["1.0.0", "1.1.0", "1.2.0"],
    ["validation_mae", "validation_rmse", "prediction_correlation"]
)
print(comparison)
```

## Best Practices

### Versioning Strategy

1. **Semantic Versioning**: Use MAJOR.MINOR.PATCH format
   - MAJOR: Incompatible changes (new model architecture)
   - MINOR: New features (additional input features)
   - PATCH: Bug fixes (parameter tuning)

2. **Version Tags**: Use descriptive tags
   - `stable`, `beta`, `experimental`
   - `baseline`, `enhanced`, `optimized`
   - `season_2425`, `gameweek_10`

### Model Registration

1. Always include performance metrics when registering
2. Document training parameters and feature sets
3. Add meaningful notes and tags for searchability
4. Record training duration for cost tracking

### Production Deployment

1. Test models thoroughly before production deployment
2. Use A/B testing for gradual rollouts
3. Monitor production performance continuously
4. Have rollback procedures in place

### A/B Testing

1. Define success metrics before starting experiments
2. Ensure sufficient sample sizes for statistical power
3. Run experiments for adequate duration
4. Document experimental design and results

## Configuration

### Environment Variables

- `AIRSENAL_MODEL_STORAGE_DIR`: Directory for model artifacts
- `AIRSENAL_MODEL_COMPRESSION`: Compression algorithm (gzip, bzip2, none)
- `AIRSENAL_MODEL_MAX_VERSIONS`: Maximum versions to keep per model
- `AIRSENAL_MODEL_S3_BUCKET`: S3 bucket for cloud storage (optional)
- `AIRSENAL_MODEL_ENABLE_S3`: Enable S3 storage (true/false)

### Storage Configuration

```python
from airsenal.framework.model_versioning import ModelVersioningConfig

config = ModelVersioningConfig()
config.model_storage_dir = Path("/custom/model/storage")
config.enable_compression = True
config.default_compression = "gzip"
config.max_versions_per_model = 20
config.enable_s3_storage = True
config.s3_bucket = "my-model-bucket"
```

## Testing

Run the test suite:

```bash
pytest airsenal/tests/test_model_versioning.py -v
```

Test specific components:

```bash
# Test model registration
pytest airsenal/tests/test_model_versioning.py::TestModelVersionManager::test_register_model_version

# Test A/B testing
pytest airsenal/tests/test_model_versioning.py::TestABTestManager

# Test artifact storage
pytest airsenal/tests/test_model_versioning.py::TestModelArtifactManager
```

## Troubleshooting

### Common Issues

1. **Storage Directory Permissions**
   ```bash
   chmod 755 /path/to/model/storage
   chown -R airsenal:airsenal /path/to/model/storage
   ```

2. **Model Loading Errors**
   - Check that model version exists
   - Verify artifact files are not corrupted
   - Ensure JAX version compatibility

3. **Database Migration**
   - Backup database before schema changes
   - Run database migration if upgrading from older versions

4. **Memory Issues with Large Models**
   - Enable compression for storage
   - Consider using S3 for large artifacts
   - Monitor memory usage during training

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('airsenal.framework.model_versioning').setLevel(logging.DEBUG)
logging.getLogger('airsenal.framework.ab_testing').setLevel(logging.DEBUG)
```

Check model registry:

```python
from airsenal.framework.model_versioning import ModelVersionManager

manager = ModelVersionManager()
df = manager.list_model_versions()
print(df.to_string())
```

## Migration from Existing System

If you have existing models, you can register them retroactively:

```python
# Register existing fitted model
existing_model = ConjugatePlayerModel()
# ... load existing model state ...

register_model(
    model=existing_model,
    model_name="legacy_player_model",
    version="0.9.0",
    notes="Migrated from legacy system",
    tags=["legacy", "migration"]
)
```

## Future Enhancements

Planned features for future versions:

1. **Model Ensemble Support**: Register and manage ensemble models
2. **Automated Hyperparameter Tuning**: Integration with optimization libraries
3. **Model Drift Detection**: Automatic monitoring of model performance degradation
4. **Multi-environment Support**: Development, staging, production environments
5. **Model Lineage Tracking**: Track data and code changes affecting models
6. **Integration with MLOps Tools**: Support for MLflow, Weights & Biases, etc.

## Contributing

To contribute to the model versioning system:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Consider backward compatibility
5. Add appropriate error handling and logging

## Support

For issues with the model versioning system:

1. Check the troubleshooting section above
2. Review test cases for usage examples
3. Search existing GitHub issues
4. Create a new issue with detailed information about the problem