"""
Factory_boy factories for Model Versioning components.

Provides realistic test data generation for model registry, versions, artifacts,
performance tracking, and A/B testing experiments.
"""

import factory
import json
import random
import hashlib
from datetime import datetime, timedelta
from factory import fuzzy
from typing import Dict, List, Any

from airsenal.framework.schema import (
    ModelRegistry,
    ModelVersion, 
    ModelArtifact,
    ModelPerformance,
    ModelExperiment,
)


class ModelRegistryFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating ModelRegistry instances."""
    
    class Meta:
        model = ModelRegistry
        sqlalchemy_session_persistence = "commit"
    
    model_name = factory.fuzzy.FuzzyChoice([
        "player_model_gk", "player_model_def", "player_model_mid", "player_model_fwd",
        "team_model", "injury_model", "transfer_model", "captaincy_model",
        "squad_optimizer", "fixture_difficulty_model", "form_predictor"
    ])
    
    model_type = factory.LazyAttribute(
        lambda obj: _get_model_type_for_name(obj.model_name)
    )
    
    description = factory.LazyAttribute(
        lambda obj: _get_model_description(obj.model_name)
    )
    
    created_at = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
    )
    
    created_by = factory.fuzzy.FuzzyChoice([
        "airsenal_system", "data_scientist_alice", "ml_engineer_bob", 
        "developer_charlie", "automated_training"
    ])
    
    is_active = True


class ModelVersionFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating ModelVersion instances with realistic training metadata."""
    
    class Meta:
        model = ModelVersion
        sqlalchemy_session_persistence = "commit"
    
    registry = factory.SubFactory(ModelRegistryFactory)
    
    version = factory.LazyFunction(lambda: _generate_semantic_version())
    
    config_hash = factory.LazyFunction(
        lambda: hashlib.sha256(f"config_{random.randint(1000, 9999)}".encode()).hexdigest()[:16]
    )
    
    training_data_version = factory.LazyFunction(
        lambda: f"data_v{random.randint(1, 50)}.{random.randint(0, 9)}"
    )
    
    # Training metadata with realistic JSON parameters
    training_params = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_training_params(obj.registry.model_type))
    )
    
    feature_set = factory.LazyAttribute(
        lambda obj: _generate_feature_set_description(obj.registry.model_name)
    )
    
    training_date = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()
    )
    
    training_duration_seconds = factory.fuzzy.FuzzyInteger(60, 7200)  # 1 min to 2 hours
    
    # Performance metrics with realistic ranges
    validation_mae = factory.LazyAttribute(
        lambda obj: _generate_mae_for_model(obj.registry.model_name)
    )
    
    validation_rmse = factory.LazyAttribute(
        lambda obj: obj.validation_mae * random.uniform(1.2, 1.8)  # RMSE > MAE
    )
    
    validation_accuracy = factory.LazyAttribute(
        lambda obj: _generate_accuracy_for_model(obj.registry.model_type)
    )
    
    cross_validation_score = factory.LazyAttribute(
        lambda obj: obj.validation_accuracy - random.uniform(0.02, 0.08) if obj.validation_accuracy else None
    )
    
    # Status and deployment
    status = factory.fuzzy.FuzzyChoice([
        "ready", "ready", "ready", "ready",  # Most models should be ready
        "training", "deployed", "deprecated", "failed"
    ], weights=[4, 1, 1, 1, 1])
    
    is_production = False  # Most models are not in production initially
    
    deployment_date = factory.Maybe(
        "is_production",
        yes_declaration=factory.LazyFunction(
            lambda: (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
        ),
        no_declaration=None
    )
    
    # Metadata
    notes = factory.LazyAttribute(
        lambda obj: _generate_model_notes(obj.registry.model_name, obj.version)
    )
    
    tags = factory.LazyAttribute(
        lambda obj: _generate_model_tags(obj.registry.model_name)
    )


class ProductionModelVersionFactory(ModelVersionFactory):
    """Factory for creating production model versions."""
    
    status = "deployed"
    is_production = True
    deployment_date = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
    )
    
    # Production models should have good performance
    validation_mae = factory.LazyAttribute(
        lambda obj: _generate_mae_for_model(obj.registry.model_name) * 0.8  # 20% better
    )
    
    tags = factory.LazyAttribute(
        lambda obj: f"{_generate_model_tags(obj.registry.model_name)},production,stable"
    )


class BetaModelVersionFactory(ModelVersionFactory):
    """Factory for creating beta/experimental model versions."""
    
    version = factory.LazyFunction(lambda: f"{_generate_semantic_version()}-beta")
    status = "ready"
    
    tags = factory.LazyAttribute(
        lambda obj: f"{_generate_model_tags(obj.registry.model_name)},beta,experimental"
    )
    
    notes = factory.LazyAttribute(
        lambda obj: f"Beta version of {obj.registry.model_name} with experimental features. {_generate_model_notes(obj.registry.model_name, obj.version)}"
    )


class ModelArtifactFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating ModelArtifact instances."""
    
    class Meta:
        model = ModelArtifact
        sqlalchemy_session_persistence = "commit"
    
    version = factory.SubFactory(ModelVersionFactory)
    
    artifact_type = factory.fuzzy.FuzzyChoice([
        "full_model", "model_weights", "metadata", "training_state", 
        "feature_importance", "validation_results", "hyperparameters"
    ])
    
    file_path = factory.LazyAttribute(
        lambda obj: f"/models/{obj.version.registry.model_name}/v{obj.version.version}/{obj.artifact_type}.pkl"
    )
    
    s3_path = factory.LazyAttribute(
        lambda obj: f"s3://airsenal-models/{obj.version.registry.model_name}/v{obj.version.version}/{obj.artifact_type}.pkl"
    )
    
    file_size_bytes = factory.fuzzy.FuzzyInteger(1024, 50_000_000)  # 1KB to 50MB
    
    checksum = factory.LazyFunction(
        lambda: hashlib.sha256(f"artifact_{random.randint(10000, 99999)}".encode()).hexdigest()
    )
    
    compression = factory.fuzzy.FuzzyChoice(["gzip", "bzip2", "none"], weights=[3, 1, 2])
    
    serialization_format = factory.LazyAttribute(
        lambda obj: _get_serialization_format(obj.version.registry.model_type)
    )
    
    created_at = factory.LazyFunction(lambda: datetime.now().isoformat())


class ModelPerformanceFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating ModelPerformance instances with realistic metrics."""
    
    class Meta:
        model = ModelPerformance
        sqlalchemy_session_persistence = "commit"
    
    version = factory.SubFactory(ModelVersionFactory)
    
    evaluation_date = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
    )
    
    dataset_type = factory.fuzzy.FuzzyChoice([
        "validation", "test", "production", "backtest"
    ])
    
    time_period_start = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(30, 90))).isoformat()
    )
    
    time_period_end = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
    )
    
    # Core metrics
    mae = factory.LazyAttribute(
        lambda obj: _generate_mae_for_model(obj.version.registry.model_name)
    )
    
    rmse = factory.LazyAttribute(
        lambda obj: obj.mae * random.uniform(1.2, 1.8)
    )
    
    accuracy = factory.LazyAttribute(
        lambda obj: _generate_accuracy_for_model(obj.version.registry.model_type)
    )
    
    precision = factory.LazyAttribute(
        lambda obj: obj.accuracy + random.uniform(-0.05, 0.05) if obj.accuracy else None
    )
    
    recall = factory.LazyAttribute(
        lambda obj: obj.precision + random.uniform(-0.03, 0.03) if obj.precision else None
    )
    
    f1_score = factory.LazyAttribute(
        lambda obj: _calculate_f1_score(obj.precision, obj.recall) if obj.precision and obj.recall else None
    )
    
    # Domain-specific metrics
    prediction_correlation = factory.fuzzy.FuzzyFloat(0.3, 0.9)
    
    top_transfer_accuracy = factory.LazyAttribute(
        lambda obj: random.uniform(0.4, 0.8) if "transfer" in obj.version.registry.model_name else None
    )
    
    points_captured = factory.LazyAttribute(
        lambda obj: random.uniform(0.6, 0.95) if "player" in obj.version.registry.model_name else None
    )
    
    # Additional metrics as JSON
    additional_metrics = factory.LazyAttribute(
        lambda obj: json.dumps(_generate_additional_metrics(obj.version.registry.model_name))
    )


class ModelExperimentFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Factory for creating A/B testing experiments."""
    
    class Meta:
        model = ModelExperiment
        sqlalchemy_session_persistence = "commit"
    
    experiment_name = factory.LazyFunction(
        lambda: f"experiment_{random.choice(['accuracy', 'performance', 'feature', 'hyperopt'])}_{random.randint(1000, 9999)}"
    )
    
    description = factory.LazyAttribute(
        lambda obj: f"A/B testing experiment: {obj.experiment_name}. Comparing model performance on production traffic."
    )
    
    model_version = factory.SubFactory(ModelVersionFactory)
    
    control_version_id = factory.LazyAttribute(
        lambda obj: obj.model_version.id - 1 if obj.model_version.id > 1 else None
    )
    
    traffic_split = factory.fuzzy.FuzzyFloat(0.1, 0.9)
    
    start_date = factory.LazyFunction(
        lambda: (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
    )
    
    end_date = factory.LazyFunction(
        lambda: (datetime.now() + timedelta(days=random.randint(1, 14))).isoformat()
    )
    
    status = factory.fuzzy.FuzzyChoice([
        "planned", "running", "completed", "stopped"
    ], weights=[1, 3, 2, 1])
    
    winner_version_id = factory.Maybe(
        "status",
        yes_declaration=factory.SelfAttribute("model_version.id"),
        no_declaration=None
    )
    
    confidence_level = factory.Maybe(
        "winner_version_id",
        yes_declaration=factory.fuzzy.FuzzyFloat(0.8, 0.99),
        no_declaration=None
    )
    
    created_by = factory.fuzzy.FuzzyChoice([
        "ml_engineer", "data_scientist", "automated_system", "product_manager"
    ])
    
    notes = factory.LazyAttribute(
        lambda obj: f"Experiment testing {obj.model_version.registry.model_name} v{obj.model_version.version}"
    )


# Helper functions for realistic data generation
def _get_model_type_for_name(model_name: str) -> str:
    """Return appropriate model type based on model name."""
    type_mapping = {
        "player_model": "NumpyroPlayerModel",
        "team_model": "ExtendedDixonColesMatchPredictor", 
        "injury_model": "InjuryPredictionModel",
        "transfer_model": "TransferOptimizer",
        "captaincy_model": "CaptaincySelector",
        "squad_optimizer": "GeneticSquadOptimizer",
        "fixture_difficulty_model": "FixtureDifficultyPredictor",
        "form_predictor": "FormTrendPredictor",
    }
    
    for name_part, model_type in type_mapping.items():
        if name_part in model_name:
            return model_type
    
    return "BasePlayerModel"


def _get_model_description(model_name: str) -> str:
    """Generate realistic model description."""
    descriptions = {
        "player_model_gk": "Bayesian model for predicting goalkeeper performance and points",
        "player_model_def": "Defensive player performance prediction using xG and defensive metrics",
        "player_model_mid": "Midfielder points prediction incorporating creativity and attacking metrics",
        "player_model_fwd": "Forward performance model focusing on goal scoring and assist potential",
        "team_model": "Team-level performance prediction using Dixon-Coles approach",
        "injury_model": "Player injury risk assessment and return date prediction",
        "transfer_model": "Optimal transfer suggestion engine using genetic algorithms",
        "captaincy_model": "Weekly captaincy selection based on expected points and ownership",
        "squad_optimizer": "15-player squad optimization for season-long performance",
        "fixture_difficulty_model": "Team strength and fixture difficulty rating system",
        "form_predictor": "Short-term form prediction using rolling performance metrics",
    }
    
    for name_part, description in descriptions.items():
        if name_part in model_name:
            return description
    
    return f"Machine learning model for {model_name.replace('_', ' ')} optimization"


def _generate_semantic_version() -> str:
    """Generate realistic semantic version numbers."""
    major = random.randint(1, 3)
    minor = random.randint(0, 12)
    patch = random.randint(0, 20)
    return f"{major}.{minor}.{patch}"


def _generate_training_params(model_type: str) -> Dict[str, Any]:
    """Generate realistic training parameters based on model type."""
    base_params = {
        "learning_rate": round(random.uniform(0.001, 0.1), 4),
        "batch_size": random.choice([16, 32, 64, 128]),
        "epochs": random.randint(50, 500),
        "random_seed": random.randint(1, 10000),
    }
    
    if "Numpyro" in model_type:
        base_params.update({
            "num_samples": random.randint(1000, 5000),
            "num_warmup": random.randint(500, 2000),
            "num_chains": random.choice([2, 4, 8]),
            "target_accept_prob": round(random.uniform(0.8, 0.95), 2),
        })
    elif "Genetic" in model_type:
        base_params.update({
            "population_size": random.choice([50, 100, 200]),
            "mutation_rate": round(random.uniform(0.01, 0.1), 3),
            "crossover_rate": round(random.uniform(0.6, 0.9), 2),
            "generations": random.randint(100, 1000),
        })
    elif "DixonColes" in model_type:
        base_params.update({
            "xi": round(random.uniform(0.0, 0.1), 3),
            "tau": round(random.uniform(0.01, 0.05), 3),
            "rho": round(random.uniform(-0.5, 0.5), 3),
        })
    
    return base_params


def _generate_feature_set_description(model_name: str) -> str:
    """Generate description of features used in the model."""
    feature_sets = {
        "player_model": "Player stats, form metrics, fixture difficulty, team performance, injury history",
        "team_model": "Team ratings, recent results, head-to-head records, player availability",
        "injury_model": "Player age, position, injury history, workload, physical metrics",
        "transfer_model": "Player value, form, fixtures, ownership, price changes",
        "captaincy_model": "Expected points, ownership %, fixture difficulty, home advantage",
        "squad_optimizer": "Player prices, predicted points, position constraints, budget limits",
        "fixture_difficulty": "Team strength ratings, home advantage, recent form, head-to-head",
        "form_predictor": "Rolling averages, momentum indicators, underlying stats, fixture-adjusted metrics",
    }
    
    for name_part, features in feature_sets.items():
        if name_part in model_name:
            return features
    
    return "Standard player and team performance metrics"


def _generate_mae_for_model(model_name: str) -> float:
    """Generate realistic MAE values based on model type."""
    mae_ranges = {
        "player_model": (0.8, 2.5),
        "team_model": (0.5, 1.5), 
        "injury_model": (0.1, 0.4),
        "transfer_model": (1.0, 3.0),
        "captaincy_model": (2.0, 5.0),
        "squad_optimizer": (5.0, 15.0),
        "fixture_difficulty": (0.3, 0.8),
        "form_predictor": (0.5, 1.5),
    }
    
    for name_part, (min_mae, max_mae) in mae_ranges.items():
        if name_part in model_name:
            return round(random.uniform(min_mae, max_mae), 3)
    
    return round(random.uniform(0.5, 2.0), 3)


def _generate_accuracy_for_model(model_type: str) -> float:
    """Generate realistic accuracy values for classification models."""
    if any(keyword in model_type.lower() for keyword in ["classification", "selector", "predictor"]):
        return round(random.uniform(0.65, 0.92), 3)
    return None  # Regression models don't have accuracy


def _calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * (precision * recall) / (precision + recall), 3)


def _get_serialization_format(model_type: str) -> str:
    """Return appropriate serialization format based on model type."""
    if "Numpyro" in model_type or "Jax" in model_type:
        return "jax"
    elif "Genetic" in model_type or "Optimizer" in model_type:
        return "pickle"
    elif "DixonColes" in model_type:
        return "joblib"
    else:
        return "pickle"


def _generate_additional_metrics(model_name: str) -> Dict[str, float]:
    """Generate additional domain-specific metrics."""
    base_metrics = {
        "convergence_score": round(random.uniform(0.8, 1.0), 3),
        "stability_index": round(random.uniform(0.7, 0.95), 3),
    }
    
    if "player" in model_name:
        base_metrics.update({
            "top_player_accuracy": round(random.uniform(0.6, 0.85), 3),
            "differential_picks_success": round(random.uniform(0.4, 0.7), 3),
            "captain_recommendation_accuracy": round(random.uniform(0.5, 0.8), 3),
        })
    elif "transfer" in model_name:
        base_metrics.update({
            "transfer_hit_efficiency": round(random.uniform(0.3, 0.7), 3),
            "budget_utilization": round(random.uniform(0.85, 0.98), 3),
            "timing_accuracy": round(random.uniform(0.4, 0.75), 3),
        })
    elif "team" in model_name:
        base_metrics.update({
            "correct_result_percentage": round(random.uniform(0.45, 0.65), 3),
            "goal_difference_accuracy": round(random.uniform(0.35, 0.55), 3),
            "clean_sheet_prediction_accuracy": round(random.uniform(0.6, 0.8), 3),
        })
    
    return base_metrics


def _generate_model_notes(model_name: str, version: str) -> str:
    """Generate realistic model notes."""
    notes_templates = [
        f"Improved {model_name} with enhanced feature engineering and hyperparameter optimization.",
        f"Updated {model_name} v{version} with additional data sources and better preprocessing.",
        f"Experimental version of {model_name} testing new algorithm approaches.",
        f"Production-ready {model_name} with validated performance on historical data.",
        f"Maintenance update for {model_name} - bug fixes and stability improvements.",
    ]
    
    base_note = random.choice(notes_templates)
    
    # Add specific details based on model type
    if "player" in model_name:
        base_note += " Incorporates latest xG data and form metrics."
    elif "transfer" in model_name:
        base_note += " Enhanced with fixture difficulty weighting."
    elif "team" in model_name:
        base_note += " Updated with latest team strength calculations."
    
    return base_note


def _generate_model_tags(model_name: str) -> str:
    """Generate realistic model tags."""
    tags = []
    
    # Add category tags
    if "player" in model_name:
        tags.append("player_modeling")
    if "team" in model_name:
        tags.append("team_modeling")
    if "transfer" in model_name:
        tags.append("optimization")
    
    # Add algorithm tags
    tags.extend(random.sample([
        "machine_learning", "bayesian", "statistical", "predictive", 
        "optimization", "ensemble", "feature_engineering"
    ], k=random.randint(2, 4)))
    
    # Add domain tags
    tags.extend(random.sample([
        "fpl", "fantasy_football", "points_prediction", "performance_analysis"
    ], k=random.randint(1, 2)))
    
    return ",".join(tags)


# Batch creation functions for testing scenarios
def create_model_family(session, model_name: str, num_versions: int = 5) -> List[ModelVersion]:
    """Create a family of related model versions."""
    registry = ModelRegistryFactory(session=session, model_name=model_name)
    
    versions = []
    for i in range(num_versions):
        version = ModelVersionFactory(session=session, registry=registry)
        
        # Create artifacts for each version
        for artifact_type in ["full_model", "metadata"]:
            ModelArtifactFactory(session=session, version=version, artifact_type=artifact_type)
        
        # Create performance records
        for dataset_type in ["validation", "test"]:
            ModelPerformanceFactory(session=session, version=version, dataset_type=dataset_type)
        
        versions.append(version)
    
    # Set one version as production
    if versions:
        production_version = random.choice(versions[-3:])  # Choose from recent versions
        production_version.is_production = True
        production_version.status = "deployed"
        production_version.deployment_date = datetime.now().isoformat()
    
    return versions


def create_ab_test_scenario(session) -> Dict[str, Any]:
    """Create a complete A/B testing scenario with control and treatment models."""
    # Create control model (established)
    control_registry = ModelRegistryFactory(session=session, model_name="player_model_mid")
    control_version = ProductionModelVersionFactory(session=session, registry=control_registry, version="2.1.0")
    
    # Create treatment model (experimental)
    treatment_version = ModelVersionFactory(session=session, registry=control_registry, version="2.2.0-beta")
    
    # Create experiment
    experiment = ModelExperimentFactory(
        session=session,
        model_version=treatment_version,
        control_version_id=control_version.id,
        status="running"
    )
    
    # Create performance data for both versions
    for version in [control_version, treatment_version]:
        ModelPerformanceFactory(session=session, version=version, dataset_type="production")
    
    return {
        "control_version": control_version,
        "treatment_version": treatment_version,
        "experiment": experiment,
    }


def create_model_development_pipeline(session) -> Dict[str, List[ModelVersion]]:
    """Create models representing different stages of development pipeline."""
    registry = ModelRegistryFactory(session=session, model_name="transfer_optimizer")
    
    # Development versions
    dev_versions = [
        ModelVersionFactory(session=session, registry=registry, status="training", version="3.0.0-dev"),
        ModelVersionFactory(session=session, registry=registry, status="ready", version="3.0.0-alpha"),
    ]
    
    # Staging versions
    staging_versions = [
        BetaModelVersionFactory(session=session, registry=registry, version="3.0.0-beta"),
        ModelVersionFactory(session=session, registry=registry, status="ready", version="3.0.0-rc1"),
    ]
    
    # Production versions
    production_versions = [
        ProductionModelVersionFactory(session=session, registry=registry, version="2.5.0"),
        ProductionModelVersionFactory(session=session, registry=registry, version="2.4.1", status="deprecated"),
    ]
    
    return {
        "development": dev_versions,
        "staging": staging_versions,
        "production": production_versions,
    }