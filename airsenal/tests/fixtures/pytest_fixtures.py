"""
Pytest fixtures for common testing scenarios.

Provides comprehensive pytest fixtures that combine factories and mock services
for easy testing of AIrsenal components.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.schema import Base

from .database_factories import (
    MigrationHistoryFactory,
    create_compatibility_matrix,
    create_migration_timeline,
)
from .feature_factories import (
    ComputedFeatureFactory,
    FeatureCacheFactory,
    FeatureDefinitionFactory,
    create_feature_computation_pipeline,
    create_player_feature_timeline,
)
from .mock_services import (
    create_mock_fpl_fetcher,
    create_mock_log_scenario,
    create_mock_player_models,
    create_mock_redis_cache,
)
from .model_factories import (
    ModelRegistryFactory,
    ModelVersionFactory,
    ProductionModelVersionFactory,
    create_ab_test_scenario,
    create_model_development_pipeline,
    create_model_family,
)
from .player_factories import (
    DefenderAttributesFactory,
    ForwardAttributesFactory,
    GoalkeeperAttributesFactory,
    MidfielderAttributesFactory,
    PlayerAttributesExtendedFactory,
    create_form_comparison_players,
    create_gameweek_players,
    create_squad_players,
)


# Database and session fixtures
@pytest.fixture
def test_db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_session(test_db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(bind=test_db_engine)
    session = SessionLocal()

    # Configure factory_boy to use this session
    from . import (
        database_factories,
        feature_factories,
        model_factories,
        player_factories,
    )

    for factory_module in [
        player_factories,
        model_factories,
        feature_factories,
        database_factories,
    ]:
        for name in dir(factory_module):
            obj = getattr(factory_module, name)
            if hasattr(obj, "_meta") and hasattr(obj._meta, "sqlalchemy_session"):
                obj._meta.sqlalchemy_session = session

    yield session
    session.close()


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for file storage during tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Player and team fixtures
@pytest.fixture
def sample_players(test_session):
    """Create a set of sample players with extended attributes."""
    players = []
    for position in ["GK", "DEF", "MID", "FWD"]:
        for _i in range(3):  # 3 players per position
            factory_class = {
                "GK": GoalkeeperAttributesFactory,
                "DEF": DefenderAttributesFactory,
                "MID": MidfielderAttributesFactory,
                "FWD": ForwardAttributesFactory,
            }[position]

            attrs = factory_class()
            players.append(attrs.player)

    return players


@pytest.fixture
def fpl_squad(test_session):
    """Create a complete 15-player FPL squad."""
    return create_squad_players(test_session, num_players=15)


@pytest.fixture
def gameweek_players(test_session):
    """Create players for a specific gameweek scenario."""
    return create_gameweek_players(
        test_session, season="2324", gameweek=10, num_players=30
    )


@pytest.fixture
def form_analysis_players(test_session):
    """Create players with different form patterns for analysis testing."""
    return create_form_comparison_players(test_session)


# Model versioning fixtures
@pytest.fixture
def model_registry(test_session):
    """Create a sample model registry."""
    return ModelRegistryFactory(model_name="player_model_mid")


@pytest.fixture
def model_versions(test_session, model_registry):
    """Create multiple versions of a model."""
    versions = []
    for i, version in enumerate(["1.0.0", "1.1.0", "1.2.0", "2.0.0"]):
        is_production = i == 2  # Make version 1.2.0 production

        if is_production:
            mv = ProductionModelVersionFactory(registry=model_registry, version=version)
        else:
            mv = ModelVersionFactory(registry=model_registry, version=version)

        versions.append(mv)

    return versions


@pytest.fixture
def model_family(test_session):
    """Create a complete model family with versions, artifacts, and performance data."""
    return create_model_family(
        test_session, model_name="transfer_optimizer", num_versions=5
    )


@pytest.fixture
def ab_test_scenario(test_session):
    """Create an A/B testing scenario with control and treatment models."""
    return create_ab_test_scenario(test_session)


@pytest.fixture
def model_pipeline(test_session):
    """Create models representing different pipeline stages."""
    return create_model_development_pipeline(test_session)


# Feature store fixtures
@pytest.fixture
def feature_definitions(test_session):
    """Create sample feature definitions."""
    features = []
    for feature_name in [
        "rolling_goals_5",
        "xg_form_3",
        "team_attack_strength",
        "fixture_difficulty",
    ]:
        feature = FeatureDefinitionFactory(name=feature_name)
        features.append(feature)
    return features


@pytest.fixture
def computed_features(test_session, feature_definitions):
    """Create computed features based on definitions."""
    computed = []
    for definition in feature_definitions:
        for entity_id in range(1, 11):  # 10 entities per feature
            feature = ComputedFeatureFactory(
                feature_definition=definition, entity_id=entity_id
            )
            computed.append(feature)
    return computed


@pytest.fixture
def feature_cache_entries(test_session):
    """Create feature cache entries for performance testing."""
    cache_entries = []
    for feature_name in ["rolling_goals_5", "xg_form_3", "team_recent_form_5"]:
        for entity_id in range(1, 21):  # 20 entities per feature
            cache = FeatureCacheFactory(feature_name=feature_name, entity_id=entity_id)
            cache_entries.append(cache)
    return cache_entries


@pytest.fixture
def feature_pipeline(test_session):
    """Create a complete feature computation pipeline."""
    return create_feature_computation_pipeline(test_session)


@pytest.fixture
def player_timeline(test_session):
    """Create a timeline of features for a single player."""
    return create_player_feature_timeline(test_session, player_id=123, season="2324")


# Database versioning fixtures
@pytest.fixture
def database_versions(test_session):
    """Create a series of database versions."""
    return create_migration_timeline(
        test_session, start_version="1.0.0", num_versions=5
    )


@pytest.fixture
def migration_history(test_session, database_versions):
    """Create migration history for database versions."""
    history = []
    for version in database_versions:
        for i in range(2):  # 2 migrations per version
            migration = MigrationHistoryFactory(
                database_version=version, sequence_number=i + 1
            )
            history.append(migration)
    return history


@pytest.fixture
def compatibility_matrix(test_session):
    """Create a schema compatibility matrix."""
    return create_compatibility_matrix(test_session)


# Mock service fixtures
@pytest.fixture
def mock_fpl_fetcher():
    """Create a mock FPL data fetcher."""
    return create_mock_fpl_fetcher(season="2324", current_gameweek=15)


@pytest.fixture
def mock_redis_cache():
    """Create a mock Redis cache."""
    return create_mock_redis_cache()


@pytest.fixture
def mock_redis_with_data():
    """Create a mock Redis cache preloaded with test data."""
    preload_data = {
        "player:123:form": "7.5",
        "player:456:xg": "0.8",
        "team:1:strength": "1.2",
        "fixture:789:difficulty": "3.5",
    }
    return create_mock_redis_cache(preload_data)


@pytest.fixture
def mock_player_models():
    """Create mock player models for all positions."""
    return create_mock_player_models(["GK", "DEF", "MID", "FWD"])


@pytest.fixture
def mock_logging_scenario():
    """Create a mock logging scenario with sample logs."""
    return create_mock_log_scenario()


# Integration fixtures for complex scenarios
@pytest.fixture
def fpl_season_scenario(test_session, mock_fpl_fetcher):
    """Create a complete FPL season scenario with players, fixtures, and data."""
    # Create players for the season
    players = create_gameweek_players(
        test_session, season="2324", gameweek=1, num_players=100
    )

    # Create feature timeline for top players
    feature_timelines = {}
    for _i, player_attrs in enumerate(players[:10]):  # Top 10 players
        timeline = create_player_feature_timeline(
            test_session, player_id=player_attrs.player_id, season="2324"
        )
        feature_timelines[player_attrs.player_id] = timeline

    return {
        "players": players,
        "fetcher": mock_fpl_fetcher,
        "feature_timelines": feature_timelines,
        "season": "2324",
        "current_gameweek": mock_fpl_fetcher.current_gameweek,
    }


@pytest.fixture
def model_deployment_scenario(test_session, temp_storage_dir):
    """Create a model deployment scenario with versioning and A/B testing."""
    # Create model development pipeline
    pipeline = create_model_development_pipeline(test_session)

    # Create A/B test
    ab_test = create_ab_test_scenario(test_session)

    # Create model storage directory structure
    model_dir = temp_storage_dir / "models"
    model_dir.mkdir()

    return {
        "pipeline": pipeline,
        "ab_test": ab_test,
        "storage_dir": model_dir,
        "production_model": pipeline["production"][0],
        "experiment": ab_test["experiment"],
    }


@pytest.fixture
def feature_engineering_scenario(test_session, mock_redis_cache):
    """Create a feature engineering scenario with computation and caching."""
    # Create feature computation pipeline
    pipeline = create_feature_computation_pipeline(test_session)

    # Preload cache with some computed features
    for _i, computed in enumerate(pipeline["computed_features"][:5]):
        cache_key = f"feature:{computed.feature_definition.name}:{computed.entity_id}:{computed.gameweek or 'latest'}"
        mock_redis_cache.set(cache_key, str(computed.value), ex=3600)

    return {
        "feature_pipeline": pipeline,
        "cache": mock_redis_cache,
        "definitions": pipeline["definitions"],
        "computed_features": pipeline["computed_features"],
        "cache_entries": pipeline["cache_entries"],
    }


@pytest.fixture
def transfer_optimization_scenario(
    test_session, mock_fpl_fetcher, form_analysis_players
):
    """Create a transfer optimization scenario."""
    # Create current squad
    current_squad = create_squad_players(test_session, num_players=15)

    # Create transfer targets (high form players)
    transfer_targets = form_analysis_players["high_form"]

    # Create players to transfer out (low form players)
    transfer_candidates = form_analysis_players["low_form"]

    return {
        "current_squad": current_squad,
        "transfer_targets": transfer_targets,
        "transfer_candidates": transfer_candidates,
        "fetcher": mock_fpl_fetcher,
        "budget": 1000,  # 100.0 million budget
        "free_transfers": 1,
    }


@pytest.fixture
def performance_testing_scenario(test_session, mock_redis_cache):
    """Create a scenario for performance testing with large datasets."""
    # Create large number of players
    players = []
    for position in ["GK", "DEF", "MID", "FWD"]:
        for i in range(25):  # 25 players per position = 100 total
            factory_class = {
                "GK": GoalkeeperAttributesFactory,
                "DEF": DefenderAttributesFactory,
                "MID": MidfielderAttributesFactory,
                "FWD": ForwardAttributesFactory,
            }[position]

            attrs = factory_class()
            players.append(attrs)

    # Create feature cache entries for performance testing
    for i in range(1000):  # 1000 cache entries
        FeatureCacheFactory(
            entity_id=i % 100 + 1,  # Distribute across 100 entities
            hit_count=i % 50,  # Vary hit counts
        )

    return {
        "players": players,
        "cache": mock_redis_cache,
        "num_players": len(players),
        "positions": ["GK", "DEF", "MID", "FWD"],
    }


# Parameterized fixtures for testing multiple scenarios
@pytest.fixture(params=["GK", "DEF", "MID", "FWD"])
def position_specific_player(request, test_session):
    """Create players for each position (parameterized)."""
    position = request.param
    factory_class = {
        "GK": GoalkeeperAttributesFactory,
        "DEF": DefenderAttributesFactory,
        "MID": MidfielderAttributesFactory,
        "FWD": ForwardAttributesFactory,
    }[position]

    return factory_class()


@pytest.fixture(params=["high_form", "low_form", "penalty_takers", "easy_fixtures"])
def form_scenario_players(request, test_session):
    """Create players for different form scenarios (parameterized)."""
    scenario = request.param
    players = create_form_comparison_players(test_session)
    return players[scenario]


@pytest.fixture(params=[1, 5, 10, 20, 38])
def gameweek_scenario(request, test_session):
    """Create scenarios for different gameweeks (parameterized)."""
    gameweek = request.param
    return create_gameweek_players(
        test_session, season="2324", gameweek=gameweek, num_players=20
    )


# Configuration fixtures
@pytest.fixture
def test_config():
    """Provide test configuration settings."""
    return {
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
        "redis": {
            "mock": True,
            "default_ttl": 3600,
        },
        "fpl": {
            "mock": True,
            "season": "2324",
            "team_id": 742663,
        },
        "models": {
            "storage_dir": "/tmp/test_models",
            "enable_caching": True,
        },
        "features": {
            "cache_ttl": 1800,
            "batch_size": 100,
        },
        "logging": {
            "level": "DEBUG",
            "capture": True,
        },
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    return
    # Cleanup code runs after each test
    # This is where you could add any global cleanup logic


# Benchmark fixtures for performance testing
@pytest.fixture
def benchmark_data(test_session):
    """Create standardized data for benchmarking."""
    # Create consistent data for performance comparisons
    players = []
    for i in range(100):  # Fixed number for consistent benchmarks
        attrs = PlayerAttributesExtendedFactory(
            player__player_id=i + 1,
            season="2324",
            gameweek=15,
        )
        players.append(attrs)

    return {
        "players": players,
        "count": len(players),
        "gameweek": 15,
        "season": "2324",
    }


# Scope management fixtures
@pytest.fixture(scope="session")
def session_config():
    """Session-wide configuration that persists across all tests."""
    return {
        "test_start_time": datetime.now(),
        "test_session_id": "test_session_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "global_settings": {
            "strict_mode": True,
            "performance_monitoring": True,
        },
    }


@pytest.fixture(scope="module")
def module_test_data(test_session):
    """Module-scope test data that's shared within a test module."""
    # This data persists for all tests in a module
    # Useful for expensive setup that can be shared
    return {
        "reference_players": create_squad_players(test_session, num_players=15),
        "reference_season": "2324",
        "module_start_time": datetime.now(),
    }


# Error simulation fixtures
@pytest.fixture
def error_scenarios():
    """Provide various error scenarios for testing error handling."""
    return {
        "network_error": Mock(side_effect=ConnectionError("Network unavailable")),
        "timeout_error": Mock(side_effect=TimeoutError("Request timed out")),
        "auth_error": Mock(side_effect=PermissionError("Authentication failed")),
        "data_error": Mock(side_effect=ValueError("Invalid data format")),
        "resource_error": Mock(side_effect=RuntimeError("Resource exhausted")),
    }


# Patch fixtures for mocking external dependencies
@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    with patch("airsenal.framework.data_fetcher.requests.Session") as mock_session:
        mock_session.return_value.get.return_value.json.return_value = {"mock": "data"}
        mock_session.return_value.get.return_value.status_code = 200
        yield mock_session


@pytest.fixture
def mock_file_operations(temp_storage_dir):
    """Mock file operations to use temporary directory."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = temp_storage_dir
        yield temp_storage_dir


# Validation fixtures
@pytest.fixture
def data_validators():
    """Provide data validation functions for testing."""

    def validate_player_attributes(attrs):
        """Validate player attributes have required fields."""
        required_fields = ["player_id", "season", "gameweek", "position", "team"]
        for field in required_fields:
            assert hasattr(attrs, field), f"Missing required field: {field}"
            assert getattr(attrs, field) is not None, f"Field {field} cannot be None"

    def validate_fpl_squad(squad):
        """Validate FPL squad composition."""
        positions = [player.position("2324") for player in squad]
        position_counts = {
            pos: positions.count(pos) for pos in ["GK", "DEF", "MID", "FWD"]
        }

        assert position_counts["GK"] == 2, "Squad must have 2 goalkeepers"
        assert position_counts["DEF"] == 5, "Squad must have 5 defenders"
        assert position_counts["MID"] == 5, "Squad must have 5 midfielders"
        assert position_counts["FWD"] == 3, "Squad must have 3 forwards"

    def validate_model_version(version):
        """Validate model version has required metadata."""
        assert version.version is not None, "Version must have version number"
        assert version.training_date is not None, "Version must have training date"
        assert version.status in [
            "ready",
            "training",
            "deployed",
            "deprecated",
            "failed",
        ]

    return {
        "player_attributes": validate_player_attributes,
        "fpl_squad": validate_fpl_squad,
        "model_version": validate_model_version,
    }
