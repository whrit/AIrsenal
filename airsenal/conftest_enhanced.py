"""
Enhanced conftest.py with comprehensive test fixtures for AIrsenal.

This file extends the original conftest.py with comprehensive fixtures for
testing all Sprint 00 enhanced components while maintaining backward compatibility.
"""

import os
import random
import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework import env

# Set up test environment
env.AIRSENAL_HOME = Path(mkdtemp())

from airsenal.framework.mappings import alternative_team_names  # noqa: E402
from airsenal.framework.schema import Base, Player, PlayerAttributes  # noqa: E402
from airsenal.framework.utils import CURRENT_SEASON  # noqa: E402
from airsenal.tests.test_resources import dummy_players  # noqa: E402

# Import comprehensive fixtures
from airsenal.tests.fixtures.pytest_fixtures import *  # noqa: E402, F403
from airsenal.tests.fixtures.test_database import (  # noqa: E402
    TestDatabaseManager, TestDatabaseContext, create_test_database,
    quick_test_database, benchmark_database, cleanup_test_databases
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Original test constants
API_SESSION_ID = "TESTSESSION"
TEST_PAST_SEASON = "2021"

# Create test engines (maintain backward compatibility)
testengine_dummy = create_engine(f"sqlite:///{env.AIRSENAL_HOME}/test.db")
testengine_past = create_engine(
    f"sqlite:///{os.path.dirname(__file__)}/tests/testdata/testdata_1718_1819.db"
)

Base.metadata.create_all(testengine_dummy)
Base.metadata.bind = testengine_dummy


# Original context managers (maintain backward compatibility)
@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_dummy)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except Exception:
        testsession.rollback()
        raise
    finally:
        testsession.close()


@contextmanager
def past_data_session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=testengine_past)
    testsession = db_session()
    try:
        yield testsession
        testsession.commit()
    except Exception:
        testsession.rollback()
        raise
    finally:
        testsession.close()


def value_generator(index, position):
    """
    make up a price for a dummy player, based on index and position
    """
    if position == "GK":
        value = 40 + index * random.randint(0, 5)
    elif position == "DEF":
        value = 40 + index * random.randint(5, 10)
    elif position == "MID":
        value = 50 + index * random.randint(10, 20)
    elif position == "FWD":
        value = 60 + index * random.randint(15, 20)
    return value


@pytest.fixture(scope="session")
def fill_players():
    """
    fill a bunch of dummy players (original implementation for backward compatibility)
    """
    team_list = list(alternative_team_names.keys())
    season = CURRENT_SEASON
    gameweek = 1
    with session_scope() as ts:
        if len(ts.query(Player).all()) > 0:
            return
        for i, n in enumerate(dummy_players):
            p = Player()
            p.player_id = i
            p.fpl_api_id = i
            p.name = n
            logger.debug(f"Filling {i} {n}")
            try:
                ts.add(p)
            except Exception:
                logger.error(f"Error adding {i} {n}")
            # now fill player_attributes
            if i % 15 < 2:
                pos = "GK"
            elif i % 15 < 7:
                pos = "DEF"
            elif i % 15 < 12:
                pos = "MID"
            else:
                pos = "FWD"
            team = team_list[i % 20]
            # make the first 15 players affordable,
            # the next 15 almost affordable,
            # the next 15 mostly unaffordable,
            # and rest very expensive
            price = value_generator(i // 15, pos)
            pa = PlayerAttributes()
            pa.season = season
            pa.team = team
            pa.gameweek = gameweek
            pa.price = price
            pa.position = pos
            player = ts.query(Player).filter_by(player_id=i).first()
            pa.player = player
            ts.add(pa)
        ts.commit()


# Enhanced fixtures for comprehensive testing
@pytest.fixture(scope="session")
def test_database_manager():
    """
    Session-wide test database manager for comprehensive testing.
    
    Creates a test database with standard preset data that persists
    across all tests in the session.
    """
    db_manager = create_test_database(preset="standard")
    yield db_manager
    db_manager.destroy_database()


@pytest.fixture(scope="function")
def isolated_test_db():
    """
    Function-scope isolated test database.
    
    Creates a fresh test database for each test function,
    ensuring complete isolation between tests.
    """
    with TestDatabaseContext(preset="minimal") as db_manager:
        yield db_manager


@pytest.fixture(scope="function") 
def performance_test_db():
    """
    Performance-optimized test database for benchmarking tests.
    """
    db_manager = benchmark_database(size="medium")
    yield db_manager
    db_manager.destroy_database()


@pytest.fixture(scope="module")
def module_test_db():
    """
    Module-scope test database that persists for all tests in a module.
    
    Useful for expensive setup that can be shared within a test module.
    """
    db_manager = create_test_database(preset="standard")
    yield db_manager
    db_manager.destroy_database()


# Enhanced session management with comprehensive data
@pytest.fixture(scope="function")
def enhanced_session(test_session):
    """
    Enhanced database session with comprehensive test data.
    
    Builds on the basic test_session fixture to provide access to
    comprehensive test data including extended player attributes,
    model versioning, feature store, and database versioning data.
    """
    # Configure factory_boy to use this session
    from airsenal.tests.fixtures import (
        player_factories, model_factories, feature_factories, database_factories
    )
    
    for factory_module in [player_factories, model_factories, feature_factories, database_factories]:
        for name in dir(factory_module):
            obj = getattr(factory_module, name)
            if hasattr(obj, '_meta') and hasattr(obj._meta, 'sqlalchemy_session'):
                obj._meta.sqlalchemy_session = test_session
    
    yield test_session


# Test data scenarios for specific testing needs
@pytest.fixture
def basic_test_scenario(enhanced_session):
    """
    Basic test scenario with minimal realistic data.
    
    Includes:
    - 20 players with extended attributes
    - 1 model family with 3 versions
    - Basic feature definitions
    """
    from airsenal.tests.fixtures.player_factories import create_gameweek_players
    from airsenal.tests.fixtures.model_factories import create_model_family
    from airsenal.tests.fixtures.feature_factories import FeatureDefinitionFactory
    
    # Create players
    players = create_gameweek_players(enhanced_session, season="2324", gameweek=10, num_players=20)
    
    # Create model family
    model_family = create_model_family(enhanced_session, model_name="test_model", num_versions=3)
    
    # Create feature definitions
    features = [
        FeatureDefinitionFactory(name="rolling_goals_5"),
        FeatureDefinitionFactory(name="xg_form_3"),
        FeatureDefinitionFactory(name="team_strength"),
    ]
    
    enhanced_session.commit()
    
    return {
        "players": players,
        "model_family": model_family,
        "features": features,
        "session": enhanced_session,
    }


@pytest.fixture
def transfer_optimization_scenario(enhanced_session, mock_fpl_fetcher):
    """
    Complete transfer optimization test scenario.
    
    Includes current squad, transfer targets, form analysis players,
    and mock FPL data fetcher.
    """
    from airsenal.tests.fixtures.player_factories import (
        create_squad_players, create_form_comparison_players
    )
    
    # Create current squad
    current_squad = create_squad_players(enhanced_session, num_players=15)
    
    # Create form analysis players
    form_players = create_form_comparison_players(enhanced_session)
    
    enhanced_session.commit()
    
    return {
        "current_squad": current_squad,
        "form_players": form_players,
        "transfer_targets": form_players["high_form"],
        "transfer_candidates": form_players["low_form"],
        "penalty_takers": form_players["penalty_takers"],
        "fetcher": mock_fpl_fetcher,
        "session": enhanced_session,
    }


@pytest.fixture
def model_testing_scenario(enhanced_session, temp_storage_dir):
    """
    Model testing scenario with versioning and A/B testing.
    
    Includes model families, A/B tests, development pipeline,
    and temporary storage for model artifacts.
    """
    from airsenal.tests.fixtures.model_factories import (
        create_model_development_pipeline, create_ab_test_scenario
    )
    
    # Create development pipeline
    pipeline = create_model_development_pipeline(enhanced_session)
    
    # Create A/B test scenario
    ab_test = create_ab_test_scenario(enhanced_session)
    
    enhanced_session.commit()
    
    return {
        "pipeline": pipeline,
        "ab_test": ab_test,
        "storage_dir": temp_storage_dir,
        "production_model": pipeline["production"][0],
        "experiment": ab_test["experiment"],
        "session": enhanced_session,
    }


@pytest.fixture
def feature_engineering_scenario(enhanced_session, mock_redis_cache):
    """
    Feature engineering scenario with computation and caching.
    
    Includes feature definitions, computed features, cache entries,
    and mock Redis cache.
    """
    from airsenal.tests.fixtures.feature_factories import create_feature_computation_pipeline
    
    # Create feature computation pipeline
    pipeline = create_feature_computation_pipeline(enhanced_session)
    
    # Preload cache with computed features
    for i, computed in enumerate(pipeline["computed_features"][:10]):
        cache_key = f"feature:{computed.feature_definition.name}:{computed.entity_id}:gw{computed.gameweek or 'latest'}"
        mock_redis_cache.set(cache_key, str(computed.value), ex=3600)
    
    enhanced_session.commit()
    
    return {
        "pipeline": pipeline,
        "cache": mock_redis_cache,
        "definitions": pipeline["definitions"],
        "computed_features": pipeline["computed_features"],
        "cache_entries": pipeline["cache_entries"],
        "session": enhanced_session,
    }


# Parameterized fixtures for testing multiple scenarios
@pytest.fixture(params=["minimal", "standard", "comprehensive"])
def database_preset(request):
    """Parameterized fixture for testing different database presets."""
    preset = request.param
    with TestDatabaseContext(preset=preset) as db_manager:
        yield db_manager


@pytest.fixture(params=[1, 5, 10, 20, 38])
def multi_gameweek_scenario(request, enhanced_session):
    """Parameterized fixture for testing different gameweek scenarios."""
    gameweek = request.param
    from airsenal.tests.fixtures.player_factories import create_gameweek_players
    
    players = create_gameweek_players(
        enhanced_session, season="2324", gameweek=gameweek, num_players=30
    )
    enhanced_session.commit()
    
    return {
        "gameweek": gameweek,
        "players": players,
        "session": enhanced_session,
    }


# Performance and stress testing fixtures
@pytest.fixture
def stress_test_scenario(enhanced_session):
    """
    High-volume data scenario for stress testing.
    
    Creates large datasets to test performance and scalability.
    """
    from airsenal.tests.fixtures.player_factories import PlayerAttributesExtendedFactory
    from airsenal.tests.fixtures.feature_factories import FeatureCacheFactory
    
    # Create many players
    players = []
    for i in range(100):  # 100 players
        attrs = PlayerAttributesExtendedFactory()
        players.append(attrs)
    
    # Create many cache entries
    cache_entries = []
    for i in range(500):  # 500 cache entries
        cache = FeatureCacheFactory(entity_id=i % 100 + 1)
        cache_entries.append(cache)
    
    enhanced_session.commit()
    
    return {
        "players": players,
        "cache_entries": cache_entries,
        "player_count": len(players),
        "cache_count": len(cache_entries),
        "session": enhanced_session,
    }


# Configuration and environment fixtures
@pytest.fixture
def test_environment_config():
    """
    Test environment configuration with all necessary settings.
    """
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
            "current_gameweek": 15,
        },
        "models": {
            "storage_dir": str(env.AIRSENAL_HOME / "test_models"),
            "enable_versioning": True,
            "enable_caching": True,
        },
        "features": {
            "cache_ttl": 1800,
            "batch_size": 100,
            "enable_timeseries": True,
        },
        "logging": {
            "level": "INFO",
            "capture": True,
            "structured": True,
        },
        "testing": {
            "strict_mode": True,
            "performance_monitoring": True,
            "data_validation": True,
        }
    }


# Cleanup fixtures
@pytest.fixture(autouse=True, scope="session")
def session_cleanup():
    """
    Automatic session-wide cleanup.
    
    Ensures test resources are properly cleaned up at the end of the test session.
    """
    yield
    # Cleanup after all tests complete
    cleanup_test_databases()
    
    # Clean up temporary directories
    import shutil
    try:
        shutil.rmtree(env.AIRSENAL_HOME, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Could not clean up AIRSENAL_HOME: {e}")


@pytest.fixture(autouse=True, scope="function")
def function_cleanup():
    """
    Automatic function-wide cleanup.
    
    Ensures each test function starts with a clean state.
    """
    # Pre-test setup
    logger.debug("Starting test function")
    
    yield
    
    # Post-test cleanup
    logger.debug("Completing test function")


# Utility fixtures for common test operations
@pytest.fixture
def data_validators():
    """
    Data validation functions for testing.
    
    Provides validation functions to ensure test data meets expected criteria.
    """
    def validate_player_attributes(attrs):
        """Validate player attributes have required fields and realistic values."""
        required_fields = ["player_id", "season", "gameweek", "position", "team", "price"]
        for field in required_fields:
            assert hasattr(attrs, field), f"Missing required field: {field}"
            assert getattr(attrs, field) is not None, f"Field {field} cannot be None"
        
        # Validate extended fields if present
        if hasattr(attrs, "xg_per_90") and attrs.xg_per_90 is not None:
            assert 0 <= attrs.xg_per_90 <= 2.0, "xG per 90 should be between 0 and 2.0"
        
        if hasattr(attrs, "form_3_games") and attrs.form_3_games is not None:
            assert 0 <= attrs.form_3_games <= 20.0, "Form should be between 0 and 20.0"
        
        if hasattr(attrs, "price") and attrs.price is not None:
            assert 30 <= attrs.price <= 200, "Price should be between 3.0 and 20.0 million"
    
    def validate_fpl_squad(squad):
        """Validate FPL squad composition."""
        positions = [player.position("2324") for player in squad if player.position("2324")]
        position_counts = {pos: positions.count(pos) for pos in ["GK", "DEF", "MID", "FWD"]}
        
        assert position_counts.get("GK", 0) == 2, "Squad must have 2 goalkeepers"
        assert position_counts.get("DEF", 0) == 5, "Squad must have 5 defenders"
        assert position_counts.get("MID", 0) == 5, "Squad must have 5 midfielders"
        assert position_counts.get("FWD", 0) == 3, "Squad must have 3 forwards"
    
    def validate_model_version(version):
        """Validate model version has required metadata."""
        assert version.version is not None, "Version must have version number"
        assert version.training_date is not None, "Version must have training date"
        assert version.status in ["ready", "training", "deployed", "deprecated", "failed"]
        
        if version.is_production:
            assert version.status in ["deployed", "ready"], "Production models should be deployed or ready"
    
    def validate_feature_definition(definition):
        """Validate feature definition."""
        assert definition.name is not None, "Feature must have a name"
        assert definition.feature_type in ["player", "team", "fixture", "external"]
        assert definition.data_type in ["float", "int", "boolean", "string"]
    
    return {
        "player_attributes": validate_player_attributes,
        "fpl_squad": validate_fpl_squad,
        "model_version": validate_model_version,
        "feature_definition": validate_feature_definition,
    }


# Legacy compatibility fixtures
@pytest.fixture
def legacy_session():
    """
    Legacy session fixture for backward compatibility.
    
    Provides the same interface as the original session_scope context manager
    but as a pytest fixture.
    """
    with session_scope() as session:
        yield session


@pytest.fixture
def legacy_past_session():
    """
    Legacy past data session fixture for backward compatibility.
    """
    with past_data_session_scope() as session:
        yield session


# Mark fixtures for easy discovery
pytest_plugins = [
    "airsenal.tests.fixtures.pytest_fixtures",
]

# Export key functions for direct use
__all__ = [
    # Original functions
    "session_scope",
    "past_data_session_scope", 
    "value_generator",
    
    # Test database management
    "TestDatabaseManager",
    "TestDatabaseContext",
    "create_test_database",
    "quick_test_database",
    "benchmark_database",
    "cleanup_test_databases",
    
    # Constants
    "API_SESSION_ID",
    "TEST_PAST_SEASON",
    "testengine_dummy",
    "testengine_past",
]