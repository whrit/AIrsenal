"""
Validation tests for the comprehensive test fixtures.

This module contains tests that verify all the new fixtures work correctly
and can be used in the test suite without issues.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from airsenal.framework.schema import Base


class TestFixtureValidation:
    """Test that all fixtures work correctly."""

    def test_player_factories(self):
        """Test that player factories work correctly."""
        from .player_factories import (
            DefenderAttributesFactory,
            ForwardAttributesFactory,
            GoalkeeperAttributesFactory,
            HighFormPlayerAttributesFactory,
            MidfielderAttributesFactory,
            PenaltyTakerPlayerAttributesFactory,
            PlayerAttributesExtendedFactory,
            PlayerFactory,
        )

        # Test basic factories
        player = PlayerFactory.build()
        assert player.name is not None
        assert player.fpl_api_id is not None

        # Test extended attributes factory
        attrs = PlayerAttributesExtendedFactory.build()
        assert attrs.xg_per_90 is not None
        assert attrs.form_3_games is not None
        assert attrs.position in ["GK", "DEF", "MID", "FWD"]
        assert attrs.is_penalty_taker in [True, False]

        # Test position-specific factories
        gk_attrs = GoalkeeperAttributesFactory.build()
        assert gk_attrs.position == "GK"
        assert gk_attrs.xg_per_90 < 0.1  # Goalkeepers have low xG

        def_attrs = DefenderAttributesFactory.build()
        assert def_attrs.position == "DEF"
        assert def_attrs.tackles_per_90 > 1.0  # Defenders tackle more

        mid_attrs = MidfielderAttributesFactory.build()
        assert mid_attrs.position == "MID"

        fwd_attrs = ForwardAttributesFactory.build()
        assert fwd_attrs.position == "FWD"
        assert fwd_attrs.xg_per_90 > 0.2  # Forwards have higher xG

        # Test scenario-specific factories
        high_form = HighFormPlayerAttributesFactory.build()
        assert high_form.form_3_games >= 8.0

        penalty_taker = PenaltyTakerPlayerAttributesFactory.build()
        assert penalty_taker.is_penalty_taker is True
        assert penalty_taker.role_confidence >= 0.7

    def test_model_factories(self):
        """Test that model versioning factories work correctly."""
        from .model_factories import (
            ModelArtifactFactory,
            ModelExperimentFactory,
            ModelPerformanceFactory,
            ModelRegistryFactory,
            ModelVersionFactory,
            ProductionModelVersionFactory,
        )

        # Test model registry
        registry = ModelRegistryFactory.build()
        assert registry.model_name is not None
        assert registry.model_type is not None
        assert registry.is_active is True

        # Test model version
        version = ModelVersionFactory.build()
        assert version.version is not None
        assert version.status in [
            "ready",
            "training",
            "deployed",
            "deprecated",
            "failed",
        ]
        assert version.validation_mae is not None

        # Test production model
        prod_version = ProductionModelVersionFactory.build()
        assert prod_version.is_production is True
        assert prod_version.status in ["deployed", "ready"]

        # Test model artifact
        artifact = ModelArtifactFactory.build()
        assert artifact.artifact_type is not None
        assert artifact.serialization_format is not None
        assert artifact.checksum is not None

        # Test model performance
        performance = ModelPerformanceFactory.build()
        assert performance.dataset_type in [
            "validation",
            "test",
            "production",
            "backtest",
        ]
        assert performance.mae is not None

        # Test model experiment
        experiment = ModelExperimentFactory.build()
        assert experiment.experiment_name is not None
        assert experiment.status in ["planned", "running", "completed", "stopped"]
        assert 0.0 <= experiment.traffic_split <= 1.0

    def test_feature_factories(self):
        """Test that feature store factories work correctly."""
        from .feature_factories import (
            ComputedFeatureFactory,
            FeatureCacheFactory,
            FeatureDefinitionFactory,
            FeatureTimeSeriesFactory,
        )

        # Test feature definition
        definition = FeatureDefinitionFactory.build()
        assert definition.name is not None
        assert definition.feature_type in ["player", "team", "fixture", "external"]
        assert definition.data_type in ["float", "int", "boolean", "string"]
        assert definition.version is not None

        # Test computed feature
        computed = ComputedFeatureFactory.build()
        assert computed.entity_type is not None
        assert computed.entity_id is not None
        assert computed.season is not None
        assert computed.value is not None or computed.string_value is not None

        # Test feature cache
        cache = FeatureCacheFactory.build()
        assert cache.feature_name is not None
        assert cache.cache_key is not None
        assert cache.entity_id is not None
        assert cache.hit_count >= 0

        # Test feature time series
        timeseries = FeatureTimeSeriesFactory.build()
        assert timeseries.entity_type is not None
        assert timeseries.metric_name is not None
        assert timeseries.season is not None
        assert 1 <= timeseries.gameweek <= 38
        assert timeseries.value is not None

    def test_database_factories(self):
        """Test that database versioning factories work correctly."""
        from .database_factories import (
            DatabaseVersionFactory,
            MigrationHistoryFactory,
            SchemaCompatibilityFactory,
        )

        # Test database version
        db_version = DatabaseVersionFactory.build()
        assert db_version.version is not None
        assert db_version.schema_version is not None
        assert db_version.migration_type in ["initial", "upgrade", "rollback", "manual"]
        assert db_version.validation_status in ["validated", "pending", "failed"]

        # Test migration history
        migration = MigrationHistoryFactory.build()
        assert migration.migration_name is not None
        assert migration.migration_hash is not None
        assert migration.status in ["success", "failed", "partial", "rolled_back"]
        assert migration.execution_duration_seconds > 0

        # Test schema compatibility
        compatibility = SchemaCompatibilityFactory.build()
        assert compatibility.app_version_min is not None
        assert compatibility.app_version_max is not None
        assert compatibility.compatibility_level in [
            "full",
            "limited",
            "deprecated",
            "incompatible",
        ]
        assert compatibility.performance_impact in ["none", "low", "medium", "high"]

    def test_mock_services(self):
        """Test that mock services work correctly."""
        from .mock_services import (
            MockFPLDataFetcher,
            MockLogHandler,
            MockPlayerModel,
            MockRedisCache,
        )

        # Test mock FPL fetcher
        fetcher = MockFPLDataFetcher(season="2324")
        bootstrap = fetcher.get_bootstrap_data()

        assert "elements" in bootstrap
        assert "teams" in bootstrap
        assert "events" in bootstrap
        assert len(bootstrap["elements"]) > 0
        assert len(bootstrap["teams"]) == 20  # Premier League teams

        fixtures = fetcher.get_fixtures(gameweek=1)
        assert len(fixtures) == 10  # 10 fixtures per gameweek

        player_data = fetcher.get_player_data(player_id=1)
        assert "fixtures" in player_data
        assert "history" in player_data

        # Test mock Redis cache
        cache = MockRedisCache()

        # Basic operations
        cache.set("test_key", "test_value", ex=3600)
        assert cache.get("test_key") == b"test_value"
        assert cache.exists("test_key")
        assert cache.ttl("test_key") > 0

        # Bulk operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        keys = cache.keys("key*")
        assert len(keys) == 2

        # Test mock player model
        import numpy as np

        model = MockPlayerModel(position="MID")
        X = np.random.random((50, 10))
        y = np.random.uniform(0, 20, 50)

        model.fit(X, y)
        assert model.is_fitted

        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(0 <= p <= 20 for p in predictions)

        probabilities = model.predict_proba(X[:10])
        assert probabilities.shape == (10, 21)  # 0-20 points

        importance = model.get_feature_importance()
        assert len(importance) == 10  # 10 features
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to 1

        # Test mock log handler
        handler = MockLogHandler()

        # Create mock log record
        from unittest.mock import Mock

        record = Mock()
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 42

        handler.emit(record)
        logs = handler.get_logs()
        assert len(logs) == 1
        assert logs[0]["level"] == "INFO"
        assert logs[0]["message"] == "Test message"

    def test_realistic_fpl_data(self):
        """Test that realistic FPL data is working correctly."""
        from .fpl_data import (
            PREMIER_LEAGUE_TEAMS,
            REALISTIC_PLAYERS,
            STATISTICAL_DISTRIBUTIONS,
            generate_realistic_player_data,
            get_fixture_difficulty,
            get_random_team,
            get_realistic_player_name,
        )

        # Test team data
        assert len(PREMIER_LEAGUE_TEAMS) == 20
        assert "ARS" in PREMIER_LEAGUE_TEAMS
        assert PREMIER_LEAGUE_TEAMS["ARS"] == "Arsenal"

        # Test player data
        assert all(pos in REALISTIC_PLAYERS for pos in ["GK", "DEF", "MID", "FWD"])
        assert len(REALISTIC_PLAYERS["GK"]) > 20
        assert len(REALISTIC_PLAYERS["FWD"]) > 30

        # Test statistical distributions
        assert all(
            pos in STATISTICAL_DISTRIBUTIONS for pos in ["GK", "DEF", "MID", "FWD"]
        )
        mid_stats = STATISTICAL_DISTRIBUTIONS["MID"]
        assert "xg_per_90" in mid_stats
        assert "form_3_games" in mid_stats

        # Test utility functions
        player_name = get_realistic_player_name("MID")
        assert player_name in REALISTIC_PLAYERS["MID"]

        team_code, team_name = get_random_team()
        assert team_code in PREMIER_LEAGUE_TEAMS
        assert team_name == PREMIER_LEAGUE_TEAMS[team_code]

        player_data = generate_realistic_player_data(position="FWD")
        assert player_data["position"] == "FWD"
        assert player_data["name"] in REALISTIC_PLAYERS["FWD"]
        assert 0 <= player_data["xg_per_90"] <= 2.0
        assert 0 <= player_data["form_3_games"] <= 20.0

        # Test fixture difficulty
        difficulty = get_fixture_difficulty("ARS", "LIV")
        assert 1.0 <= difficulty <= 5.0

    def test_database_integration(self):
        """Test database integration with factories."""
        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        try:
            # Configure factories to use this session
            from . import model_factories, player_factories

            # Test with database persistence
            from .player_factories import PlayerAttributesExtendedFactory

            PlayerAttributesExtendedFactory._meta.sqlalchemy_session = session

            # Create and persist player
            player_attrs = PlayerAttributesExtendedFactory()
            session.commit()

            # Query back from database
            retrieved = session.query(player_factories.PlayerAttributes).first()
            assert retrieved is not None
            assert retrieved.player_id == player_attrs.player_id
            assert retrieved.xg_per_90 == player_attrs.xg_per_90

            # Test model versioning integration
            from .model_factories import ModelRegistryFactory, ModelVersionFactory

            ModelRegistryFactory._meta.sqlalchemy_session = session
            ModelVersionFactory._meta.sqlalchemy_session = session

            registry = ModelRegistryFactory()
            ModelVersionFactory(registry=registry)
            session.commit()

            # Query back model data
            retrieved_registry = session.query(model_factories.ModelRegistry).first()
            assert retrieved_registry is not None
            assert retrieved_registry.model_name == registry.model_name

            retrieved_version = session.query(model_factories.ModelVersion).first()
            assert retrieved_version is not None
            assert retrieved_version.registry_id == registry.id

        finally:
            session.close()

    def test_batch_creation_functions(self):
        """Test batch creation functions work correctly."""
        # Create in-memory database for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        try:
            from .feature_factories import create_feature_computation_pipeline
            from .model_factories import create_ab_test_scenario, create_model_family
            from .player_factories import (
                create_form_comparison_players,
                create_squad_players,
            )

            # Test squad creation
            squad = create_squad_players(session, num_players=15)
            assert len(squad) == 15

            positions = [
                player.position("2324") for player in squad if player.position("2324")
            ]
            position_counts = {
                pos: positions.count(pos) for pos in ["GK", "DEF", "MID", "FWD"]
            }
            assert position_counts.get("GK", 0) == 2
            assert position_counts.get("DEF", 0) == 5
            assert position_counts.get("MID", 0) == 5
            assert position_counts.get("FWD", 0) == 3

            # Test form comparison players
            form_players = create_form_comparison_players(session)
            assert "high_form" in form_players
            assert "low_form" in form_players
            assert "penalty_takers" in form_players
            assert len(form_players["high_form"]) > 0
            assert len(form_players["penalty_takers"]) > 0

            # Test model family creation
            model_family = create_model_family(
                session, model_name="test_model", num_versions=3
            )
            assert len(model_family) == 3
            assert all(v.registry.model_name == "test_model" for v in model_family)

            # Test A/B test scenario
            ab_test = create_ab_test_scenario(session)
            assert "control_version" in ab_test
            assert "treatment_version" in ab_test
            assert "experiment" in ab_test
            assert ab_test["experiment"].status == "running"

            # Test feature pipeline
            feature_pipeline = create_feature_computation_pipeline(session)
            assert "definitions" in feature_pipeline
            assert "computed_features" in feature_pipeline
            assert "cache_entries" in feature_pipeline
            assert len(feature_pipeline["definitions"]) > 0

        finally:
            session.close()

    def test_fixture_consistency(self):
        """Test that fixtures produce consistent, realistic data."""
        from .player_factories import PlayerAttributesExtendedFactory

        # Create multiple players and check consistency
        players = [PlayerAttributesExtendedFactory.build() for _ in range(10)]

        for player in players:
            # Check that all required fields are populated
            assert player.player_id is not None
            assert player.season is not None
            assert player.gameweek is not None
            assert player.position in ["GK", "DEF", "MID", "FWD"]
            assert player.team is not None
            assert player.price is not None

            # Check that extended fields have realistic values
            if player.xg_per_90 is not None:
                assert 0 <= player.xg_per_90 <= 2.0

            if player.form_3_games is not None:
                assert 0 <= player.form_3_games <= 20.0

            if player.momentum is not None:
                assert -1.0 <= player.momentum <= 1.0

            if player.next_3_fixture_difficulty is not None:
                assert 1.0 <= player.next_3_fixture_difficulty <= 5.0

            # Check role consistency
            if player.is_penalty_taker:
                # Penalty takers should have reasonable role confidence
                if player.role_confidence is not None:
                    assert player.role_confidence >= 0.4

    def test_performance_characteristics(self):
        """Test that fixtures have reasonable performance characteristics."""
        import time

        from .player_factories import PlayerAttributesExtendedFactory

        # Test single player creation performance
        start_time = time.time()
        PlayerAttributesExtendedFactory.build()
        single_creation_time = time.time() - start_time

        assert single_creation_time < 0.1  # Should create in less than 100ms

        # Test batch creation performance
        start_time = time.time()
        players = [PlayerAttributesExtendedFactory.build() for _ in range(100)]
        batch_creation_time = time.time() - start_time

        assert (
            batch_creation_time < 5.0
        )  # Should create 100 players in less than 5 seconds
        assert len(players) == 100

        # Test mock service performance
        from .mock_services import MockFPLDataFetcher

        start_time = time.time()
        fetcher = MockFPLDataFetcher()
        bootstrap_data = fetcher.get_bootstrap_data()
        api_simulation_time = time.time() - start_time

        assert (
            api_simulation_time < 1.0
        )  # Should simulate API call in less than 1 second
        assert len(bootstrap_data["elements"]) > 0

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        from .mock_services import MockRedisCache
        from .player_factories import PlayerAttributesExtendedFactory

        # Test with extreme values
        player = PlayerAttributesExtendedFactory.build(
            form_3_games=0.0,  # Minimum form
            xg_per_90=0.0,  # Minimum xG
            price=30,  # Minimum realistic price
        )

        assert player.form_3_games == 0.0
        assert player.xg_per_90 == 0.0
        assert player.price == 30

        # Test cache with many operations
        cache = MockRedisCache()

        # Add many keys
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        assert len(cache.keys("*")) == 1000

        # Test cache expiration
        cache.set("expire_test", "value", ex=1)
        assert cache.exists("expire_test")

        # Simulate time passing
        import time

        time.sleep(1.1)
        cache._cleanup_expired()
        assert not cache.exists("expire_test")


def run_validation():
    """Run all validation tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_validation()
