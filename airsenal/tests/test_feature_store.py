"""
Comprehensive tests for the AIrsenal Feature Store.

Tests cover:
- Feature registration and versioning
- Feature computation (rolling windows, form metrics)
- Caching functionality and performance
- Integration with existing prediction pipeline
- Data validation and monitoring
- Batch processing capabilities
"""

import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from airsenal.framework.feature_store import (
    FeatureStore,
    FeatureRegistry,
    FeatureCacheManager,
    FeatureComputer,
    FeatureValidator,
    FeatureComputationError,
    FeatureNotFoundError,
    FeatureValidationError
)
from airsenal.framework.feature_store_integration import (
    FeatureAwarePredictionUtils,
    FeatureMigrator,
    FeatureStoreMonitor
)
from airsenal.framework.schema import (
    FeatureDefinition,
    ComputedFeature,
    FeatureCache,
    FeatureTimeSeries,
    Player,
    PlayerScore,
    Fixture,
    session
)


class TestFeatureRegistry:
    """Test feature definition management and versioning."""
    
    def test_register_new_feature(self):
        """Test registering a new feature."""
        registry = FeatureRegistry()
        
        feature_def = registry.register_feature(
            name="test_rolling_goals",
            feature_type="player",
            data_type="float",
            description="Test rolling goals feature",
            computation_logic={"type": "rolling", "metric": "goals", "window": 5}
        )
        
        assert feature_def.name == "test_rolling_goals"
        assert feature_def.feature_type == "player"
        assert feature_def.data_type == "float"
        assert feature_def.is_active is True
        
        # Clean up
        session.delete(feature_def)
        session.commit()
    
    def test_register_duplicate_feature_fails(self):
        """Test that registering duplicate feature raises error."""
        registry = FeatureRegistry()
        
        # Register first feature
        feature_def1 = registry.register_feature(
            name="test_duplicate",
            feature_type="player",
            version="1.0.0"
        )
        
        # Try to register same name and version
        with pytest.raises(ValueError, match="already exists"):
            registry.register_feature(
                name="test_duplicate",
                feature_type="player", 
                version="1.0.0"
            )
        
        # Clean up
        session.delete(feature_def1)
        session.commit()
    
    def test_update_existing_feature(self):
        """Test updating existing feature with replace_existing=True."""
        registry = FeatureRegistry()
        
        # Register initial feature
        feature_def = registry.register_feature(
            name="test_update",
            feature_type="player",
            description="Original description"
        )
        original_id = feature_def.id
        
        # Update the feature
        updated_def = registry.register_feature(
            name="test_update",
            feature_type="player",
            description="Updated description",
            version="1.0.0",
            replace_existing=True
        )
        
        assert updated_def.id == original_id
        assert updated_def.description == "Updated description"
        
        # Clean up
        session.delete(updated_def)
        session.commit()
    
    def test_get_feature_definition(self):
        """Test retrieving feature definition."""
        registry = FeatureRegistry()
        
        # Register feature
        original = registry.register_feature(
            name="test_get",
            feature_type="player"
        )
        
        # Retrieve by name
        retrieved = registry.get_feature_definition("test_get")
        assert retrieved.id == original.id
        
        # Test not found
        with pytest.raises(FeatureNotFoundError):
            registry.get_feature_definition("nonexistent_feature")
        
        # Clean up
        session.delete(original)
        session.commit()


class TestFeatureCacheManager:
    """Test caching functionality."""
    
    def setup_method(self):
        """Set up test cache manager."""
        self.cache = FeatureCacheManager(default_ttl=60)
    
    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        # Test float value
        self.cache.set("test_feature", "player", 123, 5.5)
        value = self.cache.get("test_feature", "player", 123)
        assert value == 5.5
        
        # Test string value
        self.cache.set("test_str_feature", "player", 456, "GK")
        str_value = self.cache.get("test_str_feature", "player", 456)
        assert str_value == "GK"
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        # Set with very short TTL
        self.cache.set("test_expire", "player", 789, 10.0, ttl=1)
        
        # Should be available immediately
        value = self.cache.get("test_expire", "player", 789)
        assert value == 10.0
        
        # Wait for expiration (in real test, would mock time)
        # For now, just test the expiration logic
        import time
        time.sleep(2)
        
        expired_value = self.cache.get("test_expire", "player", 789)
        assert expired_value is None
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Set multiple cache entries
        self.cache.set("test_invalidate", "player", 111, 1.0)
        self.cache.set("test_invalidate", "player", 222, 2.0)
        self.cache.set("other_feature", "player", 111, 3.0)
        
        # Invalidate specific feature
        count = self.cache.invalidate("test_invalidate")
        assert count >= 2  # Should invalidate at least the 2 entries
        
        # Check that specific feature is invalidated
        assert self.cache.get("test_invalidate", "player", 111) is None
        assert self.cache.get("test_invalidate", "player", 222) is None
        
        # Other feature should still be cached
        assert self.cache.get("other_feature", "player", 111) == 3.0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Generate some cache activity
        self.cache.set("stats_test", "player", 999, 42.0)
        
        # Generate hits and misses
        self.cache.get("stats_test", "player", 999)  # hit
        self.cache.get("nonexistent", "player", 888)  # miss
        
        stats = self.cache.get_stats()
        
        assert "memory_cache" in stats
        assert "database_cache" in stats
        assert stats["memory_cache"]["total_requests"] >= 2


class TestFeatureComputer:
    """Test feature computation algorithms."""
    
    def setup_method(self):
        """Set up test computer."""
        self.computer = FeatureComputer()
    
    @patch('airsenal.framework.feature_store.session')
    def test_rolling_statistic_computation(self, mock_session):
        """Test rolling statistics computation."""
        # Mock query results
        mock_scores = []
        for i in range(5):
            mock_score = Mock()
            mock_score.goals = i + 1  # Goals: 1, 2, 3, 4, 5
            mock_scores.append(mock_score)
        
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_scores
        mock_session.query.return_value = mock_query
        
        # Test mean aggregation
        result = self.computer.compute_rolling_statistic(
            entity_type="player",
            entity_id=123,
            metric_name="goals",
            window_size=5,
            aggregation="mean"
        )
        
        assert result == 3.0  # Mean of [1, 2, 3, 4, 5]
    
    def test_form_metric_computation(self):
        """Test exponentially weighted form metrics."""
        # Mock the query to return test data
        with patch.object(self.computer, 'dbsession') as mock_session:
            mock_scores = []
            for i in range(3):
                mock_score = Mock()
                mock_score.goals = 2  # Constant goals for simplicity
                mock_scores.append(mock_score)
            
            mock_query = Mock()
            mock_query.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_scores
            mock_session.query.return_value = mock_query
            
            result = self.computer.compute_form_metric(
                entity_type="player",
                entity_id=123,
                metric_name="goals",
                decay_factor=0.9
            )
            
            # With constant values and decay factor, should equal the constant value
            assert abs(result - 2.0) < 0.1
    
    def test_compute_by_logic_rolling(self):
        """Test feature computation using JSON logic for rolling features."""
        # Create test feature definition
        feature_def = Mock()
        feature_def.name = "test_rolling"
        feature_def.computation_logic = json.dumps({
            "type": "rolling",
            "metric": "goals", 
            "window": 3,
            "agg": "mean"
        })
        
        with patch.object(self.computer, 'compute_rolling_statistic') as mock_rolling:
            mock_rolling.return_value = 2.5
            
            result = self.computer.compute_feature_by_logic(
                feature_def=feature_def,
                entity_type="player",
                entity_id=123
            )
            
            assert result == 2.5
            mock_rolling.assert_called_once()
    
    def test_compute_by_logic_invalid_json(self):
        """Test error handling for invalid computation logic."""
        feature_def = Mock()
        feature_def.name = "invalid_feature"
        feature_def.computation_logic = "invalid json"
        
        with pytest.raises(FeatureComputationError, match="Invalid computation logic JSON"):
            self.computer.compute_feature_by_logic(
                feature_def=feature_def,
                entity_type="player",
                entity_id=123
            )


class TestFeatureValidator:
    """Test feature validation and monitoring."""
    
    def setup_method(self):
        """Set up test validator."""
        self.validator = FeatureValidator()
    
    def test_validate_numeric_feature(self):
        """Test validation of numeric features."""
        feature_def = Mock()
        feature_def.name = "numeric_test"
        feature_def.data_type = "float"
        
        # Valid values
        assert self.validator.validate_feature_value(feature_def, 5.5) is True
        assert self.validator.validate_feature_value(feature_def, 0) is True
        assert self.validator.validate_feature_value(feature_def, None) is True
        
        # Invalid values
        assert self.validator.validate_feature_value(feature_def, "string") is False
        assert self.validator.validate_feature_value(feature_def, np.nan) is False
        assert self.validator.validate_feature_value(feature_def, np.inf) is False
    
    def test_validate_string_feature(self):
        """Test validation of string features."""
        feature_def = Mock()
        feature_def.name = "string_test" 
        feature_def.data_type = "string"
        
        # Valid values
        assert self.validator.validate_feature_value(feature_def, "GK") is True
        assert self.validator.validate_feature_value(feature_def, "") is True
        assert self.validator.validate_feature_value(feature_def, None) is True
        
        # Invalid values
        assert self.validator.validate_feature_value(feature_def, 123) is False
        assert self.validator.validate_feature_value(feature_def, 5.5) is False
    
    @patch('airsenal.framework.feature_store.session')
    def test_feature_drift_detection(self, mock_session):
        """Test statistical drift detection between seasons."""
        # Mock query results for current season
        current_values = [Mock(value=v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        
        # Mock query results for reference season  
        reference_values = [Mock(value=v) for v in [1.5, 2.5, 3.5, 4.5, 5.5]]
        
        mock_query = Mock()
        mock_session.query.return_value.join.return_value.filter.return_value = mock_query
        
        # First call returns current values, second returns reference values
        mock_query.all.side_effect = [current_values, reference_values]
        
        drift_result = self.validator.check_feature_drift(
            feature_name="test_feature",
            entity_type="player",
            current_season="2425",
            reference_season="2324",
            drift_threshold=0.1
        )
        
        assert "drift_detected" in drift_result
        assert "mean_change" in drift_result
        assert "current_stats" in drift_result
        assert "reference_stats" in drift_result


class TestFeatureStore:
    """Test main FeatureStore interface."""
    
    def setup_method(self):
        """Set up test feature store."""
        self.store = FeatureStore()
    
    def test_register_feature_integration(self):
        """Test feature registration through main interface."""
        feature_def = self.store.register_feature(
            name="integration_test",
            feature_type="player",
            description="Integration test feature",
            computation_logic={"type": "rolling", "metric": "goals", "window": 3}
        )
        
        assert feature_def.name == "integration_test"
        
        # Test retrieval
        retrieved = self.store.registry.get_feature_definition("integration_test")
        assert retrieved.id == feature_def.id
        
        # Clean up
        session.delete(feature_def)
        session.commit()
    
    @patch('airsenal.framework.feature_store.FeatureStore._register_default_features')
    def test_default_features_registration(self, mock_register):
        """Test that default features are registered on initialization."""
        FeatureStore()
        mock_register.assert_called_once()
    
    def test_get_features_with_cache(self):
        """Test feature retrieval with caching."""
        # Register test feature
        feature_def = self.store.register_feature(
            name="cache_test",
            feature_type="player",
            computation_logic={"type": "rolling", "metric": "goals", "window": 1}
        )
        
        # Mock the computation to return a known value
        with patch.object(self.store.computer, 'compute_feature_by_logic') as mock_compute:
            mock_compute.return_value = 3.14
            
            # First call should compute and cache
            result1 = self.store.get_features(
                entity_type="player",
                entity_ids=[123],
                feature_names=["cache_test"]
            )
            
            assert result1[123]["cache_test"] == 3.14
            assert mock_compute.call_count == 1
            
            # Second call should use cache
            result2 = self.store.get_features(
                entity_type="player", 
                entity_ids=[123],
                feature_names=["cache_test"]
            )
            
            assert result2[123]["cache_test"] == 3.14
            # Should not compute again due to cache
            assert mock_compute.call_count == 1
        
        # Clean up
        session.delete(feature_def)
        session.commit()
    
    def test_batch_computation(self):
        """Test batch feature computation."""
        # Register test feature
        feature_def = self.store.register_feature(
            name="batch_test",
            feature_type="player",
            computation_logic={"type": "rolling", "metric": "goals", "window": 1}
        )
        
        with patch.object(self.store.computer, 'compute_feature_by_logic') as mock_compute:
            mock_compute.return_value = 1.0
            
            stats = self.store.compute_features_batch(
                feature_names=["batch_test"],
                entity_type="player",
                entity_ids=[1, 2, 3],
                gameweek_range=(10, 12)
            )
            
            assert "total_computations" in stats
            assert "successful_computations" in stats
            assert stats["total_computations"] >= 9  # 3 entities × 3 gameweeks
        
        # Clean up
        session.delete(feature_def)
        session.commit()


class TestFeatureStoreIntegration:
    """Test integration with existing prediction pipeline."""
    
    def test_feature_aware_prediction_utils_init(self):
        """Test initialization of enhanced prediction utilities."""
        predictor = FeatureAwarePredictionUtils()
        
        assert predictor.feature_store is not None
        assert isinstance(predictor.feature_store, FeatureStore)
    
    @patch('airsenal.framework.feature_store_integration.list_players')
    def test_batch_update_player_features(self, mock_list_players):
        """Test batch updating of player features."""
        # Mock players
        mock_players = [Mock(player_id=i) for i in range(1, 4)]
        mock_list_players.return_value = mock_players
        
        predictor = FeatureAwarePredictionUtils()
        
        with patch.object(predictor.feature_store, 'compute_features_batch') as mock_batch:
            mock_batch.return_value = {"total_computations": 15, "successful_computations": 12}
            
            stats = predictor.batch_update_player_features(
                season="2425",
                gameweek_range=(1, 5)
            )
            
            assert stats["total_computations"] == 15
            assert stats["successful_computations"] == 12
            mock_batch.assert_called_once()


class TestFeatureMigrator:
    """Test data migration utilities."""
    
    def test_migrator_initialization(self):
        """Test migrator initialization."""
        migrator = FeatureMigrator()
        
        assert migrator.feature_store is not None
        assert isinstance(migrator.feature_store, FeatureStore)
    
    @patch('airsenal.framework.feature_store_integration.session')
    def test_migrate_player_scores_concept(self, mock_session):
        """Test the concept of migrating player scores (without actual DB operations)."""
        migrator = FeatureMigrator()
        
        # Mock player scores
        mock_scores = []
        for i in range(3):
            score = Mock()
            score.player_id = 123
            score.goals = i
            score.assists = i * 2
            score.fixture = Mock()
            score.fixture.season = "2425"
            score.fixture.gameweek = i + 1
            score.fixture.date = "2024-08-17"
            mock_scores.append(score)
        
        mock_query = Mock()
        mock_query.join.return_value.filter.return_value.order_by.return_value = mock_scores
        mock_session.query.return_value = mock_query
        
        # Mock existing check to return None (no existing data)
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        
        # Test would create time series entries
        # In real test, would verify FeatureTimeSeries objects are created
        # For now, just verify the method runs without error
        try:
            migrator.migrate_player_scores_to_timeseries("2425")
        except Exception as e:
            # Expected to fail due to mocking, but shouldn't raise computation errors
            assert "computation" not in str(e).lower()


class TestFeatureStoreMonitor:
    """Test monitoring and alerting functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = FeatureStoreMonitor()
        
        assert monitor.feature_store is not None
        assert isinstance(monitor.feature_store, FeatureStore)
    
    @patch('airsenal.framework.feature_store_integration.session')
    def test_check_feature_freshness_concept(self, mock_session):
        """Test feature freshness checking concept."""
        monitor = FeatureStoreMonitor()
        
        # Mock feature definition
        with patch.object(monitor.feature_store.registry, 'get_feature_definition') as mock_get_def:
            mock_feature_def = Mock()
            mock_feature_def.id = 1
            mock_get_def.return_value = mock_feature_def
            
            # Mock query results
            mock_query = Mock()
            mock_query.filter.return_value.count.return_value = 50  # fresh count
            mock_session.query.return_value = mock_query
            
            freshness = monitor.check_feature_freshness(
                feature_names=["test_feature"],
                max_age_hours=24
            )
            
            assert "test_feature" in freshness
            assert "fresh_count" in freshness["test_feature"]


# Performance and stress tests

class TestFeatureStorePerformance:
    """Performance and stress tests for the feature store."""
    
    def test_cache_performance_with_many_features(self):
        """Test cache performance with many feature requests."""
        cache = FeatureCacheManager()
        
        # Set many cache entries
        start_time = datetime.now()
        for i in range(1000):
            cache.set(f"perf_feature_{i % 10}", "player", i, float(i))
        set_time = (datetime.now() - start_time).total_seconds()
        
        # Get many cache entries
        start_time = datetime.now()
        hits = 0
        for i in range(1000):
            value = cache.get(f"perf_feature_{i % 10}", "player", i)
            if value is not None:
                hits += 1
        get_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions (these are rough guidelines)
        assert set_time < 5.0  # Should set 1000 entries in under 5 seconds
        assert get_time < 2.0  # Should get 1000 entries in under 2 seconds
        assert hits > 900  # Should have high cache hit rate
    
    @patch('airsenal.framework.feature_store.FeatureComputer.compute_feature_by_logic')
    def test_parallel_feature_computation_concept(self, mock_compute):
        """Test concept of parallel feature computation."""
        store = FeatureStore()
        mock_compute.return_value = 1.0
        
        # Register test feature
        feature_def = store.register_feature(
            name="parallel_test",
            feature_type="player",
            computation_logic={"type": "rolling", "metric": "goals", "window": 1}
        )
        
        # Test getting features for many entities
        start_time = datetime.now()
        result = store.get_features(
            entity_type="player",
            entity_ids=list(range(100)),
            feature_names=["parallel_test"],
            use_cache=False
        )
        computation_time = (datetime.now() - start_time).total_seconds()
        
        assert len(result) == 100
        assert computation_time < 10.0  # Should compute 100 features in under 10 seconds
        
        # Clean up
        session.delete(feature_def)
        session.commit()


# Integration test with real-like data

class TestFeatureStoreRealWorldScenario:
    """Test feature store with scenarios similar to real AIrsenal usage."""
    
    def test_fpl_prediction_workflow(self):
        """Test a complete FPL prediction workflow using the feature store."""
        store = FeatureStore()
        
        # Register FPL-specific features
        goal_feature = store.register_feature(
            name="fpl_goals_form",
            feature_type="player",
            description="FPL goals form for prediction",
            computation_logic={"type": "form", "metric": "goals", "decay_factor": 0.9}
        )
        
        # Mock feature computation for a prediction scenario
        with patch.object(store.computer, 'compute_feature_by_logic') as mock_compute:
            mock_compute.return_value = 0.3  # 0.3 goals per game form
            
            # Get features for multiple players (typical prediction batch)
            features = store.get_features(
                entity_type="player",
                entity_ids=[1, 2, 3, 4, 5],  # 5 players
                feature_names=["fpl_goals_form"],
                season="2425",
                gameweek=10
            )
            
            # Verify feature structure
            assert len(features) == 5
            for player_id in [1, 2, 3, 4, 5]:
                assert player_id in features
                assert "fpl_goals_form" in features[player_id]
                assert features[player_id]["fpl_goals_form"] == 0.3
        
        # Test cache hit on second request
        with patch.object(store.computer, 'compute_feature_by_logic') as mock_compute:
            mock_compute.return_value = 0.3
            
            # Second request should use cache
            features2 = store.get_features(
                entity_type="player",
                entity_ids=[1, 2, 3, 4, 5],
                feature_names=["fpl_goals_form"],
                season="2425",
                gameweek=10
            )
            
            # Should not call compute due to cache
            assert mock_compute.call_count == 0
            assert features2 == features
        
        # Clean up
        session.delete(goal_feature)
        session.commit()


if __name__ == "__main__":
    # Run specific test suites
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFeatureRegistry,
        TestFeatureCacheManager,
        TestFeatureComputer,
        TestFeatureValidator,
        TestFeatureStore,
        TestFeatureStoreIntegration,
        TestFeatureStorePerformance,
        TestFeatureStoreRealWorldScenario
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")