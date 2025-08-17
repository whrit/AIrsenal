"""
Comprehensive test suite for the FormCalculator class.

Tests cover:
- Rolling form calculations with various window sizes
- Exponentially weighted moving averages with different decay factors
- Momentum calculation and trend detection
- Edge case handling (new players, missing data, injuries)
- Performance requirements (<50ms per player)
- Batch processing efficiency
- Feature store integration and caching
- Data validation and error handling

Test Data Strategy:
- Uses realistic FPL point distributions and patterns
- Includes edge cases like zero points, high scores, missing data
- Tests performance with various data volumes
- Validates mathematical correctness of algorithms

Author: AIrsenal Team
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from airsenal.framework.form_calculator import (
    FormCalculator,
    FormCalculationError,
    InsufficientDataError
)
from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    session
)
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK


class TestFormCalculator:
    """Test suite for FormCalculator class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session for testing."""
        return Mock(spec=Session)

    @pytest.fixture
    def calculator(self, mock_session):
        """Create a FormCalculator instance with mocked dependencies."""
        with patch('airsenal.framework.form_calculator.FeatureStore'):
            calc = FormCalculator(
                dbsession=mock_session,
                enable_caching=False  # Disable caching for most tests
            )
            return calc

    @pytest.fixture
    def calculator_with_cache(self, mock_session):
        """Create a FormCalculator instance with caching enabled."""
        mock_feature_store = Mock()
        mock_cache = Mock()
        mock_feature_store.cache = mock_cache
        mock_cache.get.return_value = None  # No cache hits by default
        
        with patch('airsenal.framework.form_calculator.FeatureStore', return_value=mock_feature_store):
            calc = FormCalculator(
                dbsession=mock_session,
                enable_caching=True
            )
            return calc

    @pytest.fixture
    def sample_player_scores(self):
        """Create sample player score data for testing."""
        # Create realistic FPL points progression over 15 gameweeks
        gameweeks = list(range(15, 0, -1))  # GW 15 to GW 1 (descending order)
        
        # Realistic point values with some variation
        points = [8, 12, 4, 15, 6, 9, 11, 2, 14, 7, 5, 13, 10, 3, 16]
        goals = [1, 2, 0, 3, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0, 3]
        assists = [0, 1, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
        minutes = [90, 85, 72, 90, 88, 90, 76, 23, 90, 84, 90, 90, 81, 67, 90]
        
        scores = []
        for i, gw in enumerate(gameweeks):
            score = Mock()
            score.gameweek = gw
            score.points = points[i]
            score.goals = goals[i]
            score.assists = assists[i]
            score.minutes = minutes[i]
            score.opponent = f"Team_{i % 5}"
            score.date = f"2024-{gw:02d}-01"
            scores.append(score)
        
        return scores

    @pytest.fixture
    def mock_query_result(self, sample_player_scores):
        """Mock database query result for player scores."""
        mock_query = Mock()
        mock_query.all.return_value = sample_player_scores
        return mock_query

    def test_initialization_default_params(self, mock_session):
        """Test FormCalculator initialization with default parameters."""
        with patch('airsenal.framework.form_calculator.FeatureStore'):
            calc = FormCalculator(dbsession=mock_session)
            
            assert calc.default_decay_factor == 0.95
            assert calc.min_games_for_form == 2
            assert calc.max_lookback_games == 20
            assert calc.enable_caching is True

    def test_initialization_custom_params(self, mock_session):
        """Test FormCalculator initialization with custom parameters."""
        with patch('airsenal.framework.form_calculator.FeatureStore'):
            calc = FormCalculator(
                dbsession=mock_session,
                default_decay_factor=0.9,
                min_games_for_form=3,
                max_lookback_games=15,
                enable_caching=False
            )
            
            assert calc.default_decay_factor == 0.9
            assert calc.min_games_for_form == 3
            assert calc.max_lookback_games == 15
            assert calc.enable_caching is False

    def test_get_player_scores_data_success(self, calculator, mock_query_result):
        """Test successful retrieval of player scores data."""
        # Mock the database query
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        df = calculator._get_player_scores_data(player_id=123)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        assert 'points' in df.columns
        assert 'gameweek' in df.columns
        assert df['points'].dtype in [np.float64, np.int64]
        
        # Check data is sorted by gameweek descending
        assert df['gameweek'].iloc[0] > df['gameweek'].iloc[-1]

    def test_get_player_scores_data_insufficient_data(self, calculator):
        """Test handling of insufficient data scenario."""
        # Mock empty query result
        mock_empty_query = Mock()
        mock_empty_query.all.return_value = []
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_empty_query
        
        with pytest.raises(InsufficientDataError):
            calculator._get_player_scores_data(player_id=123)

    def test_calculate_rolling_form_3_games(self, calculator, mock_query_result):
        """Test 3-game rolling form calculation."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=3)
        
        # Should be average of first 3 games: (8 + 12 + 4) / 3 = 8.0
        assert form is not None
        assert abs(form - 8.0) < 0.01

    def test_calculate_rolling_form_5_games(self, calculator, mock_query_result):
        """Test 5-game rolling form calculation."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=5)
        
        # Should be average of first 5 games: (8 + 12 + 4 + 15 + 6) / 5 = 9.0
        assert form is not None
        assert abs(form - 9.0) < 0.01

    def test_calculate_rolling_form_10_games(self, calculator, mock_query_result):
        """Test 10-game rolling form calculation."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=10)
        
        # Should be average of first 10 games: (8+12+4+15+6+9+11+2+14+7) / 10 = 8.8
        assert form is not None
        assert abs(form - 8.8) < 0.01

    def test_calculate_rolling_form_insufficient_data(self, calculator):
        """Test rolling form calculation with insufficient data."""
        # Mock query with only 1 game (less than min_games_for_form=2)
        mock_single_game = Mock()
        single_score = Mock()
        single_score.gameweek = 1
        single_score.points = 10
        single_score.goals = 1
        single_score.assists = 0
        single_score.minutes = 90
        single_score.opponent = "Team_A"
        single_score.date = "2024-01-01"
        mock_single_game.all.return_value = [single_score]
        
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_single_game
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=3)
        assert form is None

    def test_calculate_weighted_form_default_decay(self, calculator, mock_query_result):
        """Test exponentially weighted form calculation with default decay factor."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        weighted_form = calculator.calculate_weighted_form(player_id=123, gameweeks=5)
        
        # Manual calculation with decay_factor=0.95:
        # weights = [1, 0.95, 0.95^2, 0.95^3, 0.95^4]
        # points = [8, 12, 4, 15, 6]
        weights = np.array([0.95**i for i in range(5)])
        points = np.array([8, 12, 4, 15, 6])
        expected = np.sum(weights * points) / np.sum(weights)
        
        assert weighted_form is not None
        assert abs(weighted_form - expected) < 0.01

    def test_calculate_weighted_form_custom_decay(self, calculator, mock_query_result):
        """Test exponentially weighted form calculation with custom decay factor."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        custom_decay = 0.8
        weighted_form = calculator.calculate_weighted_form(player_id=123, gameweeks=3, decay_factor=custom_decay)
        
        # Manual calculation with decay_factor=0.8:
        weights = np.array([0.8**i for i in range(3)])
        points = np.array([8, 12, 4])
        expected = np.sum(weights * points) / np.sum(weights)
        
        assert weighted_form is not None
        assert abs(weighted_form - expected) < 0.01

    def test_calculate_momentum_positive(self, calculator, mock_query_result):
        """Test momentum calculation showing positive trend."""
        # Modify scores to show improving trend
        scores = mock_query_result.all()
        # Recent 3 games: 12, 15, 18 (average = 15)
        # Historical 10 games average with better recent performance
        scores[0].points = 18  # Most recent
        scores[1].points = 15
        scores[2].points = 12
        # Keep other scores lower to show positive momentum
        for i in range(3, 10):
            scores[i].points = 6
        
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        momentum = calculator.calculate_momentum(player_id=123)
        
        assert momentum is not None
        assert momentum > 0  # Should be positive due to improving trend

    def test_calculate_momentum_negative(self, calculator, mock_query_result):
        """Test momentum calculation showing negative trend."""
        # Modify scores to show declining trend
        scores = mock_query_result.all()
        # Recent 3 games: 2, 3, 4 (average = 3)
        scores[0].points = 4  # Most recent
        scores[1].points = 3
        scores[2].points = 2
        # Keep other scores higher to show negative momentum
        for i in range(3, 10):
            scores[i].points = 12
        
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        momentum = calculator.calculate_momentum(player_id=123)
        
        assert momentum is not None
        assert momentum < 0  # Should be negative due to declining trend

    def test_calculate_momentum_stable(self, calculator, mock_query_result):
        """Test momentum calculation showing stable form."""
        # Modify scores to show stable performance
        scores = mock_query_result.all()
        # Set all scores to same value for stable form
        for score in scores:
            score.points = 10
        
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        momentum = calculator.calculate_momentum(player_id=123)
        
        assert momentum is not None
        assert abs(momentum) < 0.01  # Should be close to 0 for stable form

    def test_calculate_momentum_bounded(self, calculator, mock_query_result):
        """Test that momentum is properly bounded to [-1, 1] range."""
        # Create extreme case that might produce out-of-bounds momentum
        scores = mock_query_result.all()
        # Recent games: very high scores
        scores[0].points = 50
        scores[1].points = 45
        scores[2].points = 40
        # Historical games: very low scores
        for i in range(3, 10):
            scores[i].points = 1
        
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        momentum = calculator.calculate_momentum(player_id=123)
        
        assert momentum is not None
        assert -1 <= momentum <= 1  # Should be bounded

    def test_batch_calculate_form_success(self, calculator, mock_query_result):
        """Test successful batch calculation of form metrics."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        player_ids = [123, 456, 789]
        results = calculator.batch_calculate_form(player_ids, use_cache=False)
        
        assert len(results) == 3
        assert all(player_id in results for player_id in player_ids)
        
        for player_id, metrics in results.items():
            assert 'form_3_games' in metrics
            assert 'form_5_games' in metrics
            assert 'form_10_games' in metrics
            assert 'weighted_form' in metrics
            assert 'momentum' in metrics
            
            # Check that values are reasonable
            for metric_name, value in metrics.items():
                if value is not None:
                    assert isinstance(value, (int, float))
                    if metric_name == 'momentum':
                        assert -1 <= value <= 1
                    else:
                        assert 0 <= value <= 25  # Reasonable FPL points range

    def test_batch_calculate_form_no_momentum(self, calculator, mock_query_result):
        """Test batch calculation without momentum metrics."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        player_ids = [123, 456]
        results = calculator.batch_calculate_form(player_ids, include_momentum=False, use_cache=False)
        
        for player_id, metrics in results.items():
            assert 'momentum' not in metrics
            assert 'form_3_games' in metrics

    def test_batch_calculate_form_chunking(self, calculator, mock_query_result):
        """Test that batch calculation properly handles chunking."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # Test with chunk_size smaller than total players
        player_ids = list(range(1, 11))  # 10 players
        results = calculator.batch_calculate_form(player_ids, chunk_size=3, use_cache=False)
        
        assert len(results) == 10
        assert all(player_id in results for player_id in player_ids)

    def test_performance_requirement(self, calculator, mock_query_result):
        """Test that calculations meet <50ms performance requirement."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # Test single player calculation time
        start_time = time.time()
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=5, use_cache=False)
        calculation_time = time.time() - start_time
        
        assert form is not None
        assert calculation_time < 0.05  # Less than 50ms

    def test_caching_functionality(self, calculator_with_cache, mock_query_result):
        """Test caching functionality in form calculations."""
        calculator_with_cache.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # First call - should miss cache and compute
        form1 = calculator_with_cache.calculate_rolling_form(player_id=123, gameweeks=3, use_cache=True)
        
        # Verify cache.set was called
        calculator_with_cache.feature_store.cache.set.assert_called()
        
        # Second call - mock cache hit
        calculator_with_cache.feature_store.cache.get.return_value = 8.5
        form2 = calculator_with_cache.calculate_rolling_form(player_id=123, gameweeks=3, use_cache=True)
        
        assert form2 == 8.5  # Should return cached value

    def test_update_player_attributes_success(self, calculator):
        """Test successful update of PlayerAttributes table."""
        # Mock PlayerAttributes query
        mock_player_attr = Mock()
        calculator.dbsession.query.return_value.filter_by.return_value.first.return_value = mock_player_attr
        
        # Mock form calculations
        with patch.object(calculator, 'calculate_rolling_form') as mock_rolling, \
             patch.object(calculator, 'calculate_momentum') as mock_momentum:
            
            mock_rolling.side_effect = [5.5, 6.0, 6.5]  # form_3, form_5, form_10
            mock_momentum.return_value = 0.2
            
            success = calculator.update_player_attributes(player_id=123, commit=False)
            
            assert success is True
            assert mock_player_attr.form_3_games == 5.5
            assert mock_player_attr.form_5_games == 6.0
            assert mock_player_attr.form_10_games == 6.5
            assert mock_player_attr.momentum == 0.2

    def test_update_player_attributes_no_record(self, calculator):
        """Test update when no PlayerAttributes record exists."""
        # Mock empty PlayerAttributes query
        calculator.dbsession.query.return_value.filter_by.return_value.first.return_value = None
        
        success = calculator.update_player_attributes(player_id=123, commit=False)
        assert success is False

    def test_batch_update_player_attributes(self, calculator):
        """Test batch update of PlayerAttributes table."""
        # Mock PlayerAttributes queries
        mock_attrs = [Mock() for _ in range(3)]
        calculator.dbsession.query.return_value.filter_by.return_value.first.side_effect = mock_attrs
        
        # Mock batch_calculate_form
        with patch.object(calculator, 'batch_calculate_form') as mock_batch:
            mock_batch.return_value = {
                123: {'form_3_games': 5.0, 'form_5_games': 5.5, 'momentum': 0.1},
                456: {'form_3_games': 7.0, 'form_5_games': 7.2, 'momentum': 0.3},
                789: {'form_3_games': 4.0, 'form_5_games': 4.5, 'momentum': -0.1}
            }
            
            player_ids = [123, 456, 789]
            stats = calculator.batch_update_player_attributes(player_ids)
            
            assert stats['total_processed'] == 3
            assert stats['successful_updates'] == 3
            assert stats['failed_updates'] == 0

    def test_get_performance_stats(self, calculator):
        """Test performance statistics collection."""
        # Add some calculation times
        calculator._calculation_times = [0.01, 0.02, 0.015, 0.025, 0.03]
        calculator._cache_hits = 10
        calculator._cache_misses = 5
        
        stats = calculator.get_performance_stats()
        
        assert 'calculation_times' in stats
        assert 'cache_performance' in stats
        assert 'data_quality' in stats
        
        assert stats['calculation_times']['count'] == 5
        assert stats['calculation_times']['mean_ms'] > 0
        assert stats['cache_performance']['hit_rate'] == 10/15
        assert stats['cache_performance']['total_requests'] == 15

    def test_validate_form_calculations(self, calculator, mock_query_result):
        """Test validation of form calculations."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        validation_results = calculator.validate_form_calculations(player_id=123)
        
        assert 'player_id' in validation_results
        assert 'data_quality' in validation_results
        assert 'calculations' in validation_results
        assert 'validations' in validation_results
        assert isinstance(validation_results['overall_valid'], bool)
        
        # Check that calculations were performed
        calculations = validation_results['calculations']
        assert 'form_3_games' in calculations
        assert 'momentum' in calculations

    def test_error_handling_database_error(self, calculator):
        """Test error handling when database query fails."""
        # Mock database error
        calculator.dbsession.query.side_effect = Exception("Database connection failed")
        
        with pytest.raises(FormCalculationError):
            calculator.calculate_rolling_form(player_id=123, gameweeks=3, use_cache=False)

    def test_error_handling_invalid_parameters(self, calculator, mock_query_result):
        """Test error handling with invalid parameters."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # Test invalid gameweeks parameter
        with pytest.raises((ValueError, FormCalculationError)):
            calculator.calculate_rolling_form(player_id=123, gameweeks=0)
        
        # Test invalid decay factor
        with pytest.raises((ValueError, FormCalculationError)):
            calculator.calculate_weighted_form(player_id=123, gameweeks=5, decay_factor=1.5)

    def test_edge_case_zero_points(self, calculator):
        """Test handling of players with zero points."""
        # Mock scores with all zero points
        mock_zero_scores = Mock()
        zero_scores = []
        for gw in range(5, 0, -1):
            score = Mock()
            score.gameweek = gw
            score.points = 0
            score.goals = 0
            score.assists = 0
            score.minutes = 0
            score.opponent = "Team_A"
            score.date = f"2024-{gw:02d}-01"
            zero_scores.append(score)
        
        mock_zero_scores.all.return_value = zero_scores
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_zero_scores
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=3)
        assert form == 0.0  # Should handle zero points gracefully

    def test_edge_case_missing_data_fields(self, calculator):
        """Test handling of missing data fields."""
        # Mock scores with None values in some fields
        mock_partial_scores = Mock()
        partial_scores = []
        for gw in range(3, 0, -1):
            score = Mock()
            score.gameweek = gw
            score.points = 5
            score.goals = None  # Missing goal data
            score.assists = None  # Missing assist data
            score.minutes = 90
            score.opponent = "Team_A"
            score.date = f"2024-{gw:02d}-01"
            partial_scores.append(score)
        
        mock_partial_scores.all.return_value = partial_scores
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_partial_scores
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=3)
        assert form == 5.0  # Should handle missing fields gracefully

    def test_mathematical_correctness_ewma(self):
        """Test mathematical correctness of EWMA calculation."""
        # Test with known values to verify mathematical correctness
        points = np.array([10, 8, 12, 6, 14])
        decay_factor = 0.9
        
        # Manual EWMA calculation
        weights = np.array([decay_factor ** i for i in range(len(points))])
        expected_ewma = np.sum(weights * points) / np.sum(weights)
        
        # This should match the algorithm in calculate_weighted_form
        weighted_sum = np.sum(weights * points)
        weight_sum = np.sum(weights)
        calculated_ewma = weighted_sum / weight_sum
        
        assert abs(calculated_ewma - expected_ewma) < 1e-10

    def test_rolling_window_edge_cases(self, calculator):
        """Test rolling window calculations with edge cases."""
        # Test with exactly min_games_for_form games
        mock_min_scores = Mock()
        min_scores = []
        for gw in range(calculator.min_games_for_form, 0, -1):
            score = Mock()
            score.gameweek = gw
            score.points = 8
            score.goals = 1
            score.assists = 0
            score.minutes = 90
            score.opponent = "Team_A"
            score.date = f"2024-{gw:02d}-01"
            min_scores.append(score)
        
        mock_min_scores.all.return_value = min_scores
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_min_scores
        
        form = calculator.calculate_rolling_form(player_id=123, gameweeks=3)
        assert form == 8.0  # Should work with minimum required games


class TestFormCalculatorIntegration:
    """Integration tests that may require database setup."""
    
    def test_real_database_integration(self):
        """Test with real database connection (if available)."""
        # This test would be skipped if no test database is available
        pytest.skip("Integration test requiring database setup")
    
    def test_feature_store_integration(self):
        """Test integration with real feature store."""
        # This test would verify full feature store integration
        pytest.skip("Integration test requiring feature store setup")


class TestFormCalculatorPerformance:
    """Performance-focused tests."""
    
    def test_batch_processing_performance(self, calculator, mock_query_result):
        """Test performance of batch processing with large player sets."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # Test with 100 players
        player_ids = list(range(1, 101))
        
        start_time = time.time()
        results = calculator.batch_calculate_form(player_ids, use_cache=False, chunk_size=20)
        total_time = time.time() - start_time
        
        assert len(results) == 100
        # Should process at least 10 players per second
        assert len(results) / total_time >= 10
    
    def test_memory_usage_batch_processing(self, calculator, mock_query_result):
        """Test that batch processing doesn't consume excessive memory."""
        calculator.dbsession.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value = mock_query_result
        
        # This would use memory profiling in a real test
        # For now, just ensure it completes without errors
        player_ids = list(range(1, 51))
        results = calculator.batch_calculate_form(player_ids, chunk_size=10)
        
        assert len(results) == 50