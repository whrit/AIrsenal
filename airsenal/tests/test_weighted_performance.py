"""
Tests for weighted performance metrics system.

This test suite validates the weighted performance calculation system including:
- Position-specific weight matrices
- Metric normalization and scoring
- Opponent difficulty adjustments  
- Performance benchmarking
- Configuration loading and validation
"""

import tempfile
import time
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import pytest

from airsenal.framework.weighted_performance import (
    PerformanceWeightMatrix,
    WeightedPerformanceCalculator,
    calculate_weighted_performance,
    calculate_batch_weighted_performance,
)
from airsenal.framework.weighted_performance_config import (
    create_weight_matrix_from_config,
    validate_weight_matrix,
    save_weight_matrix_to_yaml,
)
from airsenal.framework.schema import PlayerScore, FixtureDifficulty, Fixture, Player


class TestPerformanceWeightMatrix:
    """Test the PerformanceWeightMatrix configuration class."""
    
    def test_default_weights_initialization(self):
        """Test that default weights are properly initialized."""
        matrix = PerformanceWeightMatrix()
        
        # Check that all position weights exist
        assert 'clean_sheets' in matrix.goalkeeper_weights
        assert 'goals' in matrix.defender_weights
        assert 'assists' in matrix.midfielder_weights
        assert 'goals' in matrix.forward_weights
        
        # Check that difficulty adjustments exist for all levels
        for difficulty in range(1, 6):
            assert difficulty in matrix.difficulty_adjustments
        
        # Check normalization ranges exist for key metrics
        assert 'goals' in matrix.normalization_ranges
        assert 'assists' in matrix.normalization_ranges
    
    def test_get_weights_for_position(self):
        """Test getting weights for different positions."""
        matrix = PerformanceWeightMatrix()
        
        gk_weights = matrix.get_weights_for_position('GK')
        assert 'clean_sheets' in gk_weights
        assert 'saves' in gk_weights
        
        def_weights = matrix.get_weights_for_position('DEF')
        assert 'clean_sheets' in def_weights
        assert 'goals' in def_weights
        
        mid_weights = matrix.get_weights_for_position('MID')
        assert 'goals' in mid_weights
        assert 'assists' in mid_weights
        
        fwd_weights = matrix.get_weights_for_position('FWD')
        assert 'goals' in fwd_weights
        assert fwd_weights['goals'] > def_weights['goals']  # Forwards weight goals more
    
    def test_invalid_position_raises_error(self):
        """Test that invalid position raises ValueError."""
        matrix = PerformanceWeightMatrix()
        
        with pytest.raises(ValueError, match="Unknown position"):
            matrix.get_weights_for_position('INVALID')
    
    def test_normalize_metric(self):
        """Test metric normalization to 0-1 scale."""
        matrix = PerformanceWeightMatrix()
        
        # Test goals normalization (0-4 range)
        assert matrix.normalize_metric('goals', 0) == 0.0
        assert matrix.normalize_metric('goals', 2) == 0.5
        assert matrix.normalize_metric('goals', 4) == 1.0
        assert matrix.normalize_metric('goals', 6) == 1.0  # Clamped to max
        
        # Test clean sheets (binary metric)
        assert matrix.normalize_metric('clean_sheets', 0) == 0.0
        assert matrix.normalize_metric('clean_sheets', 1) == 1.0
        
        # Test own goals (negative metric - inverted scale)
        assert matrix.normalize_metric('own_goals', 0) == 1.0  # No own goal = good
        assert matrix.normalize_metric('own_goals', 1) == 0.0  # Own goal = bad
    
    def test_unknown_metric_normalization(self):
        """Test normalization of unknown metrics falls back to clamping."""
        matrix = PerformanceWeightMatrix()
        
        # Unknown metric should clamp to 0-1
        assert matrix.normalize_metric('unknown_metric', -1) == 0.0
        assert matrix.normalize_metric('unknown_metric', 0.5) == 0.5  
        assert matrix.normalize_metric('unknown_metric', 2) == 1.0


class TestWeightedPerformanceCalculator:
    """Test the main WeightedPerformanceCalculator class."""
    
    @pytest.fixture
    def mock_calculator(self):
        """Create calculator with mocked database session."""
        mock_session = Mock()
        calculator = WeightedPerformanceCalculator(dbsession=mock_session)
        return calculator, mock_session
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = WeightedPerformanceCalculator()
        assert calculator.weights is not None
        assert calculator._difficulty_cache == {}
        assert calculator._calculation_times == []
    
    def test_extract_metrics_from_score_goalkeeper(self, mock_calculator):
        """Test metric extraction for goalkeepers."""
        calculator, _ = mock_calculator
        
        # Create mock PlayerScore for goalkeeper
        score = Mock(spec=PlayerScore)
        score.goals = 1
        score.assists = 0  
        score.bonus = 2
        score.clean_sheets = 1
        score.saves = 5
        score.penalties_saved = 1
        score.minutes = 90
        
        metrics = calculator.extract_metrics_from_score(score, 'GK')
        
        assert metrics['goals'] == 1.0
        assert metrics['assists'] == 0.0
        assert metrics['bonus'] == 2.0
        assert metrics['clean_sheets'] == 1.0
        assert metrics['saves'] == 5.0
        assert metrics['penalties_saved'] == 1.0
    
    def test_extract_metrics_from_score_defender(self, mock_calculator):
        """Test metric extraction for defenders."""
        calculator, _ = mock_calculator
        
        score = Mock(spec=PlayerScore)
        score.goals = 1
        score.assists = 1
        score.bonus = 3
        score.clean_sheets = 1
        score.own_goals = 0
        
        metrics = calculator.extract_metrics_from_score(score, 'DEF')
        
        assert metrics['goals'] == 1.0
        assert metrics['assists'] == 1.0
        assert metrics['bonus'] == 3.0
        assert metrics['clean_sheets'] == 1.0
        assert metrics['own_goals'] == 0.0
    
    def test_extract_metrics_from_score_midfielder(self, mock_calculator):
        """Test metric extraction for midfielders."""
        calculator, _ = mock_calculator
        
        score = Mock(spec=PlayerScore)
        score.goals = 1
        score.assists = 2
        score.bonus = 1
        score.clean_sheets = 0
        score.player_id = 123
        score.minutes = 90
        
        with patch.object(calculator, '_get_penalty_taker_boost', return_value=1.0):
            metrics = calculator.extract_metrics_from_score(score, 'MID')
        
        assert metrics['goals'] == 1.0
        assert metrics['assists'] == 2.0
        assert metrics['bonus'] == 1.0
        assert metrics['clean_sheets'] == 0.0
        assert metrics['penalty_taker_boost'] == 1.0
    
    def test_extract_metrics_from_score_forward(self, mock_calculator):
        """Test metric extraction for forwards."""
        calculator, _ = mock_calculator
        
        score = Mock(spec=PlayerScore)
        score.goals = 2
        score.assists = 1
        score.bonus = 3
        score.player_id = 456
        score.minutes = 90
        
        with patch.object(calculator, '_get_penalty_taker_boost', return_value=0.0):
            metrics = calculator.extract_metrics_from_score(score, 'FWD')
        
        assert metrics['goals'] == 2.0
        assert metrics['assists'] == 1.0
        assert metrics['bonus'] == 3.0
        assert metrics['penalty_taker_boost'] == 0.0
    
    def test_get_fixture_difficulty_with_cache(self, mock_calculator):
        """Test fixture difficulty retrieval with caching."""
        calculator, mock_session = mock_calculator
        
        # Mock database query
        mock_difficulty = Mock()
        mock_difficulty.away_difficulty = 4.0
        mock_session.query.return_value.filter.return_value.first.return_value = mock_difficulty
        
        # First call should query database
        difficulty1 = calculator.get_fixture_difficulty(123)
        assert difficulty1 == 4.0
        assert 123 in calculator._difficulty_cache
        
        # Second call should use cache
        difficulty2 = calculator.get_fixture_difficulty(123)
        assert difficulty2 == 4.0
        # Should only have been called once due to caching
        assert mock_session.query.call_count == 1
    
    def test_get_fixture_difficulty_no_record(self, mock_calculator):
        """Test fixture difficulty when no record exists."""
        calculator, mock_session = mock_calculator
        
        # Mock no difficulty record found
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        difficulty = calculator.get_fixture_difficulty(999)
        assert difficulty == 3.0  # Default to average
    
    def test_calculate_weighted_score_basic(self, mock_calculator):
        """Test basic weighted score calculation."""
        calculator, _ = mock_calculator
        
        # Create simple test score
        score = Mock(spec=PlayerScore)
        score.goals = 2  # Will normalize to 0.5 (2/4)
        score.assists = 1  # Will normalize to 0.33 (1/3)  
        score.bonus = 0
        score.fixture_id = None  # No difficulty adjustment
        
        # Mock extract_metrics to return controlled values
        with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
            mock_extract.return_value = {'goals': 2.0, 'assists': 1.0}
            
            score_result = calculator.calculate_weighted_score(score, 'FWD', apply_difficulty_adjustment=False)
            
            # Should be weighted combination of normalized metrics
            assert 0.0 <= score_result <= 1.0
    
    def test_calculate_weighted_score_with_difficulty(self, mock_calculator):
        """Test weighted score calculation with difficulty adjustment."""
        calculator, _ = mock_calculator
        
        score = Mock(spec=PlayerScore)
        score.goals = 2
        score.assists = 1
        score.bonus = 0
        score.fixture_id = 123
        
        # Mock difficulty as easy fixture (factor 1.1)
        with patch.object(calculator, 'get_fixture_difficulty', return_value=2.0):
            with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
                mock_extract.return_value = {'goals': 2.0, 'assists': 1.0}
                
                score_no_diff = calculator.calculate_weighted_score(
                    score, 'FWD', apply_difficulty_adjustment=False
                )
                score_with_diff = calculator.calculate_weighted_score(
                    score, 'FWD', apply_difficulty_adjustment=True
                )
                
                # With easy fixture, score should be boosted
                assert score_with_diff > score_no_diff
    
    def test_calculate_batch_scores_empty(self, mock_calculator):
        """Test batch calculation with empty input."""
        calculator, _ = mock_calculator
        
        result = calculator.calculate_batch_scores([])
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_calculate_batch_scores_single_position(self, mock_calculator):
        """Test batch calculation with single position."""
        calculator, _ = mock_calculator
        
        # Create multiple mock scores
        scores = []
        for i in range(3):
            score = Mock(spec=PlayerScore)
            score.goals = i
            score.assists = 1
            score.bonus = 0
            score.fixture_id = None
            scores.append(score)
        
        scores_and_positions = [(score, 'FWD') for score in scores]
        
        with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
            mock_extract.side_effect = [
                {'goals': 0.0, 'assists': 1.0},
                {'goals': 1.0, 'assists': 1.0}, 
                {'goals': 2.0, 'assists': 1.0},
            ]
            
            results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
            
            assert len(results) == 3
            assert isinstance(results, np.ndarray)
            # All scores should be in valid range
            assert all(0.0 <= score <= 1.0 for score in results)
    
    def test_calculate_batch_scores_multiple_positions(self, mock_calculator):
        """Test batch calculation with multiple positions."""
        calculator, _ = mock_calculator
        
        # Create scores for different positions
        gk_score = Mock(spec=PlayerScore)
        gk_score.fixture_id = None
        
        fwd_score = Mock(spec=PlayerScore)
        fwd_score.fixture_id = None
        
        scores_and_positions = [
            (gk_score, 'GK'),
            (fwd_score, 'FWD'),
        ]
        
        with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
            mock_extract.side_effect = [
                {'clean_sheets': 1.0, 'saves': 5.0},
                {'goals': 2.0, 'assists': 0.0},
            ]
            
            results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
            
            assert len(results) == 2
            assert all(0.0 <= score <= 1.0 for score in results)
    
    def test_performance_stats_tracking(self, mock_calculator):
        """Test that calculation times are tracked for performance monitoring."""
        calculator, _ = mock_calculator
        
        # Perform some calculations to generate timing data
        scores_and_positions = []
        for i in range(5):
            score = Mock(spec=PlayerScore)
            score.fixture_id = None
            scores_and_positions.append((score, 'FWD'))
        
        with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
            mock_extract.return_value = {'goals': 1.0}
            
            calculator.calculate_batch_scores(scores_and_positions)
            calculator.calculate_batch_scores(scores_and_positions)
        
        stats = calculator.get_performance_stats()
        
        assert stats['count'] == 2
        assert 'mean_time_ms' in stats
        assert 'max_time_ms' in stats
        assert 'min_time_ms' in stats
        assert stats['mean_time_ms'] > 0
    
    def test_calculate_historical_performance_no_scores(self, mock_calculator):
        """Test historical performance calculation with no scores."""
        calculator, mock_session = mock_calculator
        
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        result = calculator.calculate_historical_performance(123, "2023-24", 5)
        
        assert result['count'] == 0
        assert result['mean_score'] == 0.0
        assert result['trend'] == 0.0
    
    def test_calculate_historical_performance_with_scores(self, mock_calculator):
        """Test historical performance calculation with actual scores.""" 
        calculator, mock_session = mock_calculator
        
        # Mock player query
        mock_player = Mock()
        mock_player.position.return_value = 'FWD'
        mock_session.query.return_value.filter.return_value.first.return_value = mock_player
        
        # Mock score queries
        mock_scores = [Mock() for _ in range(3)]
        for score in mock_scores:
            score.fixture_id = None
        
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_scores
        
        with patch.object(calculator, 'calculate_batch_scores') as mock_batch:
            mock_batch.return_value = np.array([0.8, 0.6, 0.7])
            
            result = calculator.calculate_historical_performance(123, "2023-24", 3)
            
            assert result['count'] == 3
            assert 0.0 <= result['mean_score'] <= 1.0
            assert 'trend' in result
            assert 'std_score' in result


class TestConvenienceFunctions:
    """Test convenience functions for weighted performance calculation."""
    
    def test_calculate_weighted_performance(self):
        """Test single score calculation convenience function."""
        score = Mock(spec=PlayerScore)
        score.goals = 1
        score.assists = 0
        score.bonus = 0
        score.fixture_id = None
        
        with patch('airsenal.framework.weighted_performance.default_calculator') as mock_calc:
            mock_calc.calculate_weighted_score.return_value = 0.5
            
            result = calculate_weighted_performance(score, 'FWD')
            
            assert result == 0.5
            mock_calc.calculate_weighted_score.assert_called_once_with(score, 'FWD')
    
    def test_calculate_batch_weighted_performance(self):
        """Test batch calculation convenience function."""
        scores_and_positions = [(Mock(), 'FWD'), (Mock(), 'DEF')]
        
        with patch('airsenal.framework.weighted_performance.default_calculator') as mock_calc:
            mock_calc.calculate_batch_scores.return_value = np.array([0.5, 0.6])
            
            result = calculate_batch_weighted_performance(scores_and_positions)
            
            assert len(result) == 2
            mock_calc.calculate_batch_scores.assert_called_once_with(scores_and_positions)


class TestConfigurationSystem:
    """Test the configuration loading and validation system."""
    
    def test_validate_weight_matrix_valid(self):
        """Test validation of a valid weight matrix."""
        matrix = PerformanceWeightMatrix()
        validation = validate_weight_matrix(matrix)
        
        assert validation['valid'] is True
        assert 'position_weight_sums' in validation
        assert len(validation['position_weight_sums']) == 4  # GK, DEF, MID, FWD
    
    def test_validate_weight_matrix_warnings(self):
        """Test validation warnings for unusual configurations."""
        # Create matrix with unusual weight sums
        matrix = PerformanceWeightMatrix()
        matrix.goalkeeper_weights = {'clean_sheets': 2.0}  # Too high
        
        validation = validate_weight_matrix(matrix)
        
        assert len(validation['warnings']) > 0
        assert any('sum to' in warning for warning in validation['warnings'])
    
    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration."""
        pytest.importorskip('yaml', reason="PyYAML required for configuration tests")
        
        matrix = PerformanceWeightMatrix()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to YAML
            save_weight_matrix_to_yaml(matrix, temp_path)
            
            # Load back from YAML  
            loaded_matrix = create_weight_matrix_from_config(temp_path)
            
            # Verify loaded matrix matches original
            assert loaded_matrix.goalkeeper_weights == matrix.goalkeeper_weights
            assert loaded_matrix.defender_weights == matrix.defender_weights
            assert loaded_matrix.midfielder_weights == matrix.midfielder_weights
            assert loaded_matrix.forward_weights == matrix.forward_weights
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPerformanceBenchmarking:
    """Test performance benchmarking to ensure sub-50ms target."""
    
    @pytest.mark.performance
    def test_batch_calculation_performance(self):
        """Test that batch calculations meet the <50ms performance target."""
        calculator = WeightedPerformanceCalculator()
        
        # Create large batch of mock scores (600+ as per requirement)
        num_scores = 650
        scores_and_positions = []
        
        for i in range(num_scores):
            score = Mock(spec=PlayerScore)
            score.goals = i % 4
            score.assists = i % 3
            score.bonus = i % 4
            score.clean_sheets = i % 2
            score.saves = i % 10
            score.own_goals = 0
            score.penalties_saved = 0
            score.fixture_id = None
            
            position = ['GK', 'DEF', 'MID', 'FWD'][i % 4]
            scores_and_positions.append((score, position))
        
        # Mock the extract_metrics method to avoid complex setup
        def mock_extract_metrics(score, position):
            if position == 'GK':
                return {
                    'goals': float(score.goals),
                    'assists': float(score.assists),
                    'bonus': float(score.bonus),
                    'clean_sheets': float(score.clean_sheets),
                    'saves': float(score.saves),
                    'penalties_saved': float(score.penalties_saved),
                }
            elif position == 'DEF':
                return {
                    'goals': float(score.goals),
                    'assists': float(score.assists), 
                    'bonus': float(score.bonus),
                    'clean_sheets': float(score.clean_sheets),
                    'own_goals': float(score.own_goals),
                }
            elif position == 'MID':
                return {
                    'goals': float(score.goals),
                    'assists': float(score.assists),
                    'bonus': float(score.bonus),
                    'clean_sheets': float(score.clean_sheets),
                    'key_passes_per_90': 2.0,
                    'penalty_taker_boost': 0.0,
                }
            else:  # FWD
                return {
                    'goals': float(score.goals),
                    'assists': float(score.assists),
                    'bonus': float(score.bonus),
                    'shots_per_90': 3.0,
                    'penalty_taker_boost': 0.0,
                }
        
        # Time the calculation
        start_time = time.perf_counter()
        
        with patch.object(calculator, 'extract_metrics_from_score', side_effect=mock_extract_metrics):
            results = calculator.calculate_batch_scores(scores_and_positions, apply_difficulty_adjustment=False)
        
        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000
        
        # Verify results are reasonable
        assert len(results) == num_scores
        assert all(0.0 <= score <= 1.0 for score in results)
        
        # Check performance requirement
        print(f"Batch calculation time for {num_scores} scores: {calculation_time_ms:.2f}ms")
        assert calculation_time_ms < 50.0, f"Calculation took {calculation_time_ms:.2f}ms, exceeds 50ms target"
    
    @pytest.mark.performance
    def test_individual_calculation_performance(self):
        """Test performance of individual score calculations."""
        calculator = WeightedPerformanceCalculator()
        
        score = Mock(spec=PlayerScore)
        score.goals = 2
        score.assists = 1
        score.bonus = 3
        score.fixture_id = None
        
        # Mock extract_metrics
        with patch.object(calculator, 'extract_metrics_from_score') as mock_extract:
            mock_extract.return_value = {'goals': 2.0, 'assists': 1.0, 'bonus': 3.0}
            
            # Time multiple individual calculations
            start_time = time.perf_counter()
            
            for _ in range(1000):
                result = calculator.calculate_weighted_score(score, 'FWD', apply_difficulty_adjustment=False)
                assert 0.0 <= result <= 1.0
            
            end_time = time.perf_counter()
            avg_time_ms = ((end_time - start_time) / 1000) * 1000
            
            print(f"Average individual calculation time: {avg_time_ms:.4f}ms")
            assert avg_time_ms < 1.0, f"Individual calculation averaging {avg_time_ms:.4f}ms is too slow"


if __name__ == "__main__":
    # Run the performance tests
    pytest.main([__file__ + "::TestPerformanceBenchmarking", "-v"])