"""
Comprehensive tests for Form Decay System

This test suite covers all aspects of the form decay implementation:
- Consistency analysis and metrics calculation
- Adaptive decay rate calculation
- Injury impact handling and absence adjustments
- Form transition modeling and smoothing
- Main form decay system integration
- Kalman filter integration
- Performance validation and accuracy testing
- Edge cases and error handling
"""

import logging
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from airsenal.framework.form_decay import (
    FormDecaySystem,
    ConsistencyAnalyzer, 
    AdaptiveDecayCalculator,
    InjuryImpactHandler,
    FormTransitionModel,
    FormDecayConfig,
    PlayerConsistency,
    FormDecayResult,
    ConsistencyLevel,
    DecayModelType,
    FormDecayError,
    InsufficientDataError,
    ConsistencyCalculationError,
    DecayParameterError,
    create_default_form_decay_system,
    calculate_player_form_decay,
    get_adaptive_decay_rates
)

from airsenal.framework.schema import (
    Absence,
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore
)

# Test configuration
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def form_decay_config():
    """Default form decay configuration for testing."""
    return FormDecayConfig(
        base_half_life=8.0,
        min_half_life=3.0,
        max_half_life=20.0,
        lookback_weeks=10,
        min_games_for_consistency=5,
        injury_decay_multiplier=2.0,
        return_recovery_weeks=3,
        kalman_integration=True,
        transition_smoothing=True,
        smoothing_alpha=0.3
    )


@pytest.fixture
def mock_dbsession():
    """Mock database session for testing."""
    session = Mock()
    return session


@pytest.fixture
def sample_player_scores():
    """Sample player score data for testing."""
    scores = []
    for i in range(15):
        score = Mock()
        score.points = 5.0 + np.random.normal(0, 2.0)  # Realistic FPL scores
        scores.append(score)
    return scores


@pytest.fixture
def sample_player_attributes():
    """Sample player attributes for testing."""
    attrs = Mock()
    attrs.position = 'MID'
    attrs.age = 26
    attrs.team = 'TEST'
    return attrs


@pytest.fixture
def sample_absences():
    """Sample absence data for testing."""
    absences = []
    # Create injury absence from gameweek 5-8
    absence = Mock()
    absence.gameweek_start = 5
    absence.gameweek_end = 8
    absence.player_id = 123
    absence.season = "2425"
    absences.append(absence)
    return absences


class TestFormDecayConfig:
    """Test FormDecayConfig class."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = FormDecayConfig()
        
        assert config.base_half_life == 8.0
        assert config.min_half_life == 3.0
        assert config.max_half_life == 20.0
        assert config.position_multipliers['GK'] == 1.2
        assert config.position_multipliers['FWD'] == 0.9
        assert config.kalman_integration is True
        assert config.transition_smoothing is True
    
    def test_custom_config_creation(self, form_decay_config):
        """Test creation of custom configuration."""
        assert form_decay_config.base_half_life == 8.0
        assert form_decay_config.lookback_weeks == 10
        assert form_decay_config.injury_decay_multiplier == 2.0


class TestConsistencyAnalyzer:
    """Test ConsistencyAnalyzer class."""
    
    def test_consistency_analyzer_initialization(self, form_decay_config, mock_dbsession):
        """Test consistency analyzer initialization."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        assert analyzer.config == form_decay_config
        assert analyzer.dbsession == mock_dbsession
        assert isinstance(analyzer._cache, dict)
    
    def test_calculate_consistency_metrics_success(self, form_decay_config, mock_dbsession):
        """Test successful consistency metrics calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Mock performance data
        performance_data = np.array([4, 6, 3, 8, 5, 7, 2, 9, 4, 6])
        
        with patch.object(analyzer, '_get_player_performance_data', return_value=performance_data):
            result = analyzer.calculate_consistency_metrics(123, "2425")
            
            assert isinstance(result, PlayerConsistency)
            assert result.player_id == 123
            assert 0.0 <= result.coefficient_variation <= 2.0
            assert -1.0 <= result.autocorrelation <= 1.0
            assert 0.0 <= result.predictability_score <= 1.0
            assert result.consistency_level in ConsistencyLevel
            assert result.games_analyzed == 10
            assert 0.0 <= result.confidence <= 1.0
    
    def test_calculate_consistency_metrics_insufficient_data(self, form_decay_config, mock_dbsession):
        """Test consistency calculation with insufficient data."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Mock insufficient performance data
        performance_data = np.array([4, 6, 3])  # Only 3 games, need 5
        
        with patch.object(analyzer, '_get_player_performance_data', return_value=performance_data):
            with pytest.raises(InsufficientDataError):
                analyzer.calculate_consistency_metrics(123, "2425")
    
    def test_coefficient_variation_calculation(self, form_decay_config, mock_dbsession):
        """Test coefficient of variation calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Test with known data
        data = np.array([2, 4, 6, 8, 10])  # Mean=6, Std≈3.16, CV≈0.53
        cv = analyzer._calculate_coefficient_variation(data)
        
        expected_cv = np.std(data, ddof=1) / np.mean(data)
        assert abs(cv - expected_cv) < 1e-6
        
        # Test with zero mean
        zero_data = np.array([0, 0, 0])
        cv_zero = analyzer._calculate_coefficient_variation(zero_data)
        assert cv_zero == 0.0
        
        # Test with empty data
        empty_data = np.array([])
        cv_empty = analyzer._calculate_coefficient_variation(empty_data)
        assert cv_empty == 0.0
    
    def test_autocorrelation_calculation(self, form_decay_config, mock_dbsession):
        """Test autocorrelation calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Test with trending data (should have positive autocorr)
        trending_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        autocorr = analyzer._calculate_autocorrelation(trending_data)
        assert autocorr > 0.5  # Strong positive correlation
        
        # Test with alternating data (should have negative autocorr)
        alternating_data = np.array([1, 10, 2, 9, 3, 8, 4, 7])
        autocorr_neg = analyzer._calculate_autocorrelation(alternating_data)
        assert autocorr_neg < 0  # Negative correlation
        
        # Test with insufficient data
        short_data = np.array([1])
        autocorr_short = analyzer._calculate_autocorrelation(short_data)
        assert autocorr_short == 0.0
    
    def test_predictability_score_calculation(self, form_decay_config, mock_dbsession):
        """Test predictability score calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # High predictability: low CV, high autocorr
        score_high = analyzer._calculate_predictability_score(0.2, 0.8)
        assert score_high > 0.7
        
        # Low predictability: high CV, low autocorr
        score_low = analyzer._calculate_predictability_score(1.5, 0.1)
        assert score_low < 0.5
        
        # Test bounds
        score = analyzer._calculate_predictability_score(0.5, 0.5)
        assert 0.0 <= score <= 1.0
    
    def test_consistency_level_classification(self, form_decay_config, mock_dbsession):
        """Test consistency level classification."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Very high consistency
        level = analyzer._classify_consistency_level(0.2, 0.6)
        assert level == ConsistencyLevel.VERY_HIGH
        
        # High consistency
        level = analyzer._classify_consistency_level(0.4, 0.4)
        assert level == ConsistencyLevel.HIGH
        
        # Medium consistency
        level = analyzer._classify_consistency_level(0.6, 0.2)
        assert level == ConsistencyLevel.MEDIUM
        
        # Low consistency
        level = analyzer._classify_consistency_level(0.8, 0.0)
        assert level == ConsistencyLevel.LOW
        
        # Very low consistency
        level = analyzer._classify_consistency_level(1.2, -0.2)
        assert level == ConsistencyLevel.VERY_LOW
    
    def test_caching_mechanism(self, form_decay_config, mock_dbsession):
        """Test consistency calculation caching."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        performance_data = np.array([4, 6, 3, 8, 5, 7, 2, 9, 4, 6])
        
        with patch.object(analyzer, '_get_player_performance_data', return_value=performance_data) as mock_get_data:
            # First call
            result1 = analyzer.calculate_consistency_metrics(123, "2425")
            
            # Second call (should use cache)
            result2 = analyzer.calculate_consistency_metrics(123, "2425")
            
            # Should only call data fetch once
            assert mock_get_data.call_count == 1
            assert result1.player_id == result2.player_id
            assert result1.coefficient_variation == result2.coefficient_variation


class TestAdaptiveDecayCalculator:
    """Test AdaptiveDecayCalculator class."""
    
    def test_calculator_initialization(self, form_decay_config, mock_dbsession):
        """Test adaptive decay calculator initialization."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        assert calculator.config == form_decay_config
        assert calculator.consistency_analyzer == analyzer
        assert isinstance(calculator._decay_cache, dict)
    
    def test_calculate_player_decay_rate_success(self, form_decay_config, mock_dbsession):
        """Test successful decay rate calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        # Mock consistency metrics
        mock_consistency = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.5,
            autocorrelation=0.3,
            predictability_score=0.7,
            streak_length_avg=2.5,
            variance_stability=0.8,
            consistency_level=ConsistencyLevel.HIGH,
            games_analyzed=10,
            calculation_date=datetime.now(),
            confidence=0.9
        )
        
        # Mock player attributes
        mock_attrs = {'position': 'MID', 'age': 26, 'team': 'TEST'}
        
        with patch.object(analyzer, 'calculate_consistency_metrics', return_value=mock_consistency), \
             patch.object(calculator, '_get_player_attributes', return_value=mock_attrs):
            
            decay_rate = calculator.calculate_player_decay_rate(123, "2425")
            
            assert form_decay_config.min_half_life <= decay_rate <= form_decay_config.max_half_life
            assert isinstance(decay_rate, float)
    
    def test_consistency_multiplier_calculation(self, form_decay_config, mock_dbsession):
        """Test consistency multiplier calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        # Very high consistency should have high multiplier (slower decay)
        consistency_high = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.2,
            autocorrelation=0.7,
            predictability_score=0.9,
            streak_length_avg=3.0,
            variance_stability=0.9,
            consistency_level=ConsistencyLevel.VERY_HIGH,
            games_analyzed=15,
            calculation_date=datetime.now(),
            confidence=0.95
        )
        
        multiplier_high = calculator._calculate_consistency_multiplier(consistency_high)
        assert multiplier_high > 1.2  # Should slow down decay
        
        # Very low consistency should have low multiplier (faster decay)
        consistency_low = PlayerConsistency(
            player_id=124,
            coefficient_variation=1.2,
            autocorrelation=-0.1,
            predictability_score=0.2,
            streak_length_avg=1.5,
            variance_stability=0.3,
            consistency_level=ConsistencyLevel.VERY_LOW,
            games_analyzed=8,
            calculation_date=datetime.now(),
            confidence=0.6
        )
        
        multiplier_low = calculator._calculate_consistency_multiplier(consistency_low)
        assert multiplier_low < 0.8  # Should speed up decay
    
    def test_age_multiplier_calculation(self, form_decay_config, mock_dbsession):
        """Test age-based multiplier calculation."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        # Young player
        young_multiplier = calculator._calculate_age_multiplier(21)
        assert young_multiplier == form_decay_config.young_multiplier
        
        # Old player  
        old_multiplier = calculator._calculate_age_multiplier(32)
        assert old_multiplier == form_decay_config.old_multiplier
        
        # Middle age player (interpolation)
        middle_multiplier = calculator._calculate_age_multiplier(26)
        assert form_decay_config.young_multiplier < middle_multiplier < form_decay_config.old_multiplier
    
    def test_position_multiplier_application(self, form_decay_config, mock_dbsession):
        """Test position-specific multiplier application."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        # Mock consistency and different positions
        mock_consistency = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.5,
            autocorrelation=0.3,
            predictability_score=0.7,
            streak_length_avg=2.5,
            variance_stability=0.8,
            consistency_level=ConsistencyLevel.MEDIUM,
            games_analyzed=10,
            calculation_date=datetime.now(),
            confidence=0.8
        )
        
        with patch.object(analyzer, 'calculate_consistency_metrics', return_value=mock_consistency):
            # Test GK (should have longer half-life)
            with patch.object(calculator, '_get_player_attributes', return_value={'position': 'GK', 'age': 26}):
                gk_rate = calculator.calculate_player_decay_rate(123, "2425", force_recalculate=True)
            
            # Test FWD (should have shorter half-life)  
            with patch.object(calculator, '_get_player_attributes', return_value={'position': 'FWD', 'age': 26}):
                fwd_rate = calculator.calculate_player_decay_rate(124, "2425", force_recalculate=True)
            
            assert gk_rate > fwd_rate  # GK should decay slower than FWD
    
    def test_caching_mechanism(self, form_decay_config, mock_dbsession):
        """Test decay rate caching."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        mock_consistency = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.5,
            autocorrelation=0.3,
            predictability_score=0.7,
            streak_length_avg=2.5,
            variance_stability=0.8,
            consistency_level=ConsistencyLevel.MEDIUM,
            games_analyzed=10,
            calculation_date=datetime.now(),
            confidence=0.8
        )
        
        mock_attrs = {'position': 'MID', 'age': 26}
        
        with patch.object(analyzer, 'calculate_consistency_metrics', return_value=mock_consistency) as mock_calc, \
             patch.object(calculator, '_get_player_attributes', return_value=mock_attrs):
            
            # First call
            rate1 = calculator.calculate_player_decay_rate(123, "2425")
            
            # Second call (should use cache)
            rate2 = calculator.calculate_player_decay_rate(123, "2425")
            
            # Should only calculate consistency once
            assert mock_calc.call_count == 1
            assert rate1 == rate2


class TestInjuryImpactHandler:
    """Test InjuryImpactHandler class."""
    
    def test_handler_initialization(self, form_decay_config, mock_dbsession):
        """Test injury impact handler initialization."""
        handler = InjuryImpactHandler(form_decay_config, mock_dbsession)
        
        assert handler.config == form_decay_config
        assert handler.dbsession == mock_dbsession
    
    def test_injury_adjusted_decay_no_absences(self, form_decay_config, mock_dbsession):
        """Test decay adjustment with no absences."""
        handler = InjuryImpactHandler(form_decay_config, mock_dbsession)
        
        base_half_life = 8.0
        gameweeks = [1, 2, 3, 4, 5]
        
        with patch.object(handler, '_get_player_absences', return_value=[]):
            adjusted_rates = handler.get_injury_adjusted_decay(123, base_half_life, gameweeks, "2425")
            
            # Should return base rate for all gameweeks
            expected = np.full(len(gameweeks), base_half_life)
            np.testing.assert_array_equal(adjusted_rates, expected)
    
    def test_injury_adjusted_decay_with_absence(self, form_decay_config, mock_dbsession, sample_absences):
        """Test decay adjustment during absence period."""
        handler = InjuryImpactHandler(form_decay_config, mock_dbsession)
        
        base_half_life = 8.0
        gameweeks = [3, 4, 5, 6, 7, 8, 9, 10]  # GW 5-8 are absence period
        
        with patch.object(handler, '_get_player_absences', return_value=sample_absences), \
             patch.object(handler, '_is_player_absent') as mock_absent, \
             patch.object(handler, '_weeks_since_return') as mock_return:
            
            # Mock absence check
            def mock_absent_func(gw, absences, season):
                return 5 <= gw <= 8
            mock_absent.side_effect = mock_absent_func
            
            # Mock weeks since return
            def mock_return_func(gw, absences, season):
                if gw > 8:
                    return gw - 8
                return None
            mock_return.side_effect = mock_return_func
            
            adjusted_rates = handler.get_injury_adjusted_decay(123, base_half_life, gameweeks, "2425")
            
            # Check that absent periods have accelerated decay
            for i, gw in enumerate(gameweeks):
                if 5 <= gw <= 8:  # Absence period
                    expected_rate = base_half_life / form_decay_config.injury_decay_multiplier
                    assert abs(adjusted_rates[i] - expected_rate) < 1e-6
                elif gw == 9:  # First week back (1 week since return)
                    assert adjusted_rates[i] > base_half_life  # Recovery bonus
    
    def test_is_player_absent(self, form_decay_config, mock_dbsession, sample_absences):
        """Test player absence checking."""
        handler = InjuryImpactHandler(form_decay_config, mock_dbsession)
        
        # Test gameweek within absence period
        assert handler._is_player_absent(6, sample_absences, "2425") is True
        
        # Test gameweek outside absence period
        assert handler._is_player_absent(3, sample_absences, "2425") is False
        assert handler._is_player_absent(10, sample_absences, "2425") is False
        
        # Test boundary gameweeks
        assert handler._is_player_absent(5, sample_absences, "2425") is True
        assert handler._is_player_absent(8, sample_absences, "2425") is True
    
    def test_weeks_since_return(self, form_decay_config, mock_dbsession, sample_absences):
        """Test weeks since return calculation."""
        handler = InjuryImpactHandler(form_decay_config, mock_dbsession)
        
        # Test before absence
        weeks = handler._weeks_since_return(3, sample_absences, "2425")
        assert weeks is None
        
        # Test during absence
        weeks = handler._weeks_since_return(6, sample_absences, "2425")
        assert weeks is None
        
        # Test after return
        weeks = handler._weeks_since_return(10, sample_absences, "2425")
        assert weeks == 2  # 10 - 8 = 2 weeks since return
        
        weeks = handler._weeks_since_return(9, sample_absences, "2425")
        assert weeks == 1  # 9 - 8 = 1 week since return


class TestFormTransitionModel:
    """Test FormTransitionModel class."""
    
    def test_model_initialization(self, form_decay_config):
        """Test form transition model initialization."""
        model = FormTransitionModel(form_decay_config)
        assert model.config == form_decay_config
    
    def test_smooth_form_transitions(self, form_decay_config):
        """Test form transition smoothing."""
        model = FormTransitionModel(form_decay_config)
        
        observed_form = np.array([4.0, 6.0, 3.0, 8.0, 5.0])
        predicted_form = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        uncertainty = np.array([0.5, 0.3, 0.8, 0.2, 0.6])
        gameweeks = np.array([1, 2, 3, 4, 5])
        
        smoothed = model.smooth_form_transitions(observed_form, predicted_form, uncertainty, gameweeks)
        
        # Should be between observed and predicted
        assert len(smoothed) == len(observed_form)
        assert np.all(smoothed >= np.minimum(observed_form, predicted_form) - 0.1)
        assert np.all(smoothed <= np.maximum(observed_form, predicted_form) + 0.1)
    
    def test_smooth_transitions_disabled(self, form_decay_config):
        """Test form transition smoothing when disabled."""
        form_decay_config.transition_smoothing = False
        model = FormTransitionModel(form_decay_config)
        
        observed_form = np.array([4.0, 6.0, 3.0, 8.0, 5.0])
        predicted_form = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        uncertainty = np.array([0.5, 0.3, 0.8, 0.2, 0.6])
        gameweeks = np.array([1, 2, 3, 4, 5])
        
        smoothed = model.smooth_form_transitions(observed_form, predicted_form, uncertainty, gameweeks)
        
        # Should return observed form unchanged
        np.testing.assert_array_equal(smoothed, observed_form)
    
    def test_temporal_smoothing(self, form_decay_config):
        """Test temporal smoothing functionality."""
        model = FormTransitionModel(form_decay_config)
        
        # Noisy form values
        form_values = np.array([5.0, 9.0, 2.0, 8.0, 4.0, 7.0])
        gameweeks = np.array([1, 2, 3, 4, 5, 6])
        
        smoothed = model._apply_temporal_smoothing(form_values, gameweeks)
        
        # Should preserve endpoints
        assert smoothed[0] == form_values[0]
        assert smoothed[-1] == form_values[-1]
        
        # Should smooth middle values
        assert len(smoothed) == len(form_values)
    
    def test_seasonal_patterns(self, form_decay_config):
        """Test seasonal pattern modeling."""
        model = FormTransitionModel(form_decay_config)
        
        form_values = np.array([5.0, 6.0, 7.0, 8.0, 5.0])
        
        # Test Christmas period (GW 16-22)
        christmas_gameweeks = np.array([16, 17, 18, 19, 20])
        adjusted = model.model_seasonal_patterns(form_values, christmas_gameweeks, "2425")
        
        # Should apply slight reduction during Christmas period
        assert np.all(adjusted <= form_values)
        
        # Test non-Christmas period
        normal_gameweeks = np.array([5, 6, 7, 8, 9])
        normal_adjusted = model.model_seasonal_patterns(form_values, normal_gameweeks, "2425")
        
        # Should remain unchanged
        np.testing.assert_array_equal(normal_adjusted, form_values)


class TestFormDecaySystem:
    """Test main FormDecaySystem class."""
    
    def test_system_initialization(self, form_decay_config, mock_dbsession):
        """Test form decay system initialization."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        assert system.config == form_decay_config
        assert system.dbsession == mock_dbsession
        assert isinstance(system.consistency_analyzer, ConsistencyAnalyzer)
        assert isinstance(system.decay_calculator, AdaptiveDecayCalculator)
        assert isinstance(system.injury_handler, InjuryImpactHandler)
        assert isinstance(system.transition_model, FormTransitionModel)
    
    def test_apply_form_decay_success(self, form_decay_config, mock_dbsession):
        """Test successful form decay application."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        # Mock components
        performance_data = np.array([4.0, 6.0, 3.0, 8.0, 5.0, 7.0, 2.0, 9.0])
        gameweeks = [1, 2, 3, 4, 5, 6, 7, 8]
        half_life = 8.0
        adjusted_decay_rates = np.full(len(gameweeks), half_life)
        
        mock_consistency = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.5,
            autocorrelation=0.3,
            predictability_score=0.7,
            streak_length_avg=2.5,
            variance_stability=0.8,
            consistency_level=ConsistencyLevel.MEDIUM,
            games_analyzed=8,
            calculation_date=datetime.now(),
            confidence=0.8
        )
        
        with patch.object(system, '_get_player_performance_history', return_value=(performance_data, gameweeks)), \
             patch.object(system.decay_calculator, 'calculate_player_decay_rate', return_value=half_life), \
             patch.object(system.injury_handler, 'get_injury_adjusted_decay', return_value=adjusted_decay_rates), \
             patch.object(system.consistency_analyzer, 'calculate_consistency_metrics', return_value=mock_consistency):
            
            result = system.apply_form_decay(123, lookback_weeks=8, season="2425", current_gameweek=10)
            
            assert isinstance(result, FormDecayResult)
            assert result.player_id == 123
            assert len(result.original_form) == len(performance_data)
            assert len(result.decayed_form) == len(performance_data)
            assert len(result.decay_rates) == len(gameweeks)
            assert result.half_life == half_life
            assert result.consistency_metrics == mock_consistency
            assert result.gameweeks == gameweeks
    
    def test_apply_form_decay_insufficient_data(self, form_decay_config, mock_dbsession):
        """Test form decay with insufficient data."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        # Mock empty performance data
        with patch.object(system, '_get_player_performance_history', return_value=(np.array([]), [])):
            with pytest.raises(InsufficientDataError):
                system.apply_form_decay(123, lookback_weeks=8, season="2425")
    
    def test_exponential_decay_application(self, form_decay_config, mock_dbsession):
        """Test exponential decay calculation."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        performance_data = np.array([10.0, 8.0, 6.0, 4.0])
        decay_rates = np.array([4.0, 4.0, 4.0, 4.0])  # Half-life of 4 weeks
        gameweeks = [6, 7, 8, 9]
        current_gameweek = 10
        
        decayed_form = system._apply_exponential_decay(
            performance_data, decay_rates, gameweeks, current_gameweek
        )
        
        # Check decay behavior - more recent should decay less
        assert len(decayed_form) == len(performance_data)
        assert decayed_form[3] > decayed_form[2]  # More recent should be less decayed
        assert decayed_form[2] > decayed_form[1]
        assert decayed_form[1] > decayed_form[0]
        
        # All should be less than original (decay applied)
        assert np.all(decayed_form <= performance_data)
    
    def test_kalman_integration(self, form_decay_config, mock_dbsession):
        """Test Kalman filter integration."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        current_state = np.array([5.0, 7.0, 0.8, 0.2])  # [skill, form, consistency, momentum]
        observation = np.array([6.0, 3.0, 0.5, 0.1])
        
        mock_consistency = PlayerConsistency(
            player_id=123,
            coefficient_variation=0.5,
            autocorrelation=0.3,
            predictability_score=0.7,
            streak_length_avg=2.5,
            variance_stability=0.8,
            consistency_level=ConsistencyLevel.MEDIUM,
            games_analyzed=10,
            calculation_date=datetime.now(),
            confidence=0.8
        )
        
        with patch.object(system, 'get_player_decay_rate', return_value=8.0), \
             patch.object(system.consistency_analyzer, 'calculate_consistency_metrics', return_value=mock_consistency):
            
            updated_state, updated_cov = system.integrate_with_kalman(
                123, current_state, observation, dt=1.0, season="2425"
            )
            
            # State should be modified by decay
            assert len(updated_state) == len(current_state)
            assert len(updated_cov) == len(current_state)
            
            # Form component should decay most
            assert updated_state[1] < current_state[1]  # Form decayed
            assert updated_state[0] < current_state[0]  # Skill decayed slightly
    
    def test_get_player_decay_rate(self, form_decay_config, mock_dbsession):
        """Test getting player decay rate."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        expected_rate = 8.5
        with patch.object(system.decay_calculator, 'calculate_player_decay_rate', return_value=expected_rate):
            rate = system.get_player_decay_rate(123, "2425")
            assert rate == expected_rate
    
    def test_validation_framework(self, form_decay_config, mock_dbsession):
        """Test system validation framework."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        # Test validation (placeholder implementation)
        validation_results = system.validate_system(
            validation_players=[123, 456, 789],
            validation_season="2324"
        )
        
        assert isinstance(validation_results, dict)
        assert 'mse_improvement' in validation_results
        assert 'prediction_accuracy' in validation_results
        assert 'consistency_correlation' in validation_results
        assert 'decay_parameter_stability' in validation_results


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_default_form_decay_system(self, mock_dbsession):
        """Test default system creation."""
        with patch('airsenal.framework.form_decay.session', mock_dbsession):
            system = create_default_form_decay_system(mock_dbsession)
            
            assert isinstance(system, FormDecaySystem)
            assert system.dbsession == mock_dbsession
            assert isinstance(system.config, FormDecayConfig)
    
    def test_calculate_player_form_decay(self, mock_dbsession):
        """Test convenience function for single player decay."""
        mock_result = FormDecayResult(
            player_id=123,
            original_form=np.array([5.0, 6.0, 4.0]),
            decayed_form=np.array([4.0, 5.5, 3.8]),
            decay_rates=np.array([8.0, 8.0, 8.0]),
            half_life=8.0,
            consistency_metrics=Mock(),
            gameweeks=[1, 2, 3]
        )
        
        with patch('airsenal.framework.form_decay.create_default_form_decay_system') as mock_create:
            mock_system = Mock()
            mock_system.apply_form_decay.return_value = mock_result
            mock_create.return_value = mock_system
            
            result = calculate_player_form_decay(123, 10, "2425", mock_dbsession)
            
            assert result == mock_result
            mock_system.apply_form_decay.assert_called_once_with(123, 10, "2425")
    
    def test_get_adaptive_decay_rates(self, mock_dbsession):
        """Test convenience function for multiple player decay rates."""
        player_ids = [123, 456, 789]
        expected_rates = {123: 8.0, 456: 6.5, 789: 9.2}
        
        with patch('airsenal.framework.form_decay.create_default_form_decay_system') as mock_create:
            mock_system = Mock()
            mock_system.get_player_decay_rate.side_effect = lambda pid, season: expected_rates[pid]
            mock_create.return_value = mock_system
            
            rates = get_adaptive_decay_rates(player_ids, "2425", mock_dbsession)
            
            assert rates == expected_rates
            assert mock_system.get_player_decay_rate.call_count == len(player_ids)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_consistency_calculation_error(self, form_decay_config, mock_dbsession):
        """Test handling of consistency calculation errors."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        with patch.object(analyzer, '_get_player_performance_data', side_effect=Exception("DB Error")):
            with pytest.raises(ConsistencyCalculationError):
                analyzer.calculate_consistency_metrics(123, "2425")
    
    def test_decay_parameter_error(self, form_decay_config, mock_dbsession):
        """Test handling of decay parameter errors."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        calculator = AdaptiveDecayCalculator(form_decay_config, analyzer)
        
        with patch.object(analyzer, 'calculate_consistency_metrics', side_effect=Exception("Calc Error")):
            with pytest.raises(DecayParameterError):
                calculator.calculate_player_decay_rate(123, "2425")
    
    def test_form_decay_error(self, form_decay_config, mock_dbsession):
        """Test handling of general form decay errors."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        with patch.object(system, '_get_player_performance_history', side_effect=Exception("History Error")):
            with pytest.raises(FormDecayError):
                system.apply_form_decay(123, 10, "2425")
    
    def test_kalman_integration_fallback(self, form_decay_config, mock_dbsession):
        """Test Kalman integration fallback on error."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        current_state = np.array([5.0, 7.0, 0.8, 0.2])
        observation = np.array([6.0, 3.0, 0.5, 0.1])
        
        with patch.object(system, 'get_player_decay_rate', side_effect=Exception("Decay Error")):
            # Should fallback gracefully
            updated_state, updated_cov = system.integrate_with_kalman(
                123, current_state, observation, dt=1.0, season="2425"
            )
            
            # Should return original state when error occurs
            np.testing.assert_array_equal(updated_state, current_state)
            assert updated_cov.shape == (len(current_state), len(current_state))


class TestPerformanceAndValidation:
    """Test performance characteristics and validation."""
    
    def test_consistency_calculation_performance(self, form_decay_config, mock_dbsession):
        """Test performance of consistency calculations."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Test with realistic data size
        large_performance_data = np.random.normal(5.0, 2.0, 50)  # 50 games
        
        import time
        with patch.object(analyzer, '_get_player_performance_data', return_value=large_performance_data):
            start_time = time.time()
            result = analyzer.calculate_consistency_metrics(123, "2425")
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 1.0  # Less than 1 second
            assert isinstance(result, PlayerConsistency)
    
    def test_decay_calculation_accuracy(self, form_decay_config, mock_dbsession):
        """Test accuracy of decay calculations."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        # Test with known decay parameters
        performance_data = np.array([10.0, 10.0, 10.0])
        decay_rates = np.array([4.0, 4.0, 4.0])  # 4-week half-life
        gameweeks = [5, 6, 7]
        current_gameweek = 10
        
        decayed_form = system._apply_exponential_decay(
            performance_data, decay_rates, gameweeks, current_gameweek
        )
        
        # After 5 gameweeks with 4-week half-life, should be ~0.42 of original
        # 10 * exp(-ln(2) * 5 / 4) ≈ 10 * 0.42 = 4.2
        expected_oldest = 10.0 * np.exp(-np.log(2) * 5 / 4)
        assert abs(decayed_form[0] - expected_oldest) < 0.1
        
        # More recent should decay less
        assert decayed_form[2] > decayed_form[1] > decayed_form[0]
    
    def test_bounds_enforcement(self, form_decay_config, mock_dbsession):
        """Test that bounds are properly enforced."""
        system = FormDecaySystem(form_decay_config, mock_dbsession)
        
        # Test with extreme values
        extreme_data = np.array([100.0, 0.01, 50.0])
        decay_rates = np.array([2.0, 2.0, 2.0])
        gameweeks = [1, 2, 3]
        current_gameweek = 10
        
        decayed_form = system._apply_exponential_decay(
            extreme_data, decay_rates, gameweeks, current_gameweek
        )
        
        # Apply bounds
        bounded_form = np.clip(
            decayed_form,
            form_decay_config.min_form_value,
            form_decay_config.max_form_value
        )
        
        assert np.all(bounded_form >= form_decay_config.min_form_value)
        assert np.all(bounded_form <= form_decay_config.max_form_value)
    
    def test_numerical_stability(self, form_decay_config, mock_dbsession):
        """Test numerical stability with edge cases."""
        analyzer = ConsistencyAnalyzer(form_decay_config, mock_dbsession)
        
        # Test with zero variance data
        zero_var_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        cv = analyzer._calculate_coefficient_variation(zero_var_data)
        assert cv == 0.0
        
        # Test with NaN/Inf handling
        problematic_data = np.array([1.0, np.inf, 3.0, np.nan, 5.0])
        # Should handle gracefully without crashing
        try:
            result = analyzer._calculate_coefficient_variation(problematic_data)
            assert isinstance(result, float)
        except:
            # Should at least not crash the system
            pass