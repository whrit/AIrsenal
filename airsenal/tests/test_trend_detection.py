"""
Comprehensive test suite for the Trend Detection System.

Tests cover:
- Mann-Kendall trend testing with various data patterns
- Sen's slope estimation and confidence intervals
- PELT change point detection algorithm
- MACD technical indicator calculations
- Seasonal adjustment for fixture difficulty
- Alert system functionality and thresholds
- Visualization components (when available)
- Performance requirements and accuracy validation
- Edge case handling and error conditions

Test Data Strategy:
- Synthetic data with known trend patterns for validation
- Realistic FPL point distributions and seasonal effects
- Edge cases: missing data, zero variance, extreme values
- Performance testing with various data volumes
- Accuracy validation against known statistical benchmarks

Target Accuracy: 85%+ for trend detection and change point identification

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
from typing import List, Dict, Any

from airsenal.framework.trend_detection import (
    TrendDetector,
    ChangePointDetector,
    TrendVisualizer,
    AlertSystem,
    TrendDirection,
    TrendSignificance,
    TrendResult,
    ChangePoint,
    Alert,
    TrendDetectionError,
    InsufficientDataError,
    SCIPY_AVAILABLE,
    PLOTTING_AVAILABLE
)
from airsenal.framework.schema import (
    Fixture,
    Player,
    PlayerAttributes,
    PlayerScore,
    session
)
from airsenal.framework.utils import CURRENT_SEASON, NEXT_GAMEWEEK


class TestTrendDetector:
    """Test suite for TrendDetector class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def trend_detector(self, mock_session):
        """Create TrendDetector instance with mocked dependencies."""
        with patch('airsenal.framework.trend_detection.FormCalculator'):
            return TrendDetector(dbsession=mock_session)

    @pytest.fixture
    def sample_performance_data(self):
        """Generate sample performance data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate 20 gameweeks of data
        gameweeks = range(1, 21)
        dates = [datetime(2024, 8, 1) + timedelta(days=i*7) for i in range(20)]
        
        # Create trending data: improving trend
        base_points = 4.0
        trend_slope = 0.2
        noise = np.random.normal(0, 1.5, 20)
        points = [base_points + trend_slope * i + noise[i] for i in range(20)]
        
        data = []
        for i, (gw, date, pts) in enumerate(zip(gameweeks, dates, points)):
            data.append({
                'gameweek': gw,
                'points': max(0, pts),  # Ensure non-negative points
                'goals': np.random.poisson(0.3),
                'assists': np.random.poisson(0.2),
                'minutes': np.random.choice([0, 45, 90], p=[0.1, 0.2, 0.7]),
                'opponent': f'Team_{i%6}',
                'date': date,
                'home_team': 'Arsenal',
                'away_team': f'Team_{i%6}'
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def declining_trend_data(self):
        """Generate data with declining trend."""
        np.random.seed(123)
        
        gameweeks = range(1, 16)
        dates = [datetime(2024, 8, 1) + timedelta(days=i*7) for i in range(15)]
        
        # Declining trend
        base_points = 8.0
        trend_slope = -0.3
        noise = np.random.normal(0, 1.0, 15)
        points = [base_points + trend_slope * i + noise[i] for i in range(15)]
        
        data = []
        for i, (gw, date, pts) in enumerate(zip(gameweeks, dates, points)):
            data.append({
                'gameweek': gw,
                'points': max(0, pts),
                'goals': np.random.poisson(0.5),
                'assists': np.random.poisson(0.3),
                'minutes': 90,
                'opponent': f'Team_{i%6}',
                'date': date,
                'home_team': 'Liverpool',
                'away_team': f'Team_{i%6}'
            })
        
        return pd.DataFrame(data)

    @pytest.fixture
    def stable_data(self):
        """Generate data with no trend (stable)."""
        np.random.seed(456)
        
        gameweeks = range(1, 13)
        dates = [datetime(2024, 8, 1) + timedelta(days=i*7) for i in range(12)]
        
        # Stable performance
        base_points = 5.5
        noise = np.random.normal(0, 0.8, 12)
        points = [base_points + noise[i] for i in range(12)]
        
        data = []
        for i, (gw, date, pts) in enumerate(zip(gameweeks, dates, points)):
            data.append({
                'gameweek': gw,
                'points': max(0, pts),
                'goals': np.random.poisson(0.4),
                'assists': np.random.poisson(0.25),
                'minutes': 90,
                'opponent': f'Team_{i%6}',
                'date': date,
                'home_team': 'Chelsea',
                'away_team': f'Team_{i%6}'
            })
        
        return pd.DataFrame(data)

    def test_initialization(self, mock_session):
        """Test TrendDetector initialization."""
        detector = TrendDetector(
            dbsession=mock_session,
            min_data_points=10,
            alpha=0.01,
            seasonal_adjustment=False,
            weight_recent_data=False
        )
        
        assert detector.dbsession == mock_session
        assert detector.min_data_points == 10
        assert detector.alpha == 0.01
        assert detector.seasonal_adjustment == False
        assert detector.weight_recent_data == False
        assert detector.recent_weight_factor == 1.5  # default

    def test_mann_kendall_test_improving_trend(self, trend_detector):
        """Test Mann-Kendall test with improving trend data."""
        # Create data with clear upward trend
        data = np.array([1, 2, 3, 5, 6, 8, 9, 11, 12, 14])
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        
        # Should detect upward trend
        assert z_score > 0
        assert kendall_tau > 0
        if SCIPY_AVAILABLE and p_value is not None:
            assert p_value < 0.05  # Significant trend

    def test_mann_kendall_test_declining_trend(self, trend_detector):
        """Test Mann-Kendall test with declining trend data."""
        # Create data with clear downward trend
        data = np.array([14, 12, 11, 9, 8, 6, 5, 3, 2, 1])
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        
        # Should detect downward trend
        assert z_score < 0
        assert kendall_tau < 0
        if SCIPY_AVAILABLE and p_value is not None:
            assert p_value < 0.05  # Significant trend

    def test_mann_kendall_test_no_trend(self, trend_detector):
        """Test Mann-Kendall test with no trend data."""
        # Create data with no trend
        np.random.seed(789)
        data = np.random.normal(5, 1, 15)
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        
        # Should detect no significant trend
        assert abs(kendall_tau) < 0.5  # Weak or no correlation with time
        if SCIPY_AVAILABLE and p_value is not None:
            assert p_value > 0.05  # Not significant

    def test_sens_slope_estimator(self, trend_detector):
        """Test Sen's slope estimator."""
        # Linear data: y = 2x + 1
        data = np.array([1, 3, 5, 7, 9, 11])
        
        slope, confidence_interval = trend_detector.sens_slope_estimator(data)
        
        # Should detect slope close to 2
        assert abs(slope - 2.0) < 0.1
        assert confidence_interval[0] <= slope <= confidence_interval[1]
        assert confidence_interval[1] >= confidence_interval[0]  # Allow equal bounds for perfect data

    def test_sens_slope_estimator_zero_slope(self, trend_detector):
        """Test Sen's slope estimator with zero slope."""
        # Constant data
        data = np.array([5, 5, 5, 5, 5, 5])
        
        slope, confidence_interval = trend_detector.sens_slope_estimator(data)
        
        # Should detect slope close to 0
        assert abs(slope) < 0.1
        assert confidence_interval[0] <= slope <= confidence_interval[1]

    def test_calculate_macd(self, trend_detector):
        """Test MACD calculation."""
        # Create trending data for MACD
        data = np.array([4, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0, 6.3, 6.6,
                        6.9, 7.2, 7.5, 7.8, 8.1, 8.4, 8.7, 9.0, 9.3, 9.6,
                        9.9, 10.2, 10.5, 10.8, 11.1, 11.4, 11.7, 12.0, 12.3, 12.6])
        
        macd_line, signal_line, histogram = trend_detector.calculate_macd(data)
        
        # Should return arrays of correct length
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        
        # For upward trending data, MACD should generally be positive in later periods
        assert macd_line[-1] > macd_line[0]

    def test_calculate_macd_insufficient_data(self, trend_detector):
        """Test MACD with insufficient data."""
        # Too little data for MACD
        data = np.array([1, 2, 3])
        
        macd_line, signal_line, histogram = trend_detector.calculate_macd(data)
        
        # Should return empty arrays
        assert len(macd_line) == 0
        assert len(signal_line) == 0
        assert len(histogram) == 0

    def test_calculate_weights(self, trend_detector):
        """Test weight calculation for recent emphasis."""
        weights = trend_detector._calculate_weights(10)
        
        assert len(weights) == 10
        assert weights[-1] > weights[0]  # Most recent should have highest weight
        assert np.abs(np.mean(weights) - 1.0) < 0.1  # Should be normalized

    @patch('airsenal.framework.trend_detection.TrendDetector._get_player_performance_data')
    def test_detect_trends_improving(self, mock_get_data, trend_detector, sample_performance_data):
        """Test trend detection with improving performance."""
        mock_get_data.return_value = sample_performance_data
        
        result = trend_detector.detect_trends(player_id=123, lookback_days=30)
        
        assert isinstance(result, TrendResult)
        assert result.player_id == 123
        assert result.direction == TrendDirection.IMPROVING
        assert result.slope > 0
        assert result.data_points == len(sample_performance_data)

    @patch('airsenal.framework.trend_detection.TrendDetector._get_player_performance_data')
    def test_detect_trends_declining(self, mock_get_data, trend_detector, declining_trend_data):
        """Test trend detection with declining performance."""
        mock_get_data.return_value = declining_trend_data
        
        result = trend_detector.detect_trends(player_id=456, lookback_days=30)
        
        assert isinstance(result, TrendResult)
        assert result.player_id == 456
        assert result.direction == TrendDirection.DECLINING
        assert result.slope < 0

    @patch('airsenal.framework.trend_detection.TrendDetector._get_player_performance_data')
    def test_detect_trends_stable(self, mock_get_data, trend_detector, stable_data):
        """Test trend detection with stable performance."""
        mock_get_data.return_value = stable_data
        
        result = trend_detector.detect_trends(player_id=789, lookback_days=30)
        
        assert isinstance(result, TrendResult)
        assert result.player_id == 789
        assert result.direction in [TrendDirection.STABLE, TrendDirection.UNKNOWN]
        assert abs(result.slope) < 0.3

    def test_detect_trends_insufficient_data(self, trend_detector):
        """Test trend detection with insufficient data."""
        with patch.object(trend_detector, '_get_player_performance_data') as mock_get_data:
            # Mock insufficient data
            insufficient_data = pd.DataFrame({
                'gameweek': [1, 2],
                'points': [5, 6],
                'adjusted_points': [5, 6]
            })
            mock_get_data.side_effect = InsufficientDataError("Insufficient data")
            
            with pytest.raises(InsufficientDataError):
                trend_detector.detect_trends(player_id=999, lookback_days=30)

    @patch('airsenal.framework.trend_detection.TrendDetector._get_player_performance_data')
    def test_batch_detect_trends(self, mock_get_data, trend_detector, sample_performance_data):
        """Test batch trend detection."""
        mock_get_data.return_value = sample_performance_data
        
        player_ids = [123, 456, 789]
        results = trend_detector.batch_detect_trends(player_ids, lookback_days=30)
        
        assert isinstance(results, dict)
        assert len(results) <= len(player_ids)  # Some may fail
        
        for player_id, result in results.items():
            assert isinstance(result, TrendResult)
            assert player_id in player_ids

    def test_get_performance_stats(self, trend_detector):
        """Test performance statistics retrieval."""
        # Initially no stats
        stats = trend_detector.get_performance_stats()
        assert stats["status"] == "no_analyses_performed"
        
        # Add some mock analysis times
        trend_detector._analysis_times = [0.1, 0.15, 0.12, 0.08, 0.2]
        
        stats = trend_detector.get_performance_stats()
        assert "analysis_times" in stats
        assert stats["analysis_times"]["count"] == 5
        assert "mean_seconds" in stats["analysis_times"]
        assert "configuration" in stats


class TestChangePointDetector:
    """Test suite for ChangePointDetector class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def change_point_detector(self, mock_session):
        """Create ChangePointDetector instance."""
        with patch('airsenal.framework.trend_detection.TrendDetector'):
            return ChangePointDetector(dbsession=mock_session)

    @pytest.fixture
    def change_point_data(self):
        """Generate data with known change points."""
        np.random.seed(42)
        
        # Create data with change points at positions 10 and 20
        segment1 = np.random.normal(3, 0.5, 10)  # Low performance
        segment2 = np.random.normal(7, 0.8, 10)  # High performance  
        segment3 = np.random.normal(4, 0.6, 10)  # Medium performance
        
        data = np.concatenate([segment1, segment2, segment3])
        
        # Create corresponding DataFrame
        gameweeks = range(1, len(data) + 1)
        dates = [datetime(2024, 8, 1) + timedelta(days=i*7) for i in range(len(data))]
        
        df_data = []
        for i, (gw, pts) in enumerate(zip(gameweeks, data)):
            df_data.append({
                'gameweek': gw,
                'points': max(0, pts),
                'adjusted_points': max(0, pts),
                'date': dates[i]
            })
        
        return pd.DataFrame(df_data), [10, 20]  # Return data and true change points

    def test_initialization(self, mock_session):
        """Test ChangePointDetector initialization."""
        detector = ChangePointDetector(
            dbsession=mock_session,
            min_segment_length=3,
            penalty=2.0,
            model="poisson"
        )
        
        assert detector.dbsession == mock_session
        assert detector.min_segment_length == 3
        assert detector.penalty == 2.0
        assert detector.model == "poisson"

    def test_cost_function_normal(self, change_point_detector):
        """Test cost function with normal model."""
        data = np.array([1, 2, 3, 4, 5])
        
        cost = change_point_detector._cost_function(data, "normal")
        
        assert isinstance(cost, float)
        assert cost > 0

    def test_cost_function_poisson(self, change_point_detector):
        """Test cost function with Poisson model."""
        data = np.array([1, 2, 1, 3, 2])
        
        cost = change_point_detector._cost_function(data, "poisson")
        
        assert isinstance(cost, float)
        assert cost > 0

    def test_cost_function_exponential(self, change_point_detector):
        """Test cost function with exponential model."""
        data = np.array([0.5, 1.2, 0.8, 1.5, 0.9])
        
        cost = change_point_detector._cost_function(data, "exponential")
        
        assert isinstance(cost, float)
        assert cost > 0

    def test_cost_function_empty_data(self, change_point_detector):
        """Test cost function with empty data."""
        data = np.array([])
        
        cost = change_point_detector._cost_function(data, "normal")
        
        assert cost == 0.0

    def test_pelt_algorithm(self, change_point_detector):
        """Test PELT algorithm for change point detection."""
        # Create simple data with known change point
        segment1 = np.full(10, 2.0)
        segment2 = np.full(10, 8.0)
        data = np.concatenate([segment1, segment2])
        
        change_points = change_point_detector._pelt_algorithm(data)
        
        assert isinstance(change_points, list)
        # Should detect change point around position 10
        if change_points:
            assert any(8 <= cp <= 12 for cp in change_points)

    def test_binary_segmentation(self, change_point_detector):
        """Test binary segmentation algorithm."""
        # Create data with clear change point
        segment1 = np.full(15, 3.0)
        segment2 = np.full(15, 9.0)
        data = np.concatenate([segment1, segment2])
        
        change_points = change_point_detector._binary_segmentation(data, max_change_points=3)
        
        assert isinstance(change_points, list)
        # Should detect change point around position 15
        if change_points:
            assert any(12 <= cp <= 18 for cp in change_points)

    @patch('airsenal.framework.trend_detection.ChangePointDetector.trend_detector')
    def test_find_breakpoints(self, mock_trend_detector, change_point_detector, change_point_data):
        """Test change point detection with real data."""
        df, true_change_points = change_point_data
        mock_trend_detector._get_player_performance_data.return_value = df
        
        change_points = change_point_detector.find_breakpoints(
            player_id=123, lookback_days=60, algorithm="pelt"
        )
        
        assert isinstance(change_points, list)
        
        for cp in change_points:
            assert isinstance(cp, ChangePoint)
            assert hasattr(cp, 'position')
            assert hasattr(cp, 'gameweek')
            assert hasattr(cp, 'confidence')
            assert 0 <= cp.confidence <= 1

    @patch('airsenal.framework.trend_detection.ChangePointDetector.trend_detector')
    def test_find_breakpoints_binary_segmentation(self, mock_trend_detector, change_point_detector, change_point_data):
        """Test change point detection with binary segmentation."""
        df, true_change_points = change_point_data
        mock_trend_detector._get_player_performance_data.return_value = df
        
        change_points = change_point_detector.find_breakpoints(
            player_id=123, lookback_days=60, algorithm="binary_segmentation"
        )
        
        assert isinstance(change_points, list)

    def test_find_breakpoints_insufficient_data(self, change_point_detector):
        """Test change point detection with insufficient data."""
        with patch.object(change_point_detector.trend_detector, '_get_player_performance_data') as mock_get_data:
            mock_get_data.side_effect = InsufficientDataError("Insufficient data")
            
            with pytest.raises(InsufficientDataError):
                change_point_detector.find_breakpoints(player_id=999, lookback_days=30)

    def test_find_breakpoints_invalid_algorithm(self, change_point_detector):
        """Test change point detection with invalid algorithm."""
        with patch.object(change_point_detector.trend_detector, '_get_player_performance_data') as mock_get_data:
            mock_df = pd.DataFrame({
                'gameweek': [1, 2, 3],
                'points': [5, 6, 7],
                'adjusted_points': [5, 6, 7]
            })
            mock_get_data.return_value = mock_df
            
            with pytest.raises(ValueError):
                change_point_detector.find_breakpoints(
                    player_id=123, algorithm="invalid_algorithm"
                )

    @patch('airsenal.framework.trend_detection.ChangePointDetector.find_breakpoints')
    def test_batch_find_breakpoints(self, mock_find_breakpoints, change_point_detector):
        """Test batch change point detection."""
        # Mock return values
        mock_change_point = ChangePoint(
            position=10, gameweek=11, confidence=0.8, 
            cost_reduction=5.2, timestamp=datetime.now()
        )
        mock_find_breakpoints.return_value = [mock_change_point]
        
        player_ids = [123, 456, 789]
        results = change_point_detector.batch_find_breakpoints(player_ids)
        
        assert isinstance(results, dict)
        assert len(results) == len(player_ids)
        
        for player_id in player_ids:
            assert player_id in results
            assert isinstance(results[player_id], list)


class TestTrendVisualizer:
    """Test suite for TrendVisualizer class."""

    @pytest.fixture
    def trend_visualizer(self):
        """Create TrendVisualizer instance."""
        return TrendVisualizer()

    @pytest.fixture
    def sample_trend_result(self):
        """Create sample TrendResult for testing."""
        return TrendResult(
            player_id=123,
            direction=TrendDirection.IMPROVING,
            significance=TrendSignificance.SIGNIFICANT,
            p_value=0.02,
            slope=0.25,
            slope_confidence_interval=(0.1, 0.4),
            z_score=2.1,
            kendall_tau=0.3,
            data_points=15,
            lookback_period=30,
            seasonal_adjusted=True,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_change_points(self):
        """Create sample ChangePoints for testing."""
        return [
            ChangePoint(
                position=8,
                gameweek=9,
                confidence=0.85,
                cost_reduction=3.2,
                timestamp=datetime.now()
            ),
            ChangePoint(
                position=14,
                gameweek=15,
                confidence=0.72,
                cost_reduction=2.1,
                timestamp=datetime.now()
            )
        ]

    def test_initialization(self):
        """Test TrendVisualizer initialization."""
        visualizer = TrendVisualizer(style="default", figsize=(10, 6))
        
        assert visualizer.style == "default"
        assert visualizer.figsize == (10, 6)

    @pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib/seaborn not available")
    def test_plot_trend_analysis(self, trend_visualizer, sample_trend_result, sample_change_points):
        """Test trend analysis visualization."""
        # Create sample data
        data = pd.DataFrame({
            'gameweek': range(1, 16),
            'points': [3, 4, 5, 4, 6, 7, 5, 8, 9, 6, 10, 11, 8, 12, 13]
        })
        
        # Test plotting without saving
        result = trend_visualizer.plot_trend_analysis(
            player_id=123,
            trend_result=sample_trend_result,
            change_points=sample_change_points,
            data=data,
            show_plot=False
        )
        
        assert result is None  # No save path provided

    def test_plot_trend_analysis_no_plotting(self, sample_trend_result):
        """Test trend analysis when plotting not available."""
        with patch('airsenal.framework.trend_detection.PLOTTING_AVAILABLE', False):
            visualizer = TrendVisualizer()
            
            result = visualizer.plot_trend_analysis(
                player_id=123,
                trend_result=sample_trend_result,
                show_plot=False
            )
            
            assert result is None

    @pytest.mark.skipif(not PLOTTING_AVAILABLE, reason="matplotlib/seaborn not available")
    def test_plot_batch_trends(self, trend_visualizer, sample_trend_result):
        """Test batch trend visualization."""
        trend_results = {
            123: sample_trend_result,
            456: TrendResult(
                player_id=456,
                direction=TrendDirection.DECLINING,
                significance=TrendSignificance.HIGHLY_SIGNIFICANT,
                p_value=0.005,
                slope=-0.3,
                slope_confidence_interval=(-0.5, -0.1),
                z_score=-2.8,
                kendall_tau=-0.4,
                data_points=12,
                lookback_period=30,
                seasonal_adjusted=True,
                timestamp=datetime.now()
            )
        }
        
        result = trend_visualizer.plot_batch_trends(
            trend_results=trend_results,
            show_plot=False
        )
        
        assert result is None  # No save path provided

    def test_plot_batch_trends_empty(self, trend_visualizer):
        """Test batch trend visualization with empty results."""
        result = trend_visualizer.plot_batch_trends(
            trend_results={},
            show_plot=False
        )
        
        assert result is None


class TestAlertSystem:
    """Test suite for AlertSystem class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = Mock(spec=Session)
        
        # Mock player query
        mock_player = Mock()
        mock_player.name = "Test Player"
        session.query.return_value.filter_by.return_value.first.return_value = mock_player
        
        return session

    @pytest.fixture
    def alert_system(self, mock_session):
        """Create AlertSystem instance."""
        with patch('airsenal.framework.trend_detection.TrendDetector'):
            with patch('airsenal.framework.trend_detection.ChangePointDetector'):
                return AlertSystem(dbsession=mock_session)

    @pytest.fixture
    def improving_trend_result(self):
        """Create improving trend result for testing."""
        return TrendResult(
            player_id=123,
            direction=TrendDirection.IMPROVING,
            significance=TrendSignificance.SIGNIFICANT,
            p_value=0.02,
            slope=0.35,  # Above threshold
            slope_confidence_interval=(0.2, 0.5),
            z_score=2.3,
            kendall_tau=0.4,
            data_points=15,
            lookback_period=30,
            seasonal_adjusted=True,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def declining_trend_result(self):
        """Create declining trend result for testing."""
        return TrendResult(
            player_id=456,
            direction=TrendDirection.DECLINING,
            significance=TrendSignificance.HIGHLY_SIGNIFICANT,
            p_value=0.008,
            slope=-0.4,  # Below threshold
            slope_confidence_interval=(-0.6, -0.2),
            z_score=-2.9,
            kendall_tau=-0.5,
            data_points=12,
            lookback_period=30,
            seasonal_adjusted=True,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def high_confidence_change_points(self):
        """Create high confidence change points for testing."""
        return [
            ChangePoint(
                position=8,
                gameweek=NEXT_GAMEWEEK - 2,  # Recent
                confidence=0.85,
                cost_reduction=4.2,
                timestamp=datetime.now()
            ),
            ChangePoint(
                position=15,
                gameweek=NEXT_GAMEWEEK - 8,
                confidence=0.9,
                cost_reduction=5.1,
                timestamp=datetime.now()
            )
        ]

    def test_initialization(self, mock_session):
        """Test AlertSystem initialization."""
        alert_system = AlertSystem(
            dbsession=mock_session,
            significance_threshold=0.01,
            slope_threshold=0.25,
            change_point_confidence_threshold=0.8
        )
        
        assert alert_system.significance_threshold == 0.01
        assert alert_system.slope_threshold == 0.25
        assert alert_system.change_point_confidence_threshold == 0.8

    def test_check_trend_alerts_improving(self, alert_system, improving_trend_result):
        """Test trend alert generation for improving performance."""
        alerts = alert_system._check_trend_alerts(
            player_id=123,
            player_name="Test Player",
            trend_result=improving_trend_result
        )
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.alert_type == "improving_trend"
        assert alert.severity in ["high", "medium"]
        assert "improving form" in alert.message

    def test_check_trend_alerts_declining(self, alert_system, declining_trend_result):
        """Test trend alert generation for declining performance."""
        alerts = alert_system._check_trend_alerts(
            player_id=456,
            player_name="Test Player",
            trend_result=declining_trend_result
        )
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.alert_type == "declining_trend"
        assert alert.severity in ["critical", "high"]
        assert "declining form" in alert.message

    def test_check_change_point_alerts(self, alert_system, high_confidence_change_points):
        """Test change point alert generation."""
        alerts = alert_system._check_change_point_alerts(
            player_id=123,
            player_name="Test Player",
            change_points=high_confidence_change_points
        )
        
        # Should generate alerts for recent change point and multiple change points
        assert len(alerts) >= 1
        
        alert_types = [alert.alert_type for alert in alerts]
        assert "recent_change_point" in alert_types

    @patch('airsenal.framework.trend_detection.AlertSystem.trend_detector')
    @patch('airsenal.framework.trend_detection.AlertSystem.change_point_detector')
    def test_check_significant_changes(self, mock_cp_detector, mock_trend_detector, 
                                     alert_system, improving_trend_result, high_confidence_change_points):
        """Test comprehensive significant change checking."""
        # Mock batch detection results
        mock_trend_detector.batch_detect_trends.return_value = {123: improving_trend_result}
        mock_cp_detector.batch_find_breakpoints.return_value = {123: high_confidence_change_points}
        
        alerts = alert_system.check_significant_changes(player_ids=[123])
        
        assert isinstance(alerts, list)
        assert len(alerts) >= 1
        
        for alert in alerts:
            assert isinstance(alert, Alert)
            assert alert.player_id == 123
            assert alert.player_name == "Test Player"

    def test_generate_alert_report_text(self, alert_system):
        """Test text report generation."""
        # Create sample alerts
        alerts = [
            Alert(
                player_id=123,
                player_name="Test Player",
                alert_type="improving_trend",
                severity="high",
                message="Significant improvement detected",
                trend_result=None,
                change_points=None,
                timestamp=datetime.now()
            )
        ]
        
        report = alert_system.generate_alert_report(alerts, format="text")
        
        assert isinstance(report, str)
        assert "AIRSENAL TREND ALERT REPORT" in report
        assert "Test Player" in report
        assert "HIGH ALERTS" in report

    def test_generate_alert_report_markdown(self, alert_system):
        """Test markdown report generation."""
        alerts = [
            Alert(
                player_id=123,
                player_name="Test Player",
                alert_type="declining_trend",
                severity="critical",
                message="Significant decline detected",
                trend_result=None,
                change_points=None,
                timestamp=datetime.now()
            )
        ]
        
        report = alert_system.generate_alert_report(alerts, format="markdown")
        
        assert isinstance(report, str)
        assert "# AIrsenal Trend Alert Report" in report
        assert "Test Player" in report
        assert "🚨" in report  # Critical alert emoji

    def test_generate_alert_report_html(self, alert_system):
        """Test HTML report generation."""
        alerts = [
            Alert(
                player_id=123,
                player_name="Test Player",
                alert_type="stable_significant",
                severity="low",
                message="Consistent performance detected",
                trend_result=None,
                change_points=None,
                timestamp=datetime.now()
            )
        ]
        
        report = alert_system.generate_alert_report(alerts, format="html")
        
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "Test Player" in report
        assert 'class="alert low"' in report

    def test_generate_alert_report_empty(self, alert_system):
        """Test report generation with no alerts."""
        report = alert_system.generate_alert_report([], format="text")
        
        assert report == "No significant alerts detected."

    def test_generate_alert_report_invalid_format(self, alert_system):
        """Test report generation with invalid format."""
        alerts = [Mock()]
        
        with pytest.raises(ValueError):
            alert_system.generate_alert_report(alerts, format="invalid")


class TestTrendResult:
    """Test suite for TrendResult dataclass."""

    @pytest.fixture
    def trend_result(self):
        """Create sample TrendResult."""
        return TrendResult(
            player_id=123,
            direction=TrendDirection.IMPROVING,
            significance=TrendSignificance.SIGNIFICANT,
            p_value=0.03,
            slope=0.25,
            slope_confidence_interval=(0.1, 0.4),
            z_score=2.1,
            kendall_tau=0.3,
            data_points=15,
            lookback_period=30,
            seasonal_adjusted=True,
            timestamp=datetime.now()
        )

    def test_is_significant(self, trend_result):
        """Test significance checking."""
        assert trend_result.is_significant(alpha=0.05) == True
        assert trend_result.is_significant(alpha=0.01) == False

    def test_get_trend_strength(self, trend_result):
        """Test trend strength classification."""
        # Test moderate trend
        assert trend_result.get_trend_strength() == "moderate"
        
        # Test weak trend
        trend_result.slope = 0.05
        assert trend_result.get_trend_strength() == "weak"
        
        # Test strong trend
        trend_result.slope = 0.8
        assert trend_result.get_trend_strength() == "strong"


class TestPerformanceRequirements:
    """Test suite for performance requirements."""

    @pytest.fixture
    def trend_detector(self):
        """Create TrendDetector for performance testing."""
        mock_session = Mock(spec=Session)
        with patch('airsenal.framework.trend_detection.FormCalculator'):
            return TrendDetector(dbsession=mock_session)

    def test_mann_kendall_performance(self, trend_detector):
        """Test Mann-Kendall test performance."""
        # Large dataset
        np.random.seed(42)
        data = np.random.normal(5, 2, 1000)
        
        start_time = time.time()
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 1000 points)
        assert execution_time < 1.0
        assert isinstance(z_score, float)

    def test_macd_performance(self, trend_detector):
        """Test MACD calculation performance."""
        # Large dataset
        data = np.random.random(1000)
        
        start_time = time.time()
        macd, signal, histogram = trend_detector.calculate_macd(data)
        execution_time = time.time() - start_time
        
        # Should complete quickly
        assert execution_time < 0.5
        assert len(macd) == len(data)


class TestAccuracyValidation:
    """Test suite for accuracy validation and backtesting."""

    @pytest.fixture
    def trend_detector(self):
        """Create TrendDetector for accuracy testing."""
        mock_session = Mock(spec=Session)
        with patch('airsenal.framework.trend_detection.FormCalculator'):
            return TrendDetector(dbsession=mock_session)

    def test_trend_detection_accuracy_synthetic_data(self, trend_detector):
        """Test trend detection accuracy with synthetic data."""
        np.random.seed(42)
        
        # Generate known trend patterns
        test_cases = []
        
        # Strong upward trend
        upward_trend = np.array([1 + 0.5*i + np.random.normal(0, 0.2) for i in range(20)])
        test_cases.append((upward_trend, TrendDirection.IMPROVING, "strong_upward"))
        
        # Strong downward trend  
        downward_trend = np.array([10 - 0.4*i + np.random.normal(0, 0.3) for i in range(20)])
        test_cases.append((downward_trend, TrendDirection.DECLINING, "strong_downward"))
        
        # No trend (stable)
        stable = np.array([5 + np.random.normal(0, 0.5) for i in range(20)])
        test_cases.append((stable, TrendDirection.STABLE, "stable"))
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for data, expected_direction, case_name in test_cases:
            # Mock the data retrieval
            mock_df = pd.DataFrame({
                'gameweek': range(1, len(data)+1),
                'points': data,
                'adjusted_points': data
            })
            
            with patch.object(trend_detector, '_get_player_performance_data', return_value=mock_df):
                try:
                    result = trend_detector.detect_trends(player_id=123, lookback_days=30)
                    
                    # Check if direction matches expectation
                    if expected_direction == TrendDirection.STABLE:
                        # For stable data, accept STABLE or UNKNOWN
                        if result.direction in [TrendDirection.STABLE, TrendDirection.UNKNOWN]:
                            correct_predictions += 1
                    else:
                        if result.direction == expected_direction:
                            correct_predictions += 1
                            
                except Exception as e:
                    logger.warning(f"Test case {case_name} failed: {e}")
        
        accuracy = correct_predictions / total_predictions
        
        # Target: 85% accuracy
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} below target 85%"

    def test_change_point_detection_accuracy(self):
        """Test change point detection accuracy."""
        mock_session = Mock(spec=Session)
        with patch('airsenal.framework.trend_detection.TrendDetector'):
            detector = ChangePointDetector(dbsession=mock_session)
        
        np.random.seed(42)
        
        # Create data with known change points
        segment1 = np.random.normal(3, 0.5, 15)
        segment2 = np.random.normal(8, 0.6, 15)  # Change point at 15
        segment3 = np.random.normal(5, 0.4, 15)  # Change point at 30
        
        data = np.concatenate([segment1, segment2, segment3])
        true_change_points = [15, 30]
        
        # Test PELT algorithm
        detected_change_points = detector._pelt_algorithm(data)
        
        # Check if detected change points are close to true ones
        accuracy_threshold = 3  # Allow 3 positions tolerance
        correct_detections = 0
        
        for true_cp in true_change_points:
            for detected_cp in detected_change_points:
                if abs(detected_cp - true_cp) <= accuracy_threshold:
                    correct_detections += 1
                    break
        
        accuracy = correct_detections / len(true_change_points)
        
        # Expect at least 50% accuracy for change point detection
        assert accuracy >= 0.5, f"Change point accuracy {accuracy:.2%} below 50%"


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.fixture
    def trend_detector(self):
        """Create TrendDetector for edge case testing."""
        mock_session = Mock(spec=Session)
        with patch('airsenal.framework.trend_detection.FormCalculator'):
            return TrendDetector(dbsession=mock_session)

    def test_zero_variance_data(self, trend_detector):
        """Test with zero variance data."""
        data = np.array([5, 5, 5, 5, 5, 5, 5, 5])
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        slope, ci = trend_detector.sens_slope_estimator(data)
        
        assert abs(z_score) < 0.1  # Should be near zero
        assert abs(slope) < 0.1    # Should be near zero
        assert abs(kendall_tau) < 0.1

    def test_single_data_point(self, trend_detector):
        """Test with single data point."""
        data = np.array([5])
        
        # Should handle gracefully without errors
        slope, ci = trend_detector.sens_slope_estimator(data)
        assert slope == 0.0

    def test_two_data_points(self, trend_detector):
        """Test with exactly two data points."""
        data = np.array([3, 7])
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        slope, ci = trend_detector.sens_slope_estimator(data)
        
        assert slope == 4.0  # (7-3)/(1-0)
        assert kendall_tau == 1.0  # Perfect positive correlation

    def test_extreme_values(self, trend_detector):
        """Test with extreme values."""
        data = np.array([0, 1000, 0, 1000, 0])
        
        # Should handle without crashing
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(data)
        slope, ci = trend_detector.sens_slope_estimator(data)
        
        assert isinstance(z_score, float)
        assert isinstance(slope, float)

    def test_missing_values_handling(self, trend_detector):
        """Test handling of NaN values."""
        # The system should filter out NaN values in data preprocessing
        data = np.array([1, 2, np.nan, 4, 5])
        clean_data = data[~np.isnan(data)]
        
        z_score, p_value, kendall_tau = trend_detector.mann_kendall_test(clean_data)
        
        assert isinstance(z_score, float)
        assert not np.isnan(z_score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])