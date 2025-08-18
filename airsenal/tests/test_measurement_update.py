"""
Comprehensive tests for the measurement update system.

Tests cover:
- MeasurementProcessor: Feature extraction and normalization
- NoiseEstimator: Adaptive noise estimation
- OutlierDetector: Statistical outlier detection
- InnovationMonitor: Filter consistency monitoring
- BatchUpdater: Efficient batch processing
- Integration with Kalman filters
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Optional

from airsenal.framework.measurement_update import (
    MeasurementProcessor,
    NoiseEstimator,
    OutlierDetector,
    InnovationMonitor,
    BatchUpdater,
    MeasurementConfig,
    NoiseConfig,
    OutlierResult,
    InnovationStats,
    create_measurement_pipeline,
    process_gameweek_measurements
)
from airsenal.framework.kalman_filter import FilterConfig, FilterState
from airsenal.framework.kalman_player_model import KalmanPlayerModel
from airsenal.framework.adaptive_player_model import StateSpaceConfig


@dataclass
class MockPlayerScore:
    """Mock PlayerScore for testing."""
    player_id: int
    player_team: str
    opponent: str
    goals: int
    assists: int
    minutes: int
    bonus: int
    expected_goals: Optional[float] = None
    expected_assists: Optional[float] = None
    player: Optional[Mock] = None
    fixture: Optional[Mock] = None


@dataclass
class MockPlayer:
    """Mock Player for testing."""
    player_id: int
    position: str
    name: str = "Test Player"


class TestMeasurementProcessor:
    """Test suite for MeasurementProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create MeasurementProcessor instance."""
        config = MeasurementConfig(
            measurement_features=["goals", "assists", "minutes", "bonus"],
            normalize_by_position=True,
            normalize_by_opponent=True
        )
        return MeasurementProcessor(config)
    
    @pytest.fixture
    def mock_player_score(self):
        """Create mock PlayerScore."""
        return MockPlayerScore(
            player_id=123,
            player_team="Arsenal",
            opponent="Liverpool",
            goals=1,
            assists=0,
            minutes=90,
            bonus=3,
            expected_goals=0.8,
            expected_assists=0.2
        )
    
    def test_extract_features_basic(self, processor, mock_player_score):
        """Test basic feature extraction."""
        features = processor.extract_features(mock_player_score)
        
        assert "goals" in features
        assert "assists" in features
        assert "minutes" in features
        assert "bonus" in features
        
        assert features["goals"] == 1.0
        assert features["assists"] == 0.0
        assert features["minutes"] == 90.0
        assert features["bonus"] == 3.0
    
    def test_extract_features_missing_values(self, processor):
        """Test feature extraction with missing values."""
        score = MockPlayerScore(
            player_id=123,
            player_team="Arsenal",
            opponent="Liverpool",
            goals=None,
            assists=1,
            minutes=45,
            bonus=None
        )
        
        features = processor.extract_features(score)
        
        assert np.isnan(features["goals"])
        assert features["assists"] == 1.0
        assert features["minutes"] == 45.0
        assert np.isnan(features["bonus"])
    
    def test_normalize_features_position_scaling(self, processor):
        """Test position-specific feature scaling."""
        features = {"goals": 1.0, "assists": 1.0, "minutes": 90.0, "bonus": 2.0}
        
        # Test midfielder (baseline)
        mid_normalized = processor.normalize_features(features, "MID")
        
        # Test forward (should have higher goal scaling)
        fwd_normalized = processor.normalize_features(features, "FWD")
        
        # Forward should have higher goal coefficient
        assert fwd_normalized[0] > mid_normalized[0]
        
        # Test goalkeeper (should have lower goal scaling)
        gk_normalized = processor.normalize_features(features, "GK")
        assert gk_normalized[0] < mid_normalized[0]
    
    def test_normalize_features_opponent_adjustment(self, processor):
        """Test opponent strength adjustment."""
        features = {"goals": 1.0, "assists": 1.0, "minutes": 90.0, "bonus": 2.0}
        
        # Strong opponent should reduce attacking stats
        strong_opponent = processor.normalize_features(
            features, "MID", opponent_strength=0.9
        )
        
        # Weak opponent should maintain/increase attacking stats
        weak_opponent = processor.normalize_features(
            features, "MID", opponent_strength=0.1
        )
        
        # Goals and assists should be lower against strong opponents
        assert strong_opponent[0] < weak_opponent[0]  # goals
        assert strong_opponent[1] < weak_opponent[1]  # assists
    
    def test_normalize_features_home_away(self, processor):
        """Test home/away adjustment."""
        features = {"goals": 1.0, "assists": 1.0, "minutes": 90.0, "bonus": 2.0}
        
        home_normalized = processor.normalize_features(features, "MID", is_home=True)
        away_normalized = processor.normalize_features(features, "MID", is_home=False)
        
        # Home should generally be better for attacking stats
        assert home_normalized[0] >= away_normalized[0]  # goals
        assert home_normalized[1] >= away_normalized[1]  # assists
    
    def test_missing_data_handling(self, processor):
        """Test different missing data strategies."""
        # Test default strategy
        default_value = processor._handle_missing_data("goals", "MID")
        assert isinstance(default_value, float)
        
        # Test minutes default
        minutes_default = processor._handle_missing_data("minutes", "MID")
        assert minutes_default >= 0
    
    def test_update_statistics(self, processor):
        """Test statistics update functionality."""
        # Create sample measurements
        measurements = [
            jnp.array([1.0, 0.5, 90.0, 2.0]),
            jnp.array([0.0, 1.0, 75.0, 1.0]),
            jnp.array([2.0, 0.0, 60.0, 3.0])
        ]
        positions = ["MID", "MID", "FWD"]
        
        processor.update_statistics(measurements, positions)
        
        # Check that statistics were updated
        assert "goals" in processor._global_stats
        assert "MID" in processor._position_stats
        assert processor._global_stats["goals"]["count"] > 0


class TestNoiseEstimator:
    """Test suite for NoiseEstimator."""
    
    @pytest.fixture
    def noise_estimator(self):
        """Create NoiseEstimator instance."""
        config = NoiseConfig(
            estimation_window=5,
            adaptation_rate=0.2
        )
        return NoiseEstimator(config)
    
    def test_estimate_noise_matrix_basic(self, noise_estimator):
        """Test basic noise matrix estimation."""
        noise_matrix = noise_estimator.estimate_noise_matrix("MID", gameweek=10)
        
        # Should be square matrix
        assert noise_matrix.shape[0] == noise_matrix.shape[1]
        
        # Should be positive definite (all eigenvalues positive)
        eigenvals = jnp.linalg.eigvals(noise_matrix)
        assert jnp.all(eigenvals > 0)
    
    def test_position_specific_noise(self, noise_estimator):
        """Test position-specific noise levels."""
        gk_noise = noise_estimator.estimate_noise_matrix("GK", gameweek=10)
        fwd_noise = noise_estimator.estimate_noise_matrix("FWD", gameweek=10)
        
        # Forwards should generally have higher noise than goalkeepers
        assert jnp.trace(fwd_noise) > jnp.trace(gk_noise)
    
    def test_temporal_adjustment(self, noise_estimator):
        """Test temporal noise adjustment."""
        early_noise = noise_estimator.estimate_noise_matrix("MID", gameweek=2)
        mid_noise = noise_estimator.estimate_noise_matrix("MID", gameweek=15)
        
        # Early season should have higher noise
        assert jnp.trace(early_noise) > jnp.trace(mid_noise)
    
    def test_adaptive_estimation(self, noise_estimator):
        """Test adaptive noise estimation from innovations."""
        # Create sample innovation sequence
        innovations = [
            jnp.array([0.5, -0.3, 2.0, 0.1]),
            jnp.array([-0.2, 0.8, -1.5, 0.5]),
            jnp.array([1.1, 0.0, 0.5, -0.3])
        ]
        
        # Store innovations
        noise_estimator._innovation_history["MID"] = innovations
        
        # Estimate with adaptation
        noise_matrix = noise_estimator.estimate_noise_matrix("MID", gameweek=10)
        
        # Should incorporate innovation information
        assert noise_matrix.shape == (4, 4)
        assert jnp.all(jnp.linalg.eigvals(noise_matrix) > 0)
    
    def test_update_noise_estimates(self, noise_estimator):
        """Test noise estimate updates."""
        innovations = {
            "MID": jnp.array([0.5, -0.3, 2.0, 0.1]),
            "FWD": jnp.array([1.0, 0.2, -0.5, 0.8])
        }
        measurements = {
            "MID": jnp.array([1.0, 0.5, 85.0, 2.0]),
            "FWD": jnp.array([2.0, 0.0, 70.0, 3.0])
        }
        
        # Should not raise errors
        noise_estimator.update_noise_estimates(innovations, measurements)
        
        # Check that history was updated
        assert len(noise_estimator._innovation_history["MID"]) > 0
        assert len(noise_estimator._innovation_history["FWD"]) > 0


class TestOutlierDetector:
    """Test suite for OutlierDetector."""
    
    @pytest.fixture
    def outlier_detector(self):
        """Create OutlierDetector instance."""
        config = MeasurementConfig(
            outlier_detection_method="zscore",
            outlier_threshold=2.0
        )
        return OutlierDetector(config)
    
    def test_zscore_detection_normal(self, outlier_detector):
        """Test z-score detection with normal values."""
        measurement = jnp.array([1.0, 0.5, 80.0, 2.0])
        historical = [
            jnp.array([0.8, 0.3, 85.0, 1.5]),
            jnp.array([1.2, 0.7, 75.0, 2.5]),
            jnp.array([0.9, 0.4, 90.0, 1.8])
        ]
        
        result = outlier_detector.detect_outliers(measurement, "MID", historical)
        
        assert isinstance(result, OutlierResult)
        assert not result.is_outlier
        assert result.confidence >= 0
    
    def test_zscore_detection_outlier(self, outlier_detector):
        """Test z-score detection with clear outlier."""
        measurement = jnp.array([5.0, 3.0, 90.0, 8.0])  # Extremely high goals/assists/bonus
        historical = [
            jnp.array([0.5, 0.2, 85.0, 1.0]),
            jnp.array([0.3, 0.1, 75.0, 0.5]),
            jnp.array([0.8, 0.4, 90.0, 1.5])
        ]
        
        result = outlier_detector.detect_outliers(measurement, "MID", historical)
        
        assert result.is_outlier
        assert result.outlier_score > outlier_detector.config.outlier_threshold
        assert result.confidence > 0.5
    
    def test_mahalanobis_detection(self, outlier_detector):
        """Test Mahalanobis distance detection."""
        outlier_detector.config.outlier_detection_method = "mahalanobis"
        
        measurement = jnp.array([2.0, 2.0, 90.0, 5.0])
        historical = [
            jnp.array([0.5, 0.2, 85.0, 1.0]),
            jnp.array([0.3, 0.1, 75.0, 0.5]),
            jnp.array([0.8, 0.4, 90.0, 1.5]),
            jnp.array([0.6, 0.3, 80.0, 1.2]),
            jnp.array([0.4, 0.2, 88.0, 0.8])
        ]
        
        result = outlier_detector.detect_outliers(measurement, "MID", historical)
        
        assert isinstance(result, OutlierResult)
        assert result.outlier_type == "multivariate"
    
    def test_prediction_based_detection(self, outlier_detector):
        """Test prediction-based outlier detection."""
        measurement = jnp.array([3.0, 1.0, 90.0, 4.0])
        predicted = jnp.array([0.5, 0.3, 85.0, 1.0])
        
        result = outlier_detector._prediction_based_detection(measurement, predicted)
        
        assert isinstance(result, OutlierResult)
        assert "prediction" in result.outlier_type
    
    def test_insufficient_data(self, outlier_detector):
        """Test outlier detection with insufficient historical data."""
        measurement = jnp.array([1.0, 0.5, 80.0, 2.0])
        historical = [jnp.array([0.8, 0.3, 85.0, 1.5])]  # Only one historical point
        
        result = outlier_detector.detect_outliers(measurement, "MID", historical)
        
        assert not result.is_outlier
        assert result.outlier_type == "insufficient_data"
    
    def test_recommended_actions(self, outlier_detector):
        """Test outlier handling recommendations."""
        # High confidence positive outlier (hat-trick)
        high_conf_result = OutlierResult(
            is_outlier=True,
            outlier_score=4.0,
            outlier_type="positive",
            confidence=0.9,
            recommended_action=""
        )
        
        action = outlier_detector._get_recommended_action(high_conf_result, "FWD")
        assert action in ["gradual_trust", "robust_update", "reject"]
        
        # Low confidence outlier
        low_conf_result = OutlierResult(
            is_outlier=True,
            outlier_score=1.5,
            outlier_type="negative",
            confidence=0.3,
            recommended_action=""
        )
        
        action = outlier_detector._get_recommended_action(low_conf_result, "MID")
        assert action == "normal_update"


class TestInnovationMonitor:
    """Test suite for InnovationMonitor."""
    
    @pytest.fixture
    def innovation_monitor(self):
        """Create InnovationMonitor instance."""
        return InnovationMonitor(window_size=10)
    
    def test_add_innovation_basic(self, innovation_monitor):
        """Test basic innovation addition."""
        innovation = jnp.array([0.5, -0.2, 1.0, 0.3])
        innovation_cov = jnp.eye(4) * 0.5
        
        stats = innovation_monitor.add_innovation(
            innovation, innovation_cov, "MID", timestamp=1.0
        )
        
        assert isinstance(stats, InnovationStats)
        assert stats.degrees_of_freedom == 4
        assert stats.normalized_innovation_squared >= 0
    
    def test_nis_computation(self, innovation_monitor):
        """Test Normalized Innovation Squared computation."""
        innovation = jnp.array([1.0, 0.0, 0.0, 0.0])
        innovation_cov = jnp.eye(4)
        
        nis = innovation_monitor._compute_nis(innovation, innovation_cov)
        
        # Should be close to 1.0 for this case
        assert abs(nis - 1.0) < 0.1
    
    def test_whiteness_test(self, innovation_monitor):
        """Test innovation whiteness test."""
        # Create correlated innovations (should fail whiteness)
        innovations = [
            jnp.array([1.0, 0.5, 0.0, 0.0]),
            jnp.array([0.8, 0.4, 0.1, 0.0]),
            jnp.array([0.9, 0.45, -0.1, 0.0]),
            jnp.array([1.1, 0.55, 0.0, 0.1]),
            jnp.array([0.7, 0.35, 0.2, 0.0])
        ]
        
        whiteness_stat = innovation_monitor._compute_whiteness_test(innovations)
        
        assert isinstance(whiteness_stat, float)
        assert whiteness_stat >= 0
    
    def test_consistency_check(self, innovation_monitor):
        """Test filter consistency checking."""
        # Normal NIS value should pass
        assert innovation_monitor._check_consistency(4.0, dof=4)
        
        # Extremely high NIS should fail
        assert not innovation_monitor._check_consistency(20.0, dof=4)
        
        # Extremely low NIS should also fail
        assert not innovation_monitor._check_consistency(0.1, dof=4)
    
    def test_adaptive_recommendations(self, innovation_monitor):
        """Test adaptive filter recommendations."""
        # Simulate high NIS values
        position = "MID"
        innovation_monitor._nis_history[position] = [8.0, 9.0, 7.5, 8.5, 9.2]
        innovation_monitor._innovation_sequences[position] = [jnp.array([1.0, 1.0, 1.0, 1.0])]
        
        recommendations = innovation_monitor.get_adaptive_recommendations(position)
        
        assert isinstance(recommendations, dict)
        assert "increase_process_noise" in recommendations
        assert "increase_measurement_noise" in recommendations
        assert "confidence" in recommendations
        
        # High NIS should recommend increasing noise
        assert recommendations["increase_measurement_noise"] or recommendations["increase_process_noise"]
    
    def test_window_size_maintenance(self, innovation_monitor):
        """Test that innovation history maintains window size."""
        position = "MID"
        
        # Add more innovations than window size
        for i in range(15):
            innovation = jnp.array([float(i), 0.0, 0.0, 0.0])
            innovation_cov = jnp.eye(4)
            innovation_monitor.add_innovation(innovation, innovation_cov, position, float(i))
        
        # Should maintain window size
        assert len(innovation_monitor._innovation_sequences[position]) <= innovation_monitor.window_size
        assert len(innovation_monitor._nis_history[position]) <= innovation_monitor.window_size


class TestBatchUpdater:
    """Test suite for BatchUpdater."""
    
    @pytest.fixture
    def batch_updater(self):
        """Create BatchUpdater instance."""
        processor = MeasurementProcessor()
        noise_estimator = NoiseEstimator()
        outlier_detector = OutlierDetector()
        innovation_monitor = InnovationMonitor()
        
        return BatchUpdater(processor, noise_estimator, outlier_detector, innovation_monitor)
    
    @pytest.fixture
    def mock_player_scores(self):
        """Create mock player scores for testing."""
        scores = []
        for i in range(5):
            player = MockPlayer(player_id=100 + i, position="MID")
            score = MockPlayerScore(
                player_id=100 + i,
                player_team="Arsenal",
                opponent="Liverpool",
                goals=np.random.randint(0, 3),
                assists=np.random.randint(0, 2),
                minutes=np.random.randint(60, 91),
                bonus=np.random.randint(0, 4)
            )
            score.player = player
            scores.append(score)
        
        return scores
    
    @pytest.fixture
    def mock_kalman_model(self):
        """Create mock KalmanPlayerModel."""
        config = StateSpaceConfig(
            state_dim=4,
            obs_dim=4,
            state_names=["skill", "form", "consistency", "momentum"],
            obs_names=["goals", "assists", "minutes", "bonus"]
        )
        
        model = KalmanPlayerModel(config)
        return model
    
    def test_process_batch_basic(self, batch_updater, mock_player_scores):
        """Test basic batch processing."""
        measurements, noise_matrices = batch_updater.process_batch(
            mock_player_scores, gameweek=10, season="2023"
        )
        
        assert isinstance(measurements, dict)
        assert isinstance(noise_matrices, dict)
        assert len(measurements) == len(mock_player_scores)
    
    def test_process_batch_position_grouping(self, batch_updater):
        """Test that batch processing groups by position correctly."""
        # Create scores with different positions
        scores = []
        positions = ["GK", "DEF", "MID", "FWD"]
        
        for i, pos in enumerate(positions):
            player = MockPlayer(player_id=200 + i, position=pos)
            score = MockPlayerScore(
                player_id=200 + i,
                player_team="Arsenal",
                opponent="Liverpool",
                goals=0 if pos == "GK" else 1,
                assists=0,
                minutes=90,
                bonus=1
            )
            score.player = player
            scores.append(score)
        
        measurements, noise_matrices = batch_updater.process_batch(
            scores, gameweek=10, season="2023"
        )
        
        # Should have noise matrices for each position
        assert len(noise_matrices) == len(positions)
        for pos in positions:
            assert pos in noise_matrices
    
    @patch('airsenal.framework.measurement_update.time.time')
    def test_batch_processing_timing(self, mock_time, batch_updater, mock_player_scores):
        """Test batch processing timing measurement."""
        mock_time.side_effect = [0.0, 0.1]  # Start and end times
        
        batch_updater.process_batch(mock_player_scores, gameweek=10, season="2023")
        
        stats = batch_updater.get_batch_statistics()
        assert stats["processing_time"] > 0
        assert stats["total_processed"] == len(mock_player_scores)
    
    def test_update_batch_mock(self, batch_updater, mock_kalman_model):
        """Test batch update with mock data."""
        # Create mock measurements and noise matrices
        player_measurements = {
            100: jnp.array([1.0, 0.5, 85.0, 2.0]),
            101: jnp.array([0.0, 1.0, 90.0, 1.0]),
            102: jnp.array([2.0, 0.0, 75.0, 3.0])
        }
        
        position_noise_matrices = {
            "MID": jnp.eye(4) * 0.5
        }
        
        player_positions = {100: "MID", 101: "MID", 102: "MID"}
        
        # Initialize some player states
        for player_id in player_measurements:
            mock_kalman_model.initialize_state(player_id, None, 10, "2023", "MID")
        
        updated_states = batch_updater.update_batch(
            mock_kalman_model,
            player_measurements,
            position_noise_matrices,
            player_positions,
            gameweek=10,
            season="2023"
        )
        
        assert isinstance(updated_states, dict)
        assert len(updated_states) <= len(player_measurements)
    
    def test_get_batch_statistics(self, batch_updater, mock_player_scores):
        """Test batch statistics collection."""
        # Process some data first
        batch_updater.process_batch(mock_player_scores, gameweek=10, season="2023")
        
        stats = batch_updater.get_batch_statistics()
        
        assert isinstance(stats, dict)
        assert "total_processed" in stats
        assert "outliers_detected" in stats
        assert "outlier_rate" in stats
        assert stats["total_processed"] > 0
    
    def test_cache_management(self, batch_updater):
        """Test cache management functionality."""
        # Add some cached data
        batch_updater._cached_noise_matrices["MID_10"] = jnp.eye(4)
        batch_updater._cached_noise_matrices["FWD_10"] = jnp.eye(4)
        
        assert len(batch_updater._cached_noise_matrices) == 2
        
        # Clear cache
        batch_updater.clear_cache()
        assert len(batch_updater._cached_noise_matrices) == 0


class TestIntegration:
    """Integration tests for the measurement update system."""
    
    def test_create_measurement_pipeline(self):
        """Test pipeline creation utility."""
        pipeline = create_measurement_pipeline()
        
        assert isinstance(pipeline, BatchUpdater)
        assert pipeline.measurement_processor is not None
        assert pipeline.noise_estimator is not None
        assert pipeline.outlier_detector is not None
        assert pipeline.innovation_monitor is not None
    
    def test_measurement_pipeline_custom_config(self):
        """Test pipeline creation with custom configurations."""
        measurement_config = MeasurementConfig(
            measurement_features=["goals", "assists"],
            outlier_threshold=2.5
        )
        
        noise_config = NoiseConfig(
            estimation_window=15,
            adaptation_rate=0.05
        )
        
        pipeline = create_measurement_pipeline(measurement_config, noise_config)
        
        assert len(pipeline.measurement_processor.config.measurement_features) == 2
        assert pipeline.noise_estimator.config.estimation_window == 15
        assert pipeline.outlier_detector.config.outlier_threshold == 2.5
    
    def test_process_gameweek_measurements_integration(self):
        """Test full gameweek processing integration."""
        # Create mock data
        scores = []
        for i in range(3):
            player = MockPlayer(player_id=300 + i, position="MID")
            score = MockPlayerScore(
                player_id=300 + i,
                player_team="Arsenal",
                opponent="Liverpool",
                goals=i % 2,
                assists=(i + 1) % 2,
                minutes=90 - i * 5,
                bonus=i
            )
            score.player = player
            scores.append(score)
        
        # Create Kalman model
        config = StateSpaceConfig(
            state_dim=4,
            obs_dim=4,
            state_names=["skill", "form", "consistency", "momentum"],
            obs_names=["goals", "assists", "minutes", "bonus"]
        )
        kalman_model = KalmanPlayerModel(config)
        
        # Process gameweek
        updated_states = process_gameweek_measurements(
            scores, kalman_model, gameweek=10, season="2023"
        )
        
        assert isinstance(updated_states, dict)
        # Some players should be successfully processed
        assert len(updated_states) >= 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_measurement_processor_error_handling(self):
        """Test MeasurementProcessor error handling."""
        processor = MeasurementProcessor()
        
        # Test with None input
        features = processor.extract_features(None)
        assert isinstance(features, dict)
        
        # Test with invalid position
        measurement = processor.normalize_features(
            {"goals": 1.0}, position="INVALID"
        )
        assert isinstance(measurement, jnp.ndarray)
    
    def test_noise_estimator_error_handling(self):
        """Test NoiseEstimator error handling."""
        estimator = NoiseEstimator()
        
        # Test with invalid position
        noise_matrix = estimator.estimate_noise_matrix("INVALID")
        assert isinstance(noise_matrix, jnp.ndarray)
        assert noise_matrix.shape[0] == noise_matrix.shape[1]
    
    def test_outlier_detector_error_handling(self):
        """Test OutlierDetector error handling."""
        detector = OutlierDetector()
        
        # Test with empty historical data
        result = detector.detect_outliers(
            jnp.array([1.0, 0.0, 90.0, 2.0]), 
            "MID", 
            historical_measurements=[]
        )
        assert isinstance(result, OutlierResult)
        assert not result.is_outlier
    
    def test_innovation_monitor_error_handling(self):
        """Test InnovationMonitor error handling."""
        monitor = InnovationMonitor()
        
        # Test with invalid input
        stats = monitor.add_innovation(
            None, jnp.eye(4), "MID", 1.0
        )
        assert isinstance(stats, InnovationStats)
    
    def test_batch_updater_error_handling(self):
        """Test BatchUpdater error handling."""
        processor = MeasurementProcessor()
        noise_estimator = NoiseEstimator()
        updater = BatchUpdater(processor, noise_estimator)
        
        # Test with empty player scores
        measurements, noise_matrices = updater.process_batch(
            [], gameweek=10, season="2023"
        )
        assert isinstance(measurements, dict)
        assert isinstance(noise_matrices, dict)


# Performance and numerical tests

class TestNumericalStability:
    """Test numerical stability and performance."""
    
    def test_noise_matrix_positive_definiteness(self):
        """Test that noise matrices are always positive definite."""
        estimator = NoiseEstimator()
        
        for position in ["GK", "DEF", "MID", "FWD"]:
            for gameweek in [1, 15, 30, 38]:
                noise_matrix = estimator.estimate_noise_matrix(position, gameweek)
                
                # Check positive definiteness
                eigenvals = jnp.linalg.eigvals(noise_matrix)
                assert jnp.all(eigenvals > 0), f"Non-positive eigenvalues for {position} GW{gameweek}"
    
    def test_measurement_vector_bounds(self):
        """Test that measurement vectors stay within reasonable bounds."""
        processor = MeasurementProcessor()
        
        # Extreme input values
        extreme_features = {
            "goals": 10.0,  # Unrealistic but possible
            "assists": 5.0,
            "minutes": 90.0,
            "bonus": 3.0
        }
        
        measurement = processor.normalize_features(extreme_features, "FWD")
        
        # Should not contain NaN or infinite values
        assert jnp.all(jnp.isfinite(measurement))
        
        # Should be within reasonable bounds (implementation dependent)
        assert jnp.all(measurement >= -10.0)
        assert jnp.all(measurement <= 20.0)
    
    def test_large_batch_processing(self):
        """Test processing large batches efficiently."""
        processor = MeasurementProcessor()
        noise_estimator = NoiseEstimator()
        updater = BatchUpdater(processor, noise_estimator)
        
        # Create large batch of mock scores
        scores = []
        for i in range(100):  # Simulate full gameweek
            player = MockPlayer(
                player_id=1000 + i, 
                position=["GK", "DEF", "MID", "FWD"][i % 4]
            )
            score = MockPlayerScore(
                player_id=1000 + i,
                player_team=f"Team{i % 10}",
                opponent=f"Opponent{(i + 5) % 10}",
                goals=np.random.poisson(0.5),
                assists=np.random.poisson(0.3),
                minutes=np.random.randint(0, 91),
                bonus=np.random.randint(0, 4)
            )
            score.player = player
            scores.append(score)
        
        # Should process without errors
        measurements, noise_matrices = updater.process_batch(
            scores, gameweek=10, season="2023"
        )
        
        assert len(measurements) <= 100
        assert len(noise_matrices) <= 4  # One per position


if __name__ == "__main__":
    pytest.main([__file__, "-v"])