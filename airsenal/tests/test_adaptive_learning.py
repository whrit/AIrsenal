"""
Comprehensive tests for Adaptive Learning Rate System

This test suite covers all components of the adaptive learning rate system:
- AdaptiveLearningRateSystem main controller
- ErrorBasedAdapter for prediction error adjustments
- VolatilityDetector for player volatility analysis
- ConvergenceMonitor for stability tracking
- LearningScheduler for time-based schedules
- All adaptation algorithms (AdaGrad, RMSProp, Adam, AdaDelta, CustomFPL)
- Integration with Kalman filters
- Risk management and safety features
- Edge cases and error handling
"""

import logging
import numpy as np
import pytest
import jax.numpy as jnp
from unittest.mock import Mock, patch
from collections import deque

from airsenal.framework.adaptive_learning import (
    AdaptiveLearningRateSystem,
    AdaptiveLearningConfig,
    PlayerLearningState,
    ErrorBasedAdapter,
    VolatilityDetector,
    ConvergenceMonitor,
    LearningScheduler,
    AdaGradOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    AdaDeltaOptimizer,
    CustomFPLOptimizer,
    AdaptationMethod,
    PlayerPosition
)

# Test configuration
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return AdaptiveLearningConfig(
        base_learning_rate=0.01,
        max_learning_rate=0.1,
        min_learning_rate=1e-6,
        adaptation_method=AdaptationMethod.ADAM,
        volatility_threshold=0.3,
        convergence_patience=5
    )


@pytest.fixture
def sample_player_state():
    """Sample player state for testing."""
    state = PlayerLearningState(
        player_id=123,
        position=PlayerPosition.MIDFIELDER,
        current_learning_rate=0.01
    )
    # Add some sample history
    state.error_history.extend([0.2, 0.3, 0.15, 0.4, 0.25])
    state.prediction_history.extend([5.0, 6.2, 4.8, 7.1, 5.5])
    state.loss_history.extend([0.5, 0.4, 0.35, 0.38, 0.32])
    return state


@pytest.fixture
def adaptive_system(basic_config):
    """Adaptive learning rate system instance."""
    return AdaptiveLearningRateSystem(basic_config)


class TestAdaptiveLearningConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveLearningConfig()
        
        assert config.base_learning_rate == 0.01
        assert config.max_learning_rate == 0.1
        assert config.min_learning_rate == 1e-6
        assert config.adaptation_method == AdaptationMethod.ADAM
        assert config.volatility_threshold == 0.3
        assert config.convergence_patience == 5
        
    def test_position_thresholds(self):
        """Test position-specific error thresholds."""
        config = AdaptiveLearningConfig()
        
        assert PlayerPosition.GOALKEEPER in config.position_error_thresholds
        assert PlayerPosition.DEFENDER in config.position_error_thresholds
        assert PlayerPosition.MIDFIELDER in config.position_error_thresholds
        assert PlayerPosition.FORWARD in config.position_error_thresholds
        
        # Forwards should have higher thresholds (more volatile)
        assert (config.position_error_thresholds[PlayerPosition.FORWARD] > 
                config.position_error_thresholds[PlayerPosition.GOALKEEPER])


class TestPlayerLearningState:
    """Test player learning state class."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        state = PlayerLearningState(
            player_id=456,
            position=PlayerPosition.DEFENDER,
            current_learning_rate=0.02
        )
        
        assert state.player_id == 456
        assert state.position == PlayerPosition.DEFENDER
        assert state.current_learning_rate == 0.02
        assert len(state.error_history) == 0
        assert len(state.prediction_history) == 0
        assert state.volatility_score == 0.0
        assert not state.is_converged
        
    def test_history_limits(self):
        """Test that history deques respect maxlen."""
        state = PlayerLearningState(
            player_id=789,
            position=PlayerPosition.FORWARD,
            current_learning_rate=0.01
        )
        
        # Add more items than maxlen
        for i in range(60):
            state.error_history.append(i)
            
        assert len(state.error_history) == 50  # maxlen=50
        assert state.error_history[0] == 10  # Oldest items dropped


class TestAdaptationAlgorithms:
    """Test all adaptation algorithms."""
    
    def test_adagrad_optimizer(self, basic_config, sample_player_state):
        """Test AdaGrad optimization algorithm."""
        optimizer = AdaGradOptimizer(basic_config)
        
        # Initialize state
        optimizer.initialize_state(sample_player_state)
        assert 'gradient_squared_sum' in sample_player_state.algorithm_state
        assert 'step_count' in sample_player_state.algorithm_state
        
        # Test update
        gradient = jnp.array([0.1, 0.2, 0.15])
        loss = 0.3
        rate = optimizer.update_learning_rate(sample_player_state, gradient, loss, 10)
        
        assert isinstance(rate, float)
        assert basic_config.min_learning_rate <= rate <= basic_config.max_learning_rate
        assert sample_player_state.algorithm_state['step_count'] == 1
        
    def test_rmsprop_optimizer(self, basic_config, sample_player_state):
        """Test RMSProp optimization algorithm."""
        optimizer = RMSPropOptimizer(basic_config)
        
        optimizer.initialize_state(sample_player_state)
        assert 'exponential_avg_squared' in sample_player_state.algorithm_state
        
        gradient = jnp.array([0.05, 0.1, 0.08])
        loss = 0.25
        rate = optimizer.update_learning_rate(sample_player_state, gradient, loss, 15)
        
        assert isinstance(rate, float)
        assert basic_config.min_learning_rate <= rate <= basic_config.max_learning_rate
        
    def test_adam_optimizer(self, basic_config, sample_player_state):
        """Test Adam optimization algorithm."""
        optimizer = AdamOptimizer(basic_config)
        
        optimizer.initialize_state(sample_player_state)
        assert 'momentum' in sample_player_state.algorithm_state
        assert 'velocity' in sample_player_state.algorithm_state
        
        gradient = jnp.array([0.08, 0.12, 0.1])
        loss = 0.28
        rate = optimizer.update_learning_rate(sample_player_state, gradient, loss, 20)
        
        assert isinstance(rate, float)
        assert basic_config.min_learning_rate <= rate <= basic_config.max_learning_rate
        
        # Test bias correction (should improve over steps)
        rate2 = optimizer.update_learning_rate(sample_player_state, gradient, loss, 21)
        assert sample_player_state.algorithm_state['step_count'] == 2
        
    def test_adadelta_optimizer(self, basic_config, sample_player_state):
        """Test AdaDelta optimization algorithm."""
        optimizer = AdaDeltaOptimizer(basic_config)
        
        optimizer.initialize_state(sample_player_state)
        assert 'exponential_avg_squared_grad' in sample_player_state.algorithm_state
        assert 'exponential_avg_squared_delta' in sample_player_state.algorithm_state
        
        gradient = jnp.array([0.06, 0.09, 0.07])
        loss = 0.22
        rate = optimizer.update_learning_rate(sample_player_state, gradient, loss, 25)
        
        assert isinstance(rate, float)
        assert basic_config.min_learning_rate <= rate <= basic_config.max_learning_rate
        
    def test_custom_fpl_optimizer(self, basic_config, sample_player_state):
        """Test custom FPL optimization algorithm."""
        optimizer = CustomFPLOptimizer(basic_config)
        
        optimizer.initialize_state(sample_player_state)
        assert 'form_momentum' in sample_player_state.algorithm_state
        assert 'consistency_factor' in sample_player_state.algorithm_state
        
        gradient = jnp.array([0.04, 0.07, 0.05])
        loss = 0.18
        rate = optimizer.update_learning_rate(sample_player_state, gradient, loss, 30)
        
        assert isinstance(rate, float)
        assert basic_config.min_learning_rate <= rate <= basic_config.max_learning_rate
        
        # Test position-specific adjustments
        sample_player_state.position = PlayerPosition.FORWARD
        rate_forward = optimizer.update_learning_rate(sample_player_state, gradient, loss, 31)
        
        sample_player_state.position = PlayerPosition.GOALKEEPER
        rate_gk = optimizer.update_learning_rate(sample_player_state, gradient, loss, 32)
        
        # Forwards should generally have higher rates than goalkeepers
        assert rate_forward > rate_gk


class TestErrorBasedAdapter:
    """Test error-based adaptation component."""
    
    def test_basic_error_adjustment(self, basic_config):
        """Test basic error-based rate adjustment."""
        adapter = ErrorBasedAdapter(basic_config)
        
        base_rate = 0.01
        prediction_error = 0.5
        actual_value = 5.0
        predicted_value = 4.5
        position = PlayerPosition.MIDFIELDER
        
        adjusted_rate = adapter.adjust_for_error(
            base_rate, prediction_error, actual_value, predicted_value, position
        )
        
        assert adjusted_rate > base_rate  # Error should increase rate
        assert basic_config.min_learning_rate <= adjusted_rate <= basic_config.max_learning_rate
        
    def test_asymmetric_adjustment(self, basic_config):
        """Test asymmetric over/under prediction adjustments."""
        basic_config.asymmetric_adjustment = True
        adapter = ErrorBasedAdapter(basic_config)
        
        base_rate = 0.01
        position = PlayerPosition.MIDFIELDER
        
        # Over-prediction case
        over_rate = adapter.adjust_for_error(
            base_rate, 0.5, 4.0, 4.5, position
        )
        
        # Under-prediction case  
        under_rate = adapter.adjust_for_error(
            base_rate, 0.5, 5.0, 4.5, position
        )
        
        # Under-predictions should get larger adjustments
        assert under_rate > over_rate
        
    def test_position_specific_thresholds(self, basic_config):
        """Test position-specific error thresholds."""
        adapter = ErrorBasedAdapter(basic_config)
        
        base_rate = 0.01
        prediction_error = 1.0
        actual_value = 5.0
        predicted_value = 4.0
        
        # Test different positions
        gk_rate = adapter.adjust_for_error(
            base_rate, prediction_error, actual_value, predicted_value,
            PlayerPosition.GOALKEEPER
        )
        
        fwd_rate = adapter.adjust_for_error(
            base_rate, prediction_error, actual_value, predicted_value,
            PlayerPosition.FORWARD
        )
        
        # Same error should have bigger impact on GK (lower threshold)
        assert gk_rate > fwd_rate
        
    def test_windowed_error_metrics(self, basic_config, sample_player_state):
        """Test windowed error metrics computation."""
        adapter = ErrorBasedAdapter(basic_config)
        
        metrics = adapter.compute_windowed_error_metrics(sample_player_state, window_size=5)
        
        assert 'mean_error' in metrics
        assert 'error_std' in metrics
        assert 'error_trend' in metrics
        assert 'max_error' in metrics
        
        assert isinstance(metrics['mean_error'], float)
        assert metrics['mean_error'] > 0  # Should have positive mean error
        
    def test_empty_history_handling(self, basic_config):
        """Test handling of empty error history."""
        adapter = ErrorBasedAdapter(basic_config)
        
        empty_state = PlayerLearningState(
            player_id=999,
            position=PlayerPosition.DEFENDER,
            current_learning_rate=0.01
        )
        
        metrics = adapter.compute_windowed_error_metrics(empty_state)
        
        assert metrics['mean_error'] == 0.0
        assert metrics['error_std'] == 0.0
        assert metrics['error_trend'] == 0.0
        assert metrics['max_error'] == 0.0


class TestVolatilityDetector:
    """Test volatility detection component."""
    
    def test_volatility_score_calculation(self, basic_config, sample_player_state):
        """Test volatility score computation."""
        detector = VolatilityDetector(basic_config)
        
        volatility = detector.compute_volatility_score(sample_player_state)
        
        assert isinstance(volatility, float)
        assert 0.0 <= volatility <= 1.0
        
    def test_empty_history_volatility(self, basic_config):
        """Test volatility calculation with empty history."""
        detector = VolatilityDetector(basic_config)
        
        empty_state = PlayerLearningState(
            player_id=888,
            position=PlayerPosition.MIDFIELDER,
            current_learning_rate=0.01
        )
        
        volatility = detector.compute_volatility_score(empty_state)
        assert volatility == 0.0
        
    def test_regime_change_detection(self, basic_config):
        """Test regime change detection."""
        detector = VolatilityDetector(basic_config)
        
        # Create state with clear regime change
        state = PlayerLearningState(
            player_id=777,
            position=PlayerPosition.FORWARD,
            current_learning_rate=0.01
        )
        
        # First period: low performance
        for i in range(4):
            state.prediction_history.append(2.0 + 0.1 * i)
            
        # Second period: high performance  
        for i in range(4):
            state.prediction_history.append(8.0 + 0.1 * i)
            
        regime_change = detector.detect_regime_change(state, 20)
        assert regime_change == True
        assert state.last_regime_change == 20
        
    def test_no_regime_change(self, basic_config):
        """Test no regime change detection for stable performance."""
        detector = VolatilityDetector(basic_config)
        
        stable_state = PlayerLearningState(
            player_id=666,
            position=PlayerPosition.DEFENDER,
            current_learning_rate=0.01
        )
        
        # Stable performance around 5.0
        for i in range(8):
            stable_state.prediction_history.append(5.0 + 0.05 * (i % 2))
            
        regime_change = detector.detect_regime_change(stable_state, 25)
        assert regime_change == False
        
    def test_outlier_identification(self, basic_config, sample_player_state):
        """Test outlier identification."""
        detector = VolatilityDetector(basic_config)
        
        # Add clear outlier
        sample_player_state.prediction_history.append(15.0)  # Much higher than others
        
        outliers = detector.identify_outliers(sample_player_state, z_threshold=2.0)
        
        assert isinstance(outliers, list)
        assert len(outliers) > 0  # Should detect the outlier
        
    def test_no_outliers(self, basic_config):
        """Test no outliers in stable data."""
        detector = VolatilityDetector(basic_config)
        
        stable_state = PlayerLearningState(
            player_id=555,
            position=PlayerPosition.MIDFIELDER, 
            current_learning_rate=0.01
        )
        
        # Very stable data
        for i in range(10):
            stable_state.prediction_history.append(5.0)
            
        outliers = detector.identify_outliers(stable_state)
        assert len(outliers) == 0


class TestConvergenceMonitor:
    """Test convergence monitoring component."""
    
    def test_convergence_detection(self, basic_config):
        """Test convergence detection."""
        monitor = ConvergenceMonitor(basic_config)
        
        converging_state = PlayerLearningState(
            player_id=444,
            position=PlayerPosition.GOALKEEPER,
            current_learning_rate=0.01
        )
        
        # Simulate convergence with decreasing losses
        converged = False
        for loss in [0.5, 0.4, 0.35, 0.33, 0.32, 0.31]:
            converged = monitor.check_convergence(converging_state, loss)
            
        assert converged == True
        assert converging_state.is_converged == True
        
    def test_no_convergence(self, basic_config):
        """Test no convergence with oscillating losses."""
        monitor = ConvergenceMonitor(basic_config)
        
        oscillating_state = PlayerLearningState(
            player_id=333,
            position=PlayerPosition.MIDFIELDER,
            current_learning_rate=0.01
        )
        
        # Oscillating losses
        converged = False
        for loss in [0.5, 0.3, 0.6, 0.2, 0.7, 0.1]:
            converged = monitor.check_convergence(oscillating_state, loss)
            
        assert converged == False
        
    def test_oscillation_detection(self, basic_config):
        """Test oscillation detection."""
        monitor = ConvergenceMonitor(basic_config)
        
        oscillating_state = PlayerLearningState(
            player_id=222,
            position=PlayerPosition.FORWARD,
            current_learning_rate=0.01
        )
        
        # Create clear oscillation pattern
        oscillating_state.loss_history.extend([0.5, 0.3, 0.6, 0.2, 0.7, 0.1])
        
        is_oscillating = monitor.detect_oscillation(oscillating_state)
        assert is_oscillating == True
        
    def test_divergence_detection(self, basic_config):
        """Test divergence detection."""
        monitor = ConvergenceMonitor(basic_config)
        
        diverging_state = PlayerLearningState(
            player_id=111,
            position=PlayerPosition.DEFENDER,
            current_learning_rate=0.01
        )
        
        # Create diverging pattern
        diverging_state.loss_history.extend([0.3, 0.32, 0.35, 1.5])  # Big jump at end
        
        is_diverging = monitor.detect_divergence(diverging_state)
        assert is_diverging == True
        
    def test_convergence_metrics(self, basic_config, sample_player_state):
        """Test convergence metrics computation."""
        monitor = ConvergenceMonitor(basic_config)
        
        metrics = monitor.get_convergence_metrics(sample_player_state)
        
        assert 'is_converged' in metrics
        assert 'is_oscillating' in metrics
        assert 'is_diverging' in metrics
        assert 'loss_trend' in metrics
        assert 'loss_variance' in metrics
        assert 'patience_counter' in metrics
        
        assert isinstance(metrics['is_converged'], bool)
        assert isinstance(metrics['loss_trend'], float)
        assert isinstance(metrics['loss_variance'], float)


class TestLearningScheduler:
    """Test learning rate scheduling component."""
    
    def test_exponential_decay(self, basic_config):
        """Test exponential decay schedule."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        rates = []
        
        for gameweek in [1, 10, 20, 30]:
            rate = scheduler.exponential_decay(base_rate, gameweek)
            rates.append(rate)
            
        # Rates should be decreasing
        assert all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
        assert rates[0] == base_rate  # First rate should be base rate
        
    def test_step_decay(self, basic_config):
        """Test step decay schedule."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        
        # Test before and after step
        rate_before = scheduler.step_decay(base_rate, 9, step_size=10)
        rate_after = scheduler.step_decay(base_rate, 10, step_size=10)
        
        assert rate_before > rate_after
        
    def test_cosine_annealing(self, basic_config):
        """Test cosine annealing schedule."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        min_rate = 0.001
        
        # Test at different points in cosine cycle
        rate_start = scheduler.cosine_annealing(base_rate, 0, 38, min_rate)
        rate_mid = scheduler.cosine_annealing(base_rate, 19, 38, min_rate)
        rate_end = scheduler.cosine_annealing(base_rate, 38, 38, min_rate)
        
        assert abs(rate_start - base_rate) < 1e-6  # Should be close to base at start
        assert abs(rate_end - min_rate) < 1e-6    # Should be close to min at end
        assert rate_mid < base_rate and rate_mid > min_rate  # Should be between
        
    def test_warm_restart(self, basic_config):
        """Test warm restart schedule."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        
        # Test restart behavior
        rate_end_cycle1 = scheduler.warm_restart(base_rate, 9, restart_period=10)
        rate_start_cycle2 = scheduler.warm_restart(base_rate, 10, restart_period=10)
        
        # Rate should increase at restart
        assert rate_start_cycle2 > rate_end_cycle1
        
    def test_custom_fpl_schedule(self, basic_config):
        """Test custom FPL schedule."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        
        # Test busy period adjustment
        normal_rate = scheduler.custom_fpl_schedule(base_rate, 10)
        busy_rate = scheduler.custom_fpl_schedule(base_rate, 16)  # Busy period
        
        assert busy_rate > normal_rate
        
        # Test early season adjustment
        early_rate = scheduler.custom_fpl_schedule(base_rate, 5)
        late_rate = scheduler.custom_fpl_schedule(base_rate, 35)
        
        assert early_rate > late_rate
        
    def test_schedule_selection(self, basic_config):
        """Test schedule type selection."""
        scheduler = LearningScheduler(basic_config)
        
        base_rate = 0.01
        gameweek = 15
        
        # Test each schedule type
        for schedule_type in ["exponential", "step", "cosine", "warm_restart", "custom_fpl"]:
            rate = scheduler.get_scheduled_rate(base_rate, gameweek, schedule_type)
            assert isinstance(rate, float)
            assert rate > 0
            
        # Test unknown schedule type
        rate = scheduler.get_scheduled_rate(base_rate, gameweek, "unknown")
        assert rate == base_rate  # Should fallback to base rate


class TestAdaptiveLearningRateSystem:
    """Test main adaptive learning rate system."""
    
    def test_system_initialization(self, basic_config):
        """Test system initialization."""
        system = AdaptiveLearningRateSystem(basic_config)
        
        assert system.config == basic_config
        assert system.error_adapter is not None
        assert system.volatility_detector is not None
        assert system.convergence_monitor is not None
        assert system.scheduler is not None
        assert system.adaptation_algorithm is not None
        assert len(system.player_states) == 0
        
    def test_player_state_creation(self, adaptive_system):
        """Test player state creation and retrieval."""
        player_id = 12345
        position = PlayerPosition.MIDFIELDER
        gameweek = 15
        
        # First call should create state
        state1 = adaptive_system.get_or_create_player_state(player_id, position, gameweek)
        assert state1.player_id == player_id
        assert state1.position == position
        assert state1.created_at == gameweek
        
        # Second call should return same state
        state2 = adaptive_system.get_or_create_player_state(player_id, position, gameweek + 1)
        assert state1 is state2
        
    def test_learning_rate_adaptation(self, adaptive_system):
        """Test complete learning rate adaptation process."""
        player_id = 54321
        position = PlayerPosition.FORWARD
        prediction_error = 0.8
        actual_value = 6.0
        predicted_value = 5.2
        gameweek = 20
        gradient = jnp.array([0.1, 0.15, 0.12])
        loss = 0.4
        
        # First adaptation
        new_rate = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=prediction_error,
            actual_value=actual_value,
            predicted_value=predicted_value,
            gameweek=gameweek,
            gradient=gradient,
            loss=loss
        )
        
        assert isinstance(new_rate, float)
        assert adaptive_system.config.min_learning_rate <= new_rate <= adaptive_system.config.max_learning_rate
        
        # Check that player state was created and updated
        assert player_id in adaptive_system.player_states
        state = adaptive_system.player_states[player_id]
        assert len(state.error_history) == 1
        assert len(state.prediction_history) == 1
        assert state.current_learning_rate == new_rate
        
    def test_adaptation_without_gradient(self, adaptive_system):
        """Test adaptation without gradient information."""
        player_id = 98765
        position = PlayerPosition.GOALKEEPER
        prediction_error = 0.3
        actual_value = 3.0
        predicted_value = 3.3
        gameweek = 10
        
        new_rate = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=prediction_error,
            actual_value=actual_value,
            predicted_value=predicted_value,
            gameweek=gameweek
        )
        
        assert isinstance(new_rate, float)
        assert new_rate > 0
        
    def test_convergence_monitoring(self, adaptive_system):
        """Test convergence monitoring integration."""
        player_id = 11111
        position = PlayerPosition.DEFENDER
        
        # Create player state first
        adaptive_system.get_or_create_player_state(player_id, position, 1)
        
        # Test convergence checking
        for loss in [0.5, 0.4, 0.35, 0.33, 0.32]:
            converged = adaptive_system.check_convergence(player_id, loss)
            
        assert isinstance(converged, bool)
        
        # Get convergence metrics
        metrics = adaptive_system.get_convergence_metrics(player_id)
        assert 'is_converged' in metrics
        
    def test_current_learning_rate_retrieval(self, adaptive_system):
        """Test current learning rate retrieval."""
        player_id = 22222
        position = PlayerPosition.MIDFIELDER
        
        # Before creating state
        rate_before = adaptive_system.get_current_learning_rate(player_id)
        assert rate_before == adaptive_system.config.base_learning_rate
        
        # After creating state
        adaptive_system.get_or_create_player_state(player_id, position, 1)
        rate_after = adaptive_system.get_current_learning_rate(player_id)
        assert rate_after == adaptive_system.config.base_learning_rate
        
    def test_player_volatility_retrieval(self, adaptive_system):
        """Test player volatility retrieval."""
        player_id = 33333
        position = PlayerPosition.FORWARD
        
        # Before creating state
        volatility_before = adaptive_system.get_player_volatility(player_id)
        assert volatility_before == 0.0
        
        # After creating and updating state
        adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=0.5,
            actual_value=4.0,
            predicted_value=4.5,
            gameweek=5
        )
        
        volatility_after = adaptive_system.get_player_volatility(player_id)
        assert isinstance(volatility_after, float)
        assert volatility_after >= 0.0
        
    def test_global_statistics(self, adaptive_system):
        """Test global statistics tracking."""
        stats_initial = adaptive_system.get_global_statistics()
        
        assert 'total_players' in stats_initial
        assert 'total_adaptations' in stats_initial
        assert 'adaptation_algorithm' in stats_initial
        assert stats_initial['total_players'] == 0
        assert stats_initial['total_adaptations'] == 0
        
        # Perform some adaptations
        for i in range(3):
            adaptive_system.adapt_learning_rate(
                player_id=i,
                position=PlayerPosition.MIDFIELDER,
                prediction_error=0.2,
                actual_value=5.0,
                predicted_value=4.8,
                gameweek=10
            )
            
        stats_after = adaptive_system.get_global_statistics()
        assert stats_after['total_players'] == 3
        assert stats_after['total_adaptations'] == 3
        
    def test_state_reset(self, adaptive_system):
        """Test player state reset."""
        player_id = 44444
        position = PlayerPosition.DEFENDER
        
        # Create state
        adaptive_system.get_or_create_player_state(player_id, position, 1)
        assert player_id in adaptive_system.player_states
        
        # Reset state
        adaptive_system.reset_player_state(player_id)
        assert player_id not in adaptive_system.player_states
        
    def test_state_persistence(self, adaptive_system):
        """Test state saving and loading."""
        # Create some player states
        for i in range(2):
            adaptive_system.adapt_learning_rate(
                player_id=i,
                position=PlayerPosition.MIDFIELDER,
                prediction_error=0.3,
                actual_value=4.0,
                predicted_value=4.3,
                gameweek=15
            )
            
        # Save state
        saved_state = adaptive_system.save_state()
        
        assert 'config' in saved_state
        assert 'player_states' in saved_state
        assert 'global_stats' in saved_state
        
        # Create new system and load state
        new_system = AdaptiveLearningRateSystem()
        new_system.load_state(saved_state)
        
        assert len(new_system.player_states) == 2
        assert new_system.global_stats['total_adaptations'] > 0
        
    def test_kalman_filter_integration(self, adaptive_system):
        """Test Kalman filter integration."""
        player_id = 55555
        position = PlayerPosition.MIDFIELDER
        
        # Create player state with some history
        adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=0.4,
            actual_value=5.0,
            predicted_value=5.4,
            gameweek=20
        )
        
        # Test integration
        kalman_gain = jnp.array([[0.1, 0.2], [0.15, 0.25]])
        process_noise = jnp.array([[0.01, 0.005], [0.005, 0.01]])
        measurement_noise = jnp.array([[0.1, 0.05], [0.05, 0.1]])
        
        adjusted_gain, adjusted_process, adjusted_measurement = adaptive_system.integrate_with_kalman_filter(
            player_id, kalman_gain, process_noise, measurement_noise
        )
        
        assert adjusted_gain.shape == kalman_gain.shape
        assert adjusted_process.shape == process_noise.shape
        assert adjusted_measurement.shape == measurement_noise.shape
        
        # Values should be adjusted based on learning rate and volatility
        assert not jnp.allclose(adjusted_gain, kalman_gain)
        

class TestRiskManagement:
    """Test risk management and safety features."""
    
    def test_learning_rate_bounds(self, adaptive_system):
        """Test learning rate bounds enforcement."""
        player_id = 77777
        position = PlayerPosition.FORWARD
        
        # Test with very high error (should be capped)
        high_error_rate = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=10.0,  # Very high error
            actual_value=1.0,
            predicted_value=11.0,
            gameweek=1
        )
        
        assert high_error_rate <= adaptive_system.config.max_learning_rate
        
    def test_divergence_handling(self, adaptive_system):
        """Test divergence detection and handling."""
        player_id = 88888
        position = PlayerPosition.MIDFIELDER
        
        # Create player state
        state = adaptive_system.get_or_create_player_state(player_id, position, 1)
        
        # Simulate diverging losses
        state.loss_history.extend([0.3, 0.32, 0.35, 1.5])  # Big jump
        
        # Next adaptation should reduce rate due to divergence
        rate_after_divergence = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=0.2,
            actual_value=5.0,
            predicted_value=4.8,
            gameweek=10
        )
        
        # Rate should be reduced from base rate
        assert rate_after_divergence < adaptive_system.config.base_learning_rate
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = AdaptiveLearningConfig(numerical_stability_eps=1e-8)
        system = AdaptiveLearningRateSystem(config)
        
        # Test with zero gradient
        state = PlayerLearningState(
            player_id=99999,
            position=PlayerPosition.GOALKEEPER,
            current_learning_rate=0.01
        )
        
        optimizer = AdamOptimizer(config)
        optimizer.initialize_state(state)
        
        zero_gradient = jnp.zeros(3)
        rate = optimizer.update_learning_rate(state, zero_gradient, 0.1, 1)
        
        assert jnp.isfinite(rate)
        assert rate > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_algorithm_methods(self, basic_config):
        """Test adaptation methods with different algorithms."""
        for method in AdaptationMethod:
            config = AdaptiveLearningConfig(adaptation_method=method)
            system = AdaptiveLearningRateSystem(config)
            
            assert system.adaptation_algorithm is not None
            assert system.adaptation_algorithm.get_name() is not None
            
    def test_invalid_player_positions(self, adaptive_system):
        """Test handling of all player positions."""
        for position in PlayerPosition:
            rate = adaptive_system.adapt_learning_rate(
                player_id=1000 + position.value.__hash__(),
                position=position,
                prediction_error=0.3,
                actual_value=4.0,
                predicted_value=4.3,
                gameweek=10
            )
            
            assert isinstance(rate, float)
            assert rate > 0
            
    def test_extreme_gameweeks(self, adaptive_system):
        """Test with extreme gameweek values."""
        player_id = 99998
        position = PlayerPosition.MIDFIELDER
        
        # Very early gameweek
        rate_early = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=0.3,
            actual_value=4.0,
            predicted_value=4.3,
            gameweek=1
        )
        
        # Very late gameweek
        rate_late = adaptive_system.adapt_learning_rate(
            player_id=player_id,
            position=position,
            prediction_error=0.3,
            actual_value=4.0,
            predicted_value=4.3,
            gameweek=100
        )
        
        assert isinstance(rate_early, float)
        assert isinstance(rate_late, float)
        
    def test_repeated_adaptations(self, adaptive_system):
        """Test system stability with many repeated adaptations."""
        player_id = 99997
        position = PlayerPosition.DEFENDER
        
        rates = []
        for i in range(50):
            rate = adaptive_system.adapt_learning_rate(
                player_id=player_id,
                position=position,
                prediction_error=0.2 + 0.01 * (i % 5),
                actual_value=5.0,
                predicted_value=5.2,
                gameweek=i + 1
            )
            rates.append(rate)
            
        # All rates should be valid
        assert all(isinstance(rate, float) and rate > 0 for rate in rates)
        
        # System should remain stable
        final_stats = adaptive_system.get_global_statistics()
        assert final_stats['total_adaptations'] == 50


if __name__ == "__main__":
    pytest.main([__file__])