"""
Comprehensive tests for Kalman Filter implementation

This test suite covers all aspects of the Kalman filter implementation:
- Basic filter functionality (predict/update cycles)
- Numerical stability features
- Different filter types (Linear, Extended, Unscented)
- Error handling and edge cases
- Integration with player modeling
- Performance characteristics
"""

import logging
import numpy as np
import pytest
import jax.numpy as jnp
from unittest.mock import Mock, patch

from airsenal.framework.kalman_filter import (
    FilterConfig,
    FilterState,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    AdaptiveNoiseEstimator,
    KalmanFilterError,
    NumericalInstabilityError,
    InvalidCovarianceError,
    create_player_ability_config,
    create_simple_linear_filter,
    create_player_transition_function,
    create_player_measurement_function
)

from airsenal.framework.kalman_player_model import (
    KalmanPlayerModel,
    PositionSpecificKalmanModel,
    KalmanPlayerModelError
)

from airsenal.framework.adaptive_player_model import (
    StateSpaceConfig,
    PlayerState
)

# Test configuration
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def basic_config():
    """Basic filter configuration for testing."""
    return FilterConfig(
        state_dim=4,
        obs_dim=4,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
        use_joseph_form=True,
        enable_adaptive_noise=False  # Disable for deterministic tests
    )


@pytest.fixture
def test_state():
    """Sample filter state for testing."""
    return FilterState(
        state_mean=jnp.array([0.5, 0.6, 0.7, 0.1]),
        state_cov=jnp.eye(4) * 0.1,
        timestamp=1.0,
        gameweek=1,
        season="2023",
        player_id=123
    )


@pytest.fixture
def test_observation():
    """Sample observation for testing."""
    return jnp.array([1.0, 0.5, 90.0, 2.0])


@pytest.fixture
def linear_filter(basic_config):
    """Linear Kalman filter for testing."""
    return KalmanFilter(basic_config)


@pytest.fixture
def extended_filter(basic_config):
    """Extended Kalman filter for testing."""
    transition_fn = create_player_transition_function()
    measurement_fn = create_player_measurement_function()
    return ExtendedKalmanFilter(basic_config, transition_fn, measurement_fn)


@pytest.fixture
def unscented_filter(basic_config):
    """Unscented Kalman filter for testing."""
    transition_fn = create_player_transition_function()
    measurement_fn = create_player_measurement_function()
    return UnscentedKalmanFilter(basic_config, transition_fn, measurement_fn)


class TestFilterConfig:
    """Test FilterConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FilterConfig()
        assert config.state_dim == 4
        assert config.obs_dim == 4
        assert config.process_noise_std == 0.1
        assert config.measurement_noise_std == 0.2
        assert config.use_joseph_form is True
        assert config.enable_adaptive_noise is True
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            FilterConfig(state_dim=0)
        
        with pytest.raises(ValueError, match="Dimensions must be positive"):
            FilterConfig(obs_dim=-1)
    
    def test_invalid_noise_parameters(self):
        """Test validation of noise parameters."""
        with pytest.raises(ValueError, match="Noise standard deviations must be positive"):
            FilterConfig(process_noise_std=0)
        
        with pytest.raises(ValueError, match="Noise standard deviations must be positive"):
            FilterConfig(measurement_noise_std=-0.1)
    
    def test_invalid_adaptation_rate(self):
        """Test validation of adaptation rate."""
        with pytest.raises(ValueError, match="Adaptation rate must be in"):
            FilterConfig(adaptation_rate=0)
        
        with pytest.raises(ValueError, match="Adaptation rate must be in"):
            FilterConfig(adaptation_rate=1.0)
    
    def test_ukf_parameters(self):
        """Test UKF parameter validation."""
        with pytest.raises(ValueError, match="UKF parameters alpha must be positive"):
            FilterConfig(alpha=0)
        
        with pytest.raises(ValueError, match="beta non-negative"):
            FilterConfig(beta=-1.0)


class TestFilterState:
    """Test FilterState class."""
    
    def test_state_creation(self, test_state):
        """Test creating filter state."""
        assert test_state.player_id == 123
        assert test_state.gameweek == 1
        assert test_state.season == "2023"
        assert test_state.state_mean.shape == (4,)
        assert test_state.state_cov.shape == (4, 4)
    
    def test_state_copy(self, test_state):
        """Test state copying."""
        copied_state = test_state.copy()
        
        # Should be equal but different objects
        assert jnp.allclose(copied_state.state_mean, test_state.state_mean)
        assert jnp.allclose(copied_state.state_cov, test_state.state_cov)
        assert copied_state.player_id == test_state.player_id
        
        # Modification should not affect original
        copied_state.state_mean = copied_state.state_mean.at[0].set(0.0)
        assert not jnp.allclose(copied_state.state_mean, test_state.state_mean)


class TestAdaptiveNoiseEstimator:
    """Test adaptive noise estimation."""
    
    def test_noise_estimator_initialization(self, basic_config):
        """Test noise estimator initialization."""
        estimator = AdaptiveNoiseEstimator(basic_config)
        assert estimator.adaptation_rate == basic_config.adaptation_rate
        assert estimator.window_size == basic_config.noise_estimation_window
        assert estimator.process_noise_cov.shape == (4, 4)
        assert estimator.measurement_noise_cov.shape == (4, 4)
    
    def test_process_noise_update(self, basic_config):
        """Test process noise estimation update."""
        estimator = AdaptiveNoiseEstimator(basic_config)
        
        # Simulate multiple updates
        for i in range(5):
            predicted_state = jnp.array([0.5, 0.6, 0.7, 0.1])
            updated_state = predicted_state + 0.01 * np.random.randn(4)
            
            updated_cov = estimator.update_process_noise(predicted_state, updated_state, 1.0)
            assert updated_cov.shape == (4, 4)
            assert jnp.all(jnp.linalg.eigvals(updated_cov) >= 0)  # Should be PSD
    
    def test_measurement_noise_update(self, basic_config):
        """Test measurement noise estimation update."""
        estimator = AdaptiveNoiseEstimator(basic_config)
        
        # Simulate multiple updates
        for i in range(5):
            innovation = jnp.array([0.1, 0.05, 5.0, 0.2])
            innovation_cov = jnp.eye(4) * 0.1
            
            updated_cov = estimator.update_measurement_noise(innovation, innovation_cov)
            assert updated_cov.shape == (4, 4)
            assert jnp.all(jnp.linalg.eigvals(updated_cov) >= 0)  # Should be PSD


class TestLinearKalmanFilter:
    """Test linear Kalman filter."""
    
    def test_filter_initialization(self, linear_filter):
        """Test filter initialization."""
        assert linear_filter.state_dim == 4
        assert linear_filter.obs_dim == 4
        assert linear_filter.F.shape == (4, 4)
        assert linear_filter.H.shape == (4, 4)
        assert linear_filter.Q.shape == (4, 4)
        assert linear_filter.R.shape == (4, 4)
    
    def test_prediction_step(self, linear_filter, test_state):
        """Test prediction step."""
        predicted_state = linear_filter.predict(test_state, dt=1.0)
        
        # Check outputs
        assert predicted_state.state_mean.shape == (4,)
        assert predicted_state.state_cov.shape == (4, 4)
        assert predicted_state.timestamp == test_state.timestamp + 1.0
        assert predicted_state.gameweek == test_state.gameweek
        assert predicted_state.player_id == test_state.player_id
        
        # Covariance should increase (uncertainty grows)
        assert jnp.trace(predicted_state.state_cov) >= jnp.trace(test_state.state_cov)
        
        # Check numerical stability
        assert jnp.all(jnp.isfinite(predicted_state.state_mean))
        assert jnp.all(jnp.isfinite(predicted_state.state_cov))
        assert jnp.all(jnp.linalg.eigvals(predicted_state.state_cov) >= -1e-10)
    
    def test_update_step(self, linear_filter, test_state, test_observation):
        """Test update step."""
        # First predict
        predicted_state = linear_filter.predict(test_state, dt=1.0)
        
        # Then update
        updated_state = linear_filter.update(predicted_state, test_observation)
        
        # Check outputs
        assert updated_state.state_mean.shape == (4,)
        assert updated_state.state_cov.shape == (4, 4)
        assert updated_state.innovation is not None
        assert updated_state.innovation_cov is not None
        assert updated_state.kalman_gain is not None
        assert updated_state.log_likelihood is not None
        
        # Covariance should decrease (uncertainty reduces)
        assert jnp.trace(updated_state.state_cov) <= jnp.trace(predicted_state.state_cov)
        
        # Check numerical stability
        assert jnp.all(jnp.isfinite(updated_state.state_mean))
        assert jnp.all(jnp.isfinite(updated_state.state_cov))
        assert jnp.all(jnp.linalg.eigvals(updated_state.state_cov) >= -1e-10)
    
    def test_full_cycle(self, linear_filter, test_state, test_observation):
        """Test full predict-update cycle."""
        # Multiple cycles
        current_state = test_state
        
        for i in range(5):
            # Predict
            predicted_state = linear_filter.predict(current_state, dt=1.0)
            
            # Update with noisy observation
            obs = test_observation + 0.1 * np.random.randn(4)
            updated_state = linear_filter.update(predicted_state, obs)
            
            # Verify state evolution
            assert jnp.all(jnp.isfinite(updated_state.state_mean))
            assert jnp.all(jnp.isfinite(updated_state.state_cov))
            assert updated_state.log_likelihood is not None
            
            current_state = updated_state
    
    def test_joseph_form_vs_standard(self, basic_config, test_state, test_observation):
        """Test Joseph form vs standard covariance update."""
        # Joseph form
        config_joseph = basic_config
        config_joseph.use_joseph_form = True
        filter_joseph = KalmanFilter(config_joseph)
        
        # Standard form
        config_standard = FilterConfig(
            state_dim=basic_config.state_dim,
            obs_dim=basic_config.obs_dim,
            process_noise_std=basic_config.process_noise_std,
            measurement_noise_std=basic_config.measurement_noise_std,
            use_joseph_form=False
        )
        filter_standard = KalmanFilter(config_standard)
        
        # Same initial state
        predicted_state = filter_joseph.predict(test_state, dt=1.0)
        
        # Update with both methods
        updated_joseph = filter_joseph.update(predicted_state, test_observation)
        updated_standard = filter_standard.update(predicted_state, test_observation)
        
        # Results should be similar but Joseph form more stable
        assert jnp.allclose(updated_joseph.state_mean, updated_standard.state_mean, rtol=1e-6)
        
        # Joseph form should be more numerically stable
        joseph_eigs = jnp.linalg.eigvals(updated_joseph.state_cov)
        standard_eigs = jnp.linalg.eigvals(updated_standard.state_cov)
        
        assert jnp.min(joseph_eigs) >= jnp.min(standard_eigs)  # Joseph form more stable
    
    def test_singular_innovation_covariance(self, linear_filter, test_state):
        """Test handling of singular innovation covariance."""
        # Create observation with zero noise in one dimension
        H_singular = jnp.array([[1, 0, 0, 0],
                               [1, 0, 0, 0],  # Same as first row - singular
                               [0, 1, 0, 0],
                               [0, 0, 1, 0]])
        
        # Create filter with singular observation matrix
        linear_filter.H = H_singular
        
        predicted_state = linear_filter.predict(test_state, dt=1.0)
        observation = jnp.array([0.5, 0.5, 0.6, 0.7])  # Consistent observation
        
        # Should handle singular case gracefully
        updated_state = linear_filter.update(predicted_state, observation)
        
        assert jnp.all(jnp.isfinite(updated_state.state_mean))
        assert jnp.all(jnp.isfinite(updated_state.state_cov))


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter."""
    
    def test_ekf_initialization(self, extended_filter):
        """Test EKF initialization."""
        assert hasattr(extended_filter, 'transition_fn')
        assert hasattr(extended_filter, 'measurement_fn')
        assert hasattr(extended_filter, 'transition_jacobian_fn')
        assert hasattr(extended_filter, 'measurement_jacobian_fn')
    
    def test_ekf_prediction(self, extended_filter, test_state):
        """Test EKF prediction with non-linear dynamics."""
        predicted_state = extended_filter.predict(test_state, dt=1.0)
        
        # Check basic properties
        assert predicted_state.state_mean.shape == (4,)
        assert predicted_state.state_cov.shape == (4, 4)
        assert jnp.all(jnp.isfinite(predicted_state.state_mean))
        assert jnp.all(jnp.isfinite(predicted_state.state_cov))
        
        # Should evolve differently from linear case
        linear_filter = create_simple_linear_filter(extended_filter.config)
        linear_predicted = linear_filter.predict(test_state, dt=1.0)
        
        # Non-linear dynamics should create different evolution
        assert not jnp.allclose(predicted_state.state_mean, linear_predicted.state_mean, rtol=1e-3)
    
    def test_ekf_update(self, extended_filter, test_state, test_observation):
        """Test EKF update with non-linear measurements."""
        predicted_state = extended_filter.predict(test_state, dt=1.0)
        updated_state = extended_filter.update(predicted_state, test_observation)
        
        # Check outputs
        assert updated_state.innovation is not None
        assert updated_state.kalman_gain is not None
        assert updated_state.log_likelihood is not None
        assert jnp.all(jnp.isfinite(updated_state.state_mean))
        assert jnp.all(jnp.isfinite(updated_state.state_cov))
    
    def test_jacobian_computation(self, extended_filter, test_state):
        """Test automatic Jacobian computation."""
        state = test_state.state_mean
        
        # Test transition Jacobian
        F_jacobian = extended_filter.transition_jacobian_fn(state, 1.0)
        assert F_jacobian.shape == (4, 4)
        assert jnp.all(jnp.isfinite(F_jacobian))
        
        # Test measurement Jacobian
        H_jacobian = extended_filter.measurement_jacobian_fn(state)
        assert H_jacobian.shape == (4, 4)
        assert jnp.all(jnp.isfinite(H_jacobian))


class TestUnscentedKalmanFilter:
    """Test Unscented Kalman Filter."""
    
    def test_ukf_initialization(self, unscented_filter):
        """Test UKF initialization."""
        assert hasattr(unscented_filter, 'alpha')
        assert hasattr(unscented_filter, 'beta')
        assert hasattr(unscented_filter, 'kappa')
        assert hasattr(unscented_filter, 'weight_m')
        assert hasattr(unscented_filter, 'weight_c')
    
    def test_sigma_point_generation(self, unscented_filter, test_state):
        """Test sigma point generation."""
        sigma_points = unscented_filter._generate_sigma_points(
            test_state.state_mean, test_state.state_cov
        )
        
        # Should have 2n+1 sigma points
        assert sigma_points.shape == (9, 4)  # 2*4+1 = 9
        assert jnp.all(jnp.isfinite(sigma_points))
        
        # Central point should be the mean
        assert jnp.allclose(sigma_points[0], test_state.state_mean)
    
    def test_ukf_prediction(self, unscented_filter, test_state):
        """Test UKF prediction."""
        predicted_state = unscented_filter.predict(test_state, dt=1.0)
        
        assert predicted_state.state_mean.shape == (4,)
        assert predicted_state.state_cov.shape == (4, 4)
        assert jnp.all(jnp.isfinite(predicted_state.state_mean))
        assert jnp.all(jnp.isfinite(predicted_state.state_cov))
        assert jnp.all(jnp.linalg.eigvals(predicted_state.state_cov) >= -1e-10)
    
    def test_ukf_update(self, unscented_filter, test_state, test_observation):
        """Test UKF update."""
        predicted_state = unscented_filter.predict(test_state, dt=1.0)
        updated_state = unscented_filter.update(predicted_state, test_observation)
        
        assert updated_state.innovation is not None
        assert updated_state.kalman_gain is not None
        assert updated_state.log_likelihood is not None
        assert jnp.all(jnp.isfinite(updated_state.state_mean))
        assert jnp.all(jnp.isfinite(updated_state.state_cov))
    
    def test_ukf_vs_ekf_comparison(self, basic_config, test_state, test_observation):
        """Compare UKF vs EKF performance on non-linear system."""
        transition_fn = create_player_transition_function()
        measurement_fn = create_player_measurement_function()
        
        ukf = UnscentedKalmanFilter(basic_config, transition_fn, measurement_fn)
        ekf = ExtendedKalmanFilter(basic_config, transition_fn, measurement_fn)
        
        # Run same sequence on both filters
        ukf_state = test_state
        ekf_state = test_state
        
        for i in range(3):
            # Predict
            ukf_pred = ukf.predict(ukf_state, dt=1.0)
            ekf_pred = ekf.predict(ekf_state, dt=1.0)
            
            # Update
            obs = test_observation + 0.05 * np.random.randn(4)
            ukf_state = ukf.update(ukf_pred, obs)
            ekf_state = ekf.update(ekf_pred, obs)
        
        # Both should converge to reasonable values
        assert jnp.all(jnp.isfinite(ukf_state.state_mean))
        assert jnp.all(jnp.isfinite(ekf_state.state_mean))


class TestNumericalStability:
    """Test numerical stability features."""
    
    def test_ill_conditioned_covariance(self, linear_filter, test_state):
        """Test handling of ill-conditioned covariance matrices."""
        # Create ill-conditioned initial covariance
        ill_conditioned_cov = jnp.array([
            [1e6, 1e6-1, 0, 0],
            [1e6-1, 1e6, 0, 0],
            [0, 0, 1e-10, 0],
            [0, 0, 0, 1e-10]
        ])
        
        ill_conditioned_state = FilterState(
            state_mean=test_state.state_mean,
            state_cov=ill_conditioned_cov,
            timestamp=test_state.timestamp,
            gameweek=test_state.gameweek,
            season=test_state.season,
            player_id=test_state.player_id
        )
        
        # Should handle gracefully with regularization
        stabilized_cov = linear_filter._ensure_numerical_stability(ill_conditioned_cov)
        
        # Check that stabilization worked
        condition_number = jnp.linalg.cond(stabilized_cov)
        assert condition_number < 1e12  # Should be better conditioned
        assert jnp.all(jnp.linalg.eigvals(stabilized_cov) >= 0)
    
    def test_negative_eigenvalue_correction(self, linear_filter):
        """Test correction of negative eigenvalues."""
        # Create matrix with small negative eigenvalue
        bad_matrix = jnp.array([
            [1.0, 0.5, 0, 0],
            [0.5, 1.0, 0, 0],
            [0, 0, -1e-12, 0],  # Small negative eigenvalue
            [0, 0, 0, 1.0]
        ])
        
        corrected_matrix = linear_filter._ensure_numerical_stability(bad_matrix)
        
        # Should have only non-negative eigenvalues
        eigenvals = jnp.linalg.eigvals(corrected_matrix)
        assert jnp.all(eigenvals >= -1e-10)
    
    def test_nan_and_inf_detection(self, linear_filter):
        """Test detection and handling of NaN/Inf values."""
        # Create matrix with NaN
        nan_matrix = jnp.array([
            [1.0, jnp.nan, 0, 0],
            [jnp.nan, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0]
        ])
        
        with pytest.raises(NumericalInstabilityError):
            linear_filter._ensure_numerical_stability(nan_matrix)
        
        # Create matrix with Inf
        inf_matrix = jnp.array([
            [jnp.inf, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0]
        ])
        
        with pytest.raises(NumericalInstabilityError):
            linear_filter._ensure_numerical_stability(inf_matrix)


class TestKalmanPlayerModel:
    """Test KalmanPlayerModel integration."""
    
    @pytest.fixture
    def config(self):
        """State space configuration for player model."""
        return StateSpaceConfig(
            state_dim=4,
            obs_dim=4,
            state_names=["skill", "form", "consistency", "momentum"],
            obs_names=["goals", "assists", "minutes", "bonus"]
        )
    
    @pytest.fixture
    def player_model(self, config):
        """Kalman player model for testing."""
        return KalmanPlayerModel(config, filter_type="extended")
    
    def test_model_initialization(self, player_model):
        """Test player model initialization."""
        assert player_model.filter_type == "extended"
        assert player_model.model_mode == "single"
        assert len(player_model.filters) == 1
        assert "main" in player_model.filters
    
    def test_state_initialization(self, player_model):
        """Test player state initialization."""
        player_state = player_model.initialize_state(
            player_id=123,
            gameweek=1,
            season="2023"
        )
        
        assert isinstance(player_state, PlayerState)
        assert player_state.player_id == 123
        assert player_state.gameweek == 1
        assert player_state.season == "2023"
        assert player_state.state_mean.shape == (4,)
        assert player_state.state_cov.shape == (4, 4)
    
    def test_predict_state(self, player_model):
        """Test state prediction."""
        # Initialize state
        current_state = player_model.initialize_state(123, gameweek=1, season="2023")
        
        # Predict forward
        predicted_state = player_model.predict_state(current_state, gameweeks_ahead=3)
        
        assert predicted_state.player_id == 123
        assert predicted_state.gameweek == 4  # 1 + 3
        assert predicted_state.season == "2023"
        assert predicted_state.state_mean.shape == (4,)
        assert predicted_state.state_cov.shape == (4, 4)
    
    def test_update_state(self, player_model):
        """Test state update with observations."""
        # Initialize and predict
        current_state = player_model.initialize_state(123, gameweek=1, season="2023")
        predicted_state = player_model.predict_state(current_state, gameweeks_ahead=1)
        
        # Update with observation
        observation = jnp.array([1.0, 0.5, 90.0, 2.0])
        updated_state = player_model.update_state(predicted_state, observation)
        
        assert updated_state.player_id == 123
        assert updated_state.gameweek == predicted_state.gameweek
        assert updated_state.state_mean.shape == (4,)
        assert updated_state.state_cov.shape == (4, 4)
        
        # Uncertainty should decrease after update
        assert jnp.trace(updated_state.state_cov) <= jnp.trace(predicted_state.state_cov)
    
    def test_observation_model(self, player_model):
        """Test observation model mapping."""
        state = jnp.array([0.8, 0.7, 0.9, 0.1])  # High skill, good form, consistent
        
        observation = player_model.observation_model(state)
        
        assert observation.shape == (4,)
        assert jnp.all(jnp.isfinite(observation))
        assert jnp.all(observation >= 0)  # Performance metrics should be non-negative
    
    def test_missing_observations(self, player_model):
        """Test handling of missing observations."""
        current_state = player_model.initialize_state(123, gameweek=1, season="2023")
        predicted_state = player_model.predict_state(current_state, gameweeks_ahead=1)
        
        # Create observation with NaN values
        observation = jnp.array([1.0, jnp.nan, 90.0, 2.0])
        
        # Should skip update and return predicted state
        result_state = player_model.update_state(predicted_state, observation)
        
        # Should return predicted state unchanged (no update)
        assert jnp.allclose(result_state.state_mean, predicted_state.state_mean)
        assert jnp.allclose(result_state.state_cov, predicted_state.state_cov)
    
    def test_model_fitting(self, player_model):
        """Test model fitting with synthetic data."""
        # Create synthetic training data
        n_players = 5
        n_gameweeks = 10
        n_features = 4
        
        player_ids = list(range(n_players))
        features = np.random.randn(n_players, n_gameweeks, n_features) * 0.5 + 2.0
        targets = np.random.randn(n_players, n_gameweeks) * 0.3 + 5.0
        gameweeks = list(range(1, n_gameweeks + 1))
        
        training_data = {
            "player_ids": player_ids,
            "features": features,
            "targets": targets,
            "gameweeks": gameweeks
        }
        
        # Fit model
        player_model.fit(training_data, season="2023", max_gameweek=n_gameweeks)
        
        assert player_model.is_fitted
        assert len(player_model.player_states) == n_players
        assert player_model.last_season == "2023"
        assert player_model.last_update_gameweek == n_gameweeks
    
    def test_prediction_pipeline(self, player_model):
        """Test end-to-end prediction pipeline."""
        # Fit with synthetic data
        n_players = 3
        n_gameweeks = 5
        
        training_data = {
            "player_ids": list(range(n_players)),
            "features": np.random.randn(n_players, n_gameweeks, 4) * 0.5 + 2.0,
            "targets": np.random.randn(n_players, n_gameweeks) * 0.3 + 5.0,
            "gameweeks": list(range(1, n_gameweeks + 1))
        }
        
        player_model.fit(training_data, season="2023", max_gameweek=n_gameweeks)
        
        # Make predictions
        predictions = player_model.predict(
            player_ids=[0, 1, 2],
            gameweeks_ahead=3
        )
        
        assert "player_ids" in predictions
        assert "predictions" in predictions
        assert "uncertainty" in predictions
        assert "states" in predictions
        assert "metadata" in predictions
        
        assert predictions["predictions"].shape == (3, 3)
        assert predictions["uncertainty"].shape == (3, 3)
        assert len(predictions["states"]) == 3


class TestPositionSpecificModel:
    """Test position-specific Kalman model."""
    
    @pytest.fixture
    def position_model(self):
        """Position-specific Kalman model."""
        config = StateSpaceConfig(state_dim=4, obs_dim=4)
        return PositionSpecificKalmanModel(config=config)
    
    def test_position_model_initialization(self, position_model):
        """Test position-specific model initialization."""
        assert position_model.model_mode == "position_specific"
        assert len(position_model.filters) == 4  # GK, DEF, MID, FWD
        assert "GK" in position_model.filters
        assert "DEF" in position_model.filters
        assert "MID" in position_model.filters
        assert "FWD" in position_model.filters
    
    def test_position_specific_configs(self, position_model):
        """Test position-specific filter configurations."""
        assert len(position_model.position_configs) == 4
        
        # Goalkeeper should have lower noise
        gk_config = position_model.position_configs["GK"]
        mid_config = position_model.position_configs["MID"]
        fwd_config = position_model.position_configs["FWD"]
        
        assert gk_config.process_noise_std < mid_config.process_noise_std
        assert fwd_config.process_noise_std > mid_config.process_noise_std
    
    def test_position_specific_dynamics(self, position_model):
        """Test position-specific state transitions."""
        # Test different positions have different dynamics
        state = jnp.array([0.8, 0.7, 0.9, 0.1])
        
        gk_filter = position_model.filters["GK"]
        fwd_filter = position_model.filters["FWD"]
        
        # Create test states
        test_state = FilterState(
            state_mean=state,
            state_cov=jnp.eye(4) * 0.1,
            timestamp=1.0,
            gameweek=1,
            season="2023",
            player_id=123
        )
        
        # Predict with different position filters
        gk_prediction = gk_filter.predict(test_state, dt=1.0)
        fwd_prediction = fwd_filter.predict(test_state, dt=1.0)
        
        # Should have different evolution patterns
        assert not jnp.allclose(gk_prediction.state_mean, fwd_prediction.state_mean, rtol=1e-2)
    
    def test_position_specific_measurements(self, position_model):
        """Test position-specific measurement functions."""
        state = jnp.array([0.8, 0.7, 0.9, 0.1])
        
        # Get measurement functions for different positions
        gk_fn = position_model._create_position_measurement_function("GK")
        fwd_fn = position_model._create_position_measurement_function("FWD")
        
        gk_obs = gk_fn(state)
        fwd_obs = fwd_fn(state)
        
        # Goalkeepers should have lower goals, forwards higher
        assert gk_obs[0] < fwd_obs[0]  # Goals
        assert gk_obs[2] > fwd_obs[2]  # Minutes (GKs more likely to play full games)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_zero_covariance_handling(self, linear_filter, test_state):
        """Test handling of zero covariance."""
        # Set covariance to near-zero
        test_state.state_cov = jnp.eye(4) * 1e-15
        
        # Should handle gracefully
        predicted_state = linear_filter.predict(test_state, dt=1.0)
        assert jnp.all(jnp.isfinite(predicted_state.state_cov))
        assert jnp.all(jnp.linalg.eigvals(predicted_state.state_cov) >= 0)
    
    def test_large_time_step(self, linear_filter, test_state):
        """Test handling of large time steps."""
        # Very large time step
        predicted_state = linear_filter.predict(test_state, dt=100.0)
        
        assert jnp.all(jnp.isfinite(predicted_state.state_mean))
        assert jnp.all(jnp.isfinite(predicted_state.state_cov))
    
    def test_extreme_observations(self, linear_filter, test_state):
        """Test handling of extreme observations."""
        predicted_state = linear_filter.predict(test_state, dt=1.0)
        
        # Extreme observation values
        extreme_obs = jnp.array([1000.0, -500.0, 1e6, -1e6])
        
        # Should handle without crashing
        updated_state = linear_filter.update(predicted_state, extreme_obs)
        
        assert jnp.all(jnp.isfinite(updated_state.state_mean))
        assert jnp.all(jnp.isfinite(updated_state.state_cov))
    
    def test_model_diagnostics(self, player_model):
        """Test model diagnostics functionality."""
        # Initialize some states
        for i in range(3):
            player_model.initialize_state(i, gameweek=1, season="2023")
        
        diagnostics = player_model.get_model_diagnostics()
        
        assert "model_metrics" in diagnostics
        assert "num_players" in diagnostics
        assert "filter_config" in diagnostics
        assert "filter_diagnostics" in diagnostics
        
        assert diagnostics["num_players"] == 3
        assert diagnostics["filter_config"]["filter_type"] == "extended"


class TestPerformanceAndScaling:
    """Test performance characteristics."""
    
    @pytest.mark.slow
    def test_large_state_dimensions(self):
        """Test with large state dimensions."""
        config = FilterConfig(state_dim=20, obs_dim=15)
        filter = create_simple_linear_filter(config)
        
        state = FilterState(
            state_mean=jnp.ones(20) * 0.5,
            state_cov=jnp.eye(20) * 0.1,
            timestamp=1.0,
            gameweek=1,
            season="2023",
            player_id=123
        )
        
        observation = jnp.ones(15) * 1.0
        
        # Should handle large dimensions
        predicted = filter.predict(state, dt=1.0)
        updated = filter.update(predicted, observation)
        
        assert jnp.all(jnp.isfinite(updated.state_mean))
        assert jnp.all(jnp.isfinite(updated.state_cov))
    
    @pytest.mark.slow
    def test_many_players_scaling(self):
        """Test scaling with many players."""
        config = StateSpaceConfig(state_dim=4, obs_dim=4)
        model = KalmanPlayerModel(config)
        
        n_players = 100
        
        # Initialize many players
        for i in range(n_players):
            model.initialize_state(i, gameweek=1, season="2023")
        
        # Make predictions for all
        predictions = model.predict(
            player_ids=list(range(n_players)),
            gameweeks_ahead=1
        )
        
        assert predictions["predictions"].shape == (n_players, 1)
        assert len(predictions["states"]) == n_players


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])