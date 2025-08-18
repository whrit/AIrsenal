"""
Unit tests for AdaptivePlayerModel implementation.

This test suite validates the state-space modeling capabilities of the AdaptivePlayerModel
base class, including state representation, temporal dynamics, and adaptive learning.

Test coverage includes:
- State initialization and validation
- State-space configuration
- Abstract method enforcement
- Data validation and error handling
- State management and updates
- Integration with BasePlayerModel interface
"""

import numpy as np
import pytest
import jax.numpy as jnp

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
)
from airsenal.framework.player_model import BasePlayerModel


class MockAdaptivePlayerModel(AdaptivePlayerModel):
    """
    Concrete implementation of AdaptivePlayerModel for testing.
    
    This mock class implements all abstract methods with simple logic
    to enable testing of the base class functionality.
    """
    
    def initialize_state(
        self, player_id, initial_data=None, gameweek=1, season="2023", **kwargs
    ):
        """Initialize state with simple default values."""
        state_mean = np.array([5.0, 0.0, 1.0])  # [skill, form, consistency]
        state_cov = np.eye(self.config.state_dim) * self.config.initial_state_std**2
        
        return PlayerState(
            player_id=player_id,
            state_mean=state_mean,
            state_cov=state_cov,
            gameweek=gameweek,
            season=season,
        )
    
    def get_probs(self):
        """Implement BasePlayerModel abstract method."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        player_ids = list(self.player_states.keys())
        n_players = len(player_ids)
        
        return {
            "player_id": np.array(player_ids),
            "prob_score": np.random.uniform(0.1, 0.3, n_players),
            "prob_assist": np.random.uniform(0.05, 0.2, n_players),
            "prob_neither": np.random.uniform(0.5, 0.8, n_players),
        }
    
    def get_probs_for_player(self, player_id):
        """Implement BasePlayerModel abstract method."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
            
        if player_id not in self.player_states:
            raise RuntimeError(f"Unknown player_id {player_id}")
            
        # Simple mock probabilities based on state
        state = self.player_states[player_id]
        skill = state.state_mean[0] if len(state.state_mean) > 0 else 5.0
        
        # Scale probabilities based on skill level
        prob_score = min(0.3, max(0.05, skill / 20.0))
        prob_assist = min(0.2, max(0.02, skill / 30.0))
        prob_neither = 1.0 - prob_score - prob_assist
        
        return np.array([prob_score, prob_assist, prob_neither])
    
    def predict_state(self, current_state, gameweeks_ahead=1, **kwargs):
        """Simple state prediction with random walk."""
        # Simple random walk model
        predicted_mean = current_state.state_mean.copy()
        
        # Add process noise
        process_noise_var = (
            self.config.process_noise_std**2 * gameweeks_ahead
        )
        predicted_cov = (
            current_state.state_cov + np.eye(self.config.state_dim) * process_noise_var
        )
        
        return PlayerState(
            player_id=current_state.player_id,
            state_mean=predicted_mean,
            state_cov=predicted_cov,
            gameweek=current_state.gameweek + gameweeks_ahead,
            season=current_state.season,
        )
    
    def update_state(self, predicted_state, observation, observation_noise=None, **kwargs):
        """Simple Kalman filter update."""
        if observation_noise is None:
            observation_noise = np.eye(self.config.obs_dim) * self.config.measurement_noise_std**2
        
        # Simple observation matrix (identity mapping to first obs_dim states)
        H = np.zeros((self.config.obs_dim, self.config.state_dim))
        H[:min(self.config.obs_dim, self.config.state_dim), :min(self.config.obs_dim, self.config.state_dim)] = np.eye(
            min(self.config.obs_dim, self.config.state_dim)
        )
        
        # Kalman filter update equations
        S = H @ predicted_state.state_cov @ H.T + observation_noise
        K = predicted_state.state_cov @ H.T @ np.linalg.inv(S)
        
        expected_obs = H @ predicted_state.state_mean
        innovation = observation - expected_obs
        
        updated_mean = predicted_state.state_mean + K @ innovation
        updated_cov = predicted_state.state_cov - K @ H @ predicted_state.state_cov
        
        return PlayerState(
            player_id=predicted_state.player_id,
            state_mean=updated_mean,
            state_cov=updated_cov,
            gameweek=predicted_state.gameweek,
            season=predicted_state.season,
        )
    
    def observation_model(self, state, **kwargs):
        """Map state to observations."""
        # Simple linear mapping
        H = np.zeros((self.config.obs_dim, self.config.state_dim))
        H[:min(self.config.obs_dim, self.config.state_dim), :min(self.config.obs_dim, self.config.state_dim)] = np.eye(
            min(self.config.obs_dim, self.config.state_dim)
        )
        return H @ state
    
    def _fit_state_space_model(self, data, season, max_gameweek, **kwargs):
        """Mock fitting method."""
        # Simple parameter learning
        self.transition_matrix = np.eye(self.config.state_dim)
        self.observation_matrix = np.zeros((self.config.obs_dim, self.config.state_dim))
        self.observation_matrix[:min(self.config.obs_dim, self.config.state_dim), :min(self.config.obs_dim, self.config.state_dim)] = np.eye(
            min(self.config.obs_dim, self.config.state_dim)
        )
        self.process_noise_cov = np.eye(self.config.state_dim) * self.config.process_noise_std**2
        self.measurement_noise_cov = np.eye(self.config.obs_dim) * self.config.measurement_noise_std**2


class TestStateSpaceConfig:
    """Test StateSpaceConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StateSpaceConfig()
        
        assert config.state_dim == 3
        assert config.obs_dim == 4
        assert config.process_noise_std == 0.1
        assert config.measurement_noise_std == 0.2
        assert config.initial_state_std == 1.0
        assert config.use_jax is True
        assert config.state_names == ["skill", "form", "consistency"]
        assert config.obs_names == ["goals", "assists", "minutes", "bonus"]
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StateSpaceConfig(
            state_dim=2,
            obs_dim=3,
            process_noise_std=0.05,
            measurement_noise_std=0.1,
            state_names=["ability", "form"],
            obs_names=["goals", "assists", "minutes"],
        )
        
        assert config.state_dim == 2
        assert config.obs_dim == 3
        assert config.process_noise_std == 0.05
        assert config.measurement_noise_std == 0.1
        assert config.state_names == ["ability", "form"]
        assert config.obs_names == ["goals", "assists", "minutes"]
    
    def test_invalid_config(self):
        """Test invalid configuration raises errors."""
        with pytest.raises(ValueError, match="state_names length"):
            StateSpaceConfig(state_dim=3, state_names=["skill", "form"])
        
        with pytest.raises(ValueError, match="obs_names length"):
            StateSpaceConfig(obs_dim=2, obs_names=["goals", "assists", "minutes"])


class TestPlayerState:
    """Test PlayerState class."""
    
    def test_player_state_creation(self):
        """Test PlayerState creation and basic properties."""
        state_mean = np.array([5.0, 0.0, 1.0])
        state_cov = np.eye(3) * 0.5
        
        state = PlayerState(
            player_id=123,
            state_mean=state_mean,
            state_cov=state_cov,
            gameweek=5,
            season="2023",
        )
        
        assert state.player_id == 123
        assert np.array_equal(state.state_mean, state_mean)
        assert np.array_equal(state.state_cov, state_cov)
        assert state.gameweek == 5
        assert state.season == "2023"
        assert state.last_updated is None
    
    def test_state_serialization(self):
        """Test state to_dict and from_dict methods."""
        state_mean = np.array([5.0, 0.0, 1.0])
        state_cov = np.eye(3) * 0.5
        
        original_state = PlayerState(
            player_id=123,
            state_mean=state_mean,
            state_cov=state_cov,
            gameweek=5,
            season="2023",
            last_updated="2023-10-15",
        )
        
        # Test serialization
        state_dict = original_state.to_dict()
        
        assert state_dict["player_id"] == 123
        assert state_dict["gameweek"] == 5
        assert state_dict["season"] == "2023"
        assert state_dict["last_updated"] == "2023-10-15"
        assert isinstance(state_dict["state_mean"], list)
        assert isinstance(state_dict["state_cov"], list)
        
        # Test deserialization
        restored_state = PlayerState.from_dict(state_dict)
        
        assert restored_state.player_id == original_state.player_id
        assert np.array_equal(restored_state.state_mean, original_state.state_mean)
        assert np.array_equal(restored_state.state_cov, original_state.state_cov)
        assert restored_state.gameweek == original_state.gameweek
        assert restored_state.season == original_state.season
        assert restored_state.last_updated == original_state.last_updated
    
    def test_state_copy(self):
        """Test state copy method."""
        state_mean = np.array([5.0, 0.0, 1.0])
        state_cov = np.eye(3) * 0.5
        
        original_state = PlayerState(
            player_id=123,
            state_mean=state_mean,
            state_cov=state_cov,
            gameweek=5,
            season="2023",
        )
        
        copied_state = original_state.copy()
        
        # Should be equal but not the same object
        assert copied_state.player_id == original_state.player_id
        assert np.array_equal(copied_state.state_mean, original_state.state_mean)
        assert np.array_equal(copied_state.state_cov, original_state.state_cov)
        assert copied_state is not original_state
        assert copied_state.state_mean is not original_state.state_mean
        assert copied_state.state_cov is not original_state.state_cov


class TestAdaptivePlayerModel:
    """Test AdaptivePlayerModel base class."""
    
    def test_inheritance(self):
        """Test that AdaptivePlayerModel inherits from BasePlayerModel."""
        config = StateSpaceConfig()
        model = MockAdaptivePlayerModel(config)
        
        assert isinstance(model, BasePlayerModel)
        assert isinstance(model, AdaptivePlayerModel)
    
    def test_initialization(self):
        """Test model initialization."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config, learning_rate=0.05, decay_factor=0.9)
        
        assert model.config == config
        assert model.learning_rate == 0.05
        assert model.decay_factor == 0.9
        assert model.is_fitted is False
        assert model.player_states == {}
        assert model.feature_importance_ is None
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        config = StateSpaceConfig()
        
        with pytest.raises(ValueError, match="learning_rate must be in"):
            MockAdaptivePlayerModel(config, learning_rate=0.0)
        
        with pytest.raises(ValueError, match="learning_rate must be in"):
            MockAdaptivePlayerModel(config, learning_rate=1.5)
        
        with pytest.raises(ValueError, match="decay_factor must be in"):
            MockAdaptivePlayerModel(config, decay_factor=0.0)
        
        with pytest.raises(ValueError, match="decay_factor must be in"):
            MockAdaptivePlayerModel(config, decay_factor=1.0)
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods are enforced."""
        # Cannot instantiate abstract base class directly
        config = StateSpaceConfig()
        
        with pytest.raises(TypeError):
            AdaptivePlayerModel(config)
    
    def test_state_initialization(self):
        """Test state initialization method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        state = model.initialize_state(player_id=123, gameweek=1, season="2023")
        
        assert isinstance(state, PlayerState)
        assert state.player_id == 123
        assert state.gameweek == 1
        assert state.season == "2023"
        assert state.state_mean.shape == (3,)
        assert state.state_cov.shape == (3, 3)
    
    def test_state_prediction(self):
        """Test state prediction method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        initial_state = model.initialize_state(player_id=123, gameweek=1, season="2023")
        predicted_state = model.predict_state(initial_state, gameweeks_ahead=2)
        
        assert isinstance(predicted_state, PlayerState)
        assert predicted_state.player_id == initial_state.player_id
        assert predicted_state.gameweek == initial_state.gameweek + 2
        assert predicted_state.season == initial_state.season
        assert predicted_state.state_mean.shape == initial_state.state_mean.shape
        assert predicted_state.state_cov.shape == initial_state.state_cov.shape
    
    def test_state_update(self):
        """Test state update method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        predicted_state = model.initialize_state(player_id=123, gameweek=1, season="2023")
        observation = np.array([1.0, 0.5, 90.0, 2.0])  # goals, assists, minutes, bonus
        
        updated_state = model.update_state(predicted_state, observation)
        
        assert isinstance(updated_state, PlayerState)
        assert updated_state.player_id == predicted_state.player_id
        assert updated_state.gameweek == predicted_state.gameweek
        assert updated_state.season == predicted_state.season
        assert updated_state.state_mean.shape == predicted_state.state_mean.shape
        assert updated_state.state_cov.shape == predicted_state.state_cov.shape
    
    def test_observation_model(self):
        """Test observation model method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        state = np.array([5.0, 0.0, 1.0])
        observation = model.observation_model(state)
        
        assert observation.shape == (config.obs_dim,)
        assert isinstance(observation, np.ndarray)
    
    def test_fit_method(self):
        """Test model fitting."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Create mock training data
        n_players, n_gameweeks, n_features = 5, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3, 4, 5]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        
        # Fit model
        fitted_model = model.fit(data, season="2023", max_gameweek=10)
        
        assert fitted_model is model  # Should return self
        assert model.is_fitted is True
        assert len(model.player_states) == n_players
        assert model.last_season == "2023"
        assert model.last_update_gameweek == 10
        assert model.feature_importance_ is not None
    
    def test_fit_validation(self):
        """Test data validation in fit method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Test missing required keys
        with pytest.raises(RuntimeError, match="Failed to fit"):
            model.fit({}, season="2023", max_gameweek=10)
        
        # Test empty player list
        with pytest.raises(RuntimeError, match="Failed to fit"):
            data = {
                "player_ids": np.array([]),
                "features": np.array([]),
                "targets": np.array([]),
                "gameweeks": np.array([]),
            }
            model.fit(data, season="2023", max_gameweek=10)
    
    def test_predict_method(self):
        """Test prediction method."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Fit model first
        n_players, n_gameweeks, n_features = 3, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        model.fit(data, season="2023", max_gameweek=10)
        
        # Test prediction
        player_ids = [1, 2]
        gameweeks_ahead = 3
        predictions = model.predict(player_ids, gameweeks_ahead=gameweeks_ahead)
        
        assert "player_ids" in predictions
        assert "predictions" in predictions
        assert "uncertainty" in predictions
        assert "states" in predictions
        assert "metadata" in predictions
        
        assert np.array_equal(predictions["player_ids"], player_ids)
        assert predictions["predictions"].shape == (len(player_ids), gameweeks_ahead)
        assert predictions["uncertainty"].shape == (len(player_ids), gameweeks_ahead)
        assert len(predictions["states"]) == len(player_ids)
    
    def test_predict_unfitted(self):
        """Test prediction fails when model not fitted."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict([1, 2, 3])
    
    def test_update_with_recent_data(self):
        """Test updating with recent data."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Fit model first
        n_players, n_gameweeks, n_features = 3, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        model.fit(data, season="2023", max_gameweek=10)
        
        # Update with recent data
        recent_data = {
            "player_ids": [1, 2],
            "observations": {
                1: [2.0, 1.0, 90.0, 3.0],  # goals, assists, minutes, bonus
                2: [0.0, 2.0, 75.0, 1.0],
            }
        }
        
        updated_model = model.update_with_recent_data(
            recent_data, gameweek=11, season="2023"
        )
        
        assert updated_model is model  # Should return self
        assert model.last_update_gameweek == 11
        assert model.last_season == "2023"
    
    def test_update_unfitted(self):
        """Test update fails when model not fitted."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.update_with_recent_data({}, gameweek=1, season="2023")
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Fit model first
        n_players, n_gameweeks, n_features = 3, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        model.fit(data, season="2023", max_gameweek=10)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == config.state_dim
        assert all(name in importance for name in config.state_names)
        assert all(0 <= score <= 1 for score in importance.values())
    
    def test_feature_importance_unfitted(self):
        """Test feature importance fails when model not fitted."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_feature_importance()
    
    def test_player_state_management(self):
        """Test player state management methods."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Fit model first
        n_players, n_gameweeks, n_features = 3, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        model.fit(data, season="2023", max_gameweek=10)
        
        # Test get_player_state
        state = model.get_player_state(1)
        assert isinstance(state, PlayerState)
        assert state.player_id == 1
        
        # Test get_player_state for non-existent player
        assert model.get_player_state(999) is None
        
        # Test get_all_player_states
        all_states = model.get_all_player_states()
        assert isinstance(all_states, dict)
        assert len(all_states) == n_players
        assert all(isinstance(state, PlayerState) for state in all_states.values())
        
        # Test get_state_summary
        summary = model.get_state_summary()
        assert isinstance(summary, dict)
        assert summary["n_players"] == n_players
        assert summary["state_names"] == config.state_names
        assert "mean_states" in summary
        assert "std_states" in summary
        assert "mean_uncertainties" in summary
    
    def test_jax_compatibility(self):
        """Test JAX array compatibility."""
        config = StateSpaceConfig(use_jax=True)
        model = MockAdaptivePlayerModel(config)
        
        # Test with JAX arrays
        state = jnp.array([5.0, 0.0, 1.0])
        observation = model.observation_model(state)
        
        # Should work with both numpy and JAX arrays
        assert isinstance(observation, (np.ndarray, jnp.ndarray))
        assert observation.shape == (config.obs_dim,)


class TestIntegration:
    """Integration tests for AdaptivePlayerModel."""
    
    def test_full_workflow(self):
        """Test complete workflow from fitting to prediction and update."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config, learning_rate=0.1, decay_factor=0.9)
        
        # Step 1: Fit model
        n_players, n_gameweeks, n_features = 5, 15, 8
        training_data = {
            "player_ids": np.array([10, 20, 30, 40, 50]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 15, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        
        model.fit(training_data, season="2023", max_gameweek=15)
        assert model.is_fitted
        
        # Step 2: Make predictions
        player_ids = [10, 20, 30]
        predictions = model.predict(player_ids, gameweeks_ahead=5)
        
        assert predictions["predictions"].shape == (3, 5)
        assert predictions["uncertainty"].shape == (3, 5)
        
        # Step 3: Update with new data
        recent_data = {
            "player_ids": [10, 20],
            "observations": {
                10: [1.0, 0.0, 90.0, 2.0],
                20: [0.0, 1.0, 85.0, 1.0],
            }
        }
        
        model.update_with_recent_data(recent_data, gameweek=16, season="2023")
        assert model.last_update_gameweek == 16
        
        # Step 4: Make new predictions after update
        new_predictions = model.predict(player_ids, gameweeks_ahead=3)
        assert new_predictions["predictions"].shape == (3, 3)
        
        # Step 5: Check state summary
        summary = model.get_state_summary()
        assert summary["n_players"] == n_players
        assert summary["last_update_gameweek"] == 16
    
    def test_model_state_persistence(self):
        """Test model state can be saved and restored."""
        config = StateSpaceConfig(state_dim=3, obs_dim=4)
        model = MockAdaptivePlayerModel(config)
        
        # Fit model
        n_players, n_gameweeks, n_features = 3, 10, 6
        data = {
            "player_ids": np.array([1, 2, 3]),
            "features": np.random.randn(n_players, n_gameweeks, n_features),
            "targets": np.random.randint(0, 10, (n_players, n_gameweeks)),
            "gameweeks": np.arange(1, n_gameweeks + 1),
        }
        model.fit(data, season="2023", max_gameweek=10)
        
        # Get states
        original_states = model.get_all_player_states()
        
        # Convert states to dict and back
        serialized_states = {}
        for player_id, state in original_states.items():
            serialized_states[player_id] = state.to_dict()
        
        restored_states = {}
        for player_id, state_dict in serialized_states.items():
            restored_states[player_id] = PlayerState.from_dict(state_dict)
        
        # Verify states are equivalent
        for player_id in original_states:
            orig = original_states[player_id]
            rest = restored_states[player_id]
            
            assert orig.player_id == rest.player_id
            assert np.allclose(orig.state_mean, rest.state_mean)
            assert np.allclose(orig.state_cov, rest.state_cov)
            assert orig.gameweek == rest.gameweek
            assert orig.season == rest.season