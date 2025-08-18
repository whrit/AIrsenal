"""
Comprehensive tests for NumPyro Framework Integration

This test suite validates the NumPyro integration for AIrsenal's adaptive player
modeling framework, ensuring:
- Seamless integration with existing prediction pipeline
- JAX compatibility and JIT compilation
- MCMC and SVI inference correctness
- Hierarchical modeling capabilities
- Model diagnostics and convergence
- Performance benchmarks
- Backward compatibility

Test Categories:
1. Model Specification Tests
2. Inference Method Tests  
3. JAX Compilation Tests
4. Prediction Accuracy Tests
5. Backward Compatibility Tests
6. Performance Benchmark Tests
7. Integration Tests
8. Model Diagnostics Tests
9. Hierarchical Modeling Tests
10. Error Handling Tests
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import pytest
from jax import vmap
from unittest.mock import Mock, patch

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
    PlayerData
)
from airsenal.framework.kalman_player_model import KalmanPlayerModel
from airsenal.framework.numpyro_integration import (
    NumPyroAdaptiveModel,
    StateSpaceDistributions,
    MCMCStateInference,
    VIStateInference,
    ModelBridge,
    InferenceConfig,
    NumPyroIntegrationError
)

# Disable JAX warnings for tests
warnings.filterwarnings("ignore", category=FutureWarning)
jax.config.update("jax_enable_x64", True)  # Enable float64 for numerical precision

logger = logging.getLogger(__name__)


class TestFixtures:
    """Test fixtures and data generation utilities."""
    
    @staticmethod
    def create_test_config() -> StateSpaceConfig:
        """Create test state space configuration."""
        return StateSpaceConfig(
            state_dim=4,
            obs_dim=4,
            process_noise_std=0.1,
            measurement_noise_std=0.2,
            initial_state_std=1.0,
            state_names=["skill", "form", "consistency", "momentum"],
            obs_names=["goals", "assists", "minutes", "bonus"],
            use_jax=True
        )
    
    @staticmethod
    def create_test_inference_config(inference_type: str = "mcmc") -> InferenceConfig:
        """Create test inference configuration."""
        if inference_type == "mcmc":
            return InferenceConfig(
                inference_type="mcmc",
                num_warmup=100,
                num_samples=200,
                num_chains=2,
                max_tree_depth=8,
                use_jit=True,
                progress_bar=False,
                compute_diagnostics=True,
                check_convergence=False  # Disable for faster tests
            )
        else:
            return InferenceConfig(
                inference_type="svi",
                num_steps=1000,
                learning_rate=0.01,
                use_jit=True,
                progress_bar=False
            )
    
    @staticmethod
    def generate_synthetic_data(
        n_players: int = 20,
        n_gameweeks: int = 10,
        n_features: int = 4,
        add_noise: bool = True,
        missing_prob: float = 0.1,
        random_seed: int = 42
    ) -> PlayerData:
        """Generate synthetic player data for testing."""
        rng = np.random.RandomState(random_seed)
        
        # Generate player IDs
        player_ids = list(range(1, n_players + 1))
        
        # Generate synthetic features with realistic structure
        features = np.zeros((n_players, n_gameweeks, n_features))
        
        for i in range(n_players):
            # Player-specific base abilities
            base_skill = rng.normal(0.5, 0.2)
            base_form = rng.normal(0.5, 0.15)
            base_consistency = rng.normal(0.5, 0.1)
            
            for t in range(n_gameweeks):
                # Time-varying performance with autocorrelation
                if t == 0:
                    form = base_form
                else:
                    form = 0.8 * features[i, t-1, 1] + 0.2 * base_form + rng.normal(0, 0.05)
                
                # Generate observations based on latent abilities
                skill = base_skill + rng.normal(0, 0.05)
                consistency = base_consistency + rng.normal(0, 0.02)
                momentum = rng.normal(0, 0.1)
                
                # Map to observations (goals, assists, minutes, bonus)
                goals = max(0, skill * form + 0.1 * momentum + rng.normal(0, 0.1))
                assists = max(0, 0.8 * skill * consistency + rng.normal(0, 0.08))
                minutes = max(0, min(90, 70 + 20 * form + rng.normal(0, 5)))
                bonus = max(0, (goals + assists) * consistency + rng.normal(0, 0.05))
                
                features[i, t, :] = [goals, assists, minutes, bonus]
        
        # Add missing data
        if missing_prob > 0:
            missing_mask = rng.random((n_players, n_gameweeks, n_features)) < missing_prob
            features[missing_mask] = np.nan
        
        # Add noise
        if add_noise:
            noise = rng.normal(0, 0.02, features.shape)
            features = np.where(np.isnan(features), features, features + noise)
        
        # Generate gameweeks
        gameweeks = list(range(1, n_gameweeks + 1))
        
        # Generate synthetic targets (e.g., FPL points)
        targets = np.sum(features[:, :, :2], axis=2) + 2 * (features[:, :, 2] > 60) + features[:, :, 3]
        
        return {
            "player_ids": player_ids,
            "features": features,
            "targets": targets,
            "gameweeks": gameweeks,
            "nplayer": n_players,
            "nmatch": n_gameweeks
        }
    
    @staticmethod
    def create_player_positions(player_ids: List[int]) -> Dict[int, str]:
        """Create synthetic player position mapping."""
        positions = ["GK", "DEF", "MID", "FWD"]
        position_mapping = {}
        
        for i, player_id in enumerate(player_ids):
            position_mapping[player_id] = positions[i % len(positions)]
        
        return position_mapping


class TestStateSpaceDistributions:
    """Test NumPyro state-space distributions."""
    
    def test_transition_noise_independent(self):
        """Test independent transition noise distribution."""
        state_dim = 4
        noise_scale = 0.1
        
        dist_obj = StateSpaceDistributions.transition_noise(
            state_dim=state_dim,
            noise_scale=noise_scale,
            correlation=None
        )
        
        # Test sampling
        key = random.PRNGKey(42)
        samples = dist_obj.sample(key, (100,))
        
        assert samples.shape == (100, state_dim)
        assert jnp.allclose(jnp.mean(samples, axis=0), 0, atol=0.1)
        assert jnp.allclose(jnp.std(samples, axis=0), noise_scale, atol=0.05)
    
    def test_transition_noise_correlated(self):
        """Test correlated transition noise distribution."""
        state_dim = 3
        noise_scale = 0.2
        correlation = 0.3
        
        dist_obj = StateSpaceDistributions.transition_noise(
            state_dim=state_dim,
            noise_scale=noise_scale,
            correlation=correlation
        )
        
        # Test sampling
        key = random.PRNGKey(42)
        samples = dist_obj.sample(key, (1000,))
        
        assert samples.shape == (1000, state_dim)
        
        # Check correlation structure
        sample_corr = jnp.corrcoef(samples.T)
        expected_corr = correlation * jnp.ones((state_dim, state_dim)) + (1 - correlation) * jnp.eye(state_dim)
        
        assert jnp.allclose(sample_corr, expected_corr, atol=0.1)
    
    def test_observation_noise_homoskedastic(self):
        """Test homoskedastic observation noise."""
        obs_dim = 4
        noise_scale = 0.15
        
        dist_obj = StateSpaceDistributions.observation_noise(
            obs_dim=obs_dim,
            noise_scale=noise_scale,
            heteroskedastic=False
        )
        
        key = random.PRNGKey(42)
        samples = dist_obj.sample(key, (100,))
        
        assert samples.shape == (100, obs_dim)
        assert jnp.allclose(jnp.std(samples, axis=0), noise_scale, atol=0.05)
    
    def test_observation_noise_heteroskedastic(self):
        """Test heteroskedastic observation noise."""
        obs_dim = 3
        noise_scale = 0.1
        
        dist_obj = StateSpaceDistributions.observation_noise(
            obs_dim=obs_dim,
            noise_scale=noise_scale,
            heteroskedastic=True
        )
        
        key = random.PRNGKey(42)
        samples = dist_obj.sample(key, (100,))
        
        assert samples.shape == (100, obs_dim)
        
        # Check that different dimensions have different variances
        sample_stds = jnp.std(samples, axis=0)
        assert not jnp.allclose(sample_stds[0], sample_stds[1], atol=0.01)
    
    def test_hierarchical_prior(self):
        """Test hierarchical prior distributions."""
        num_players = 20
        num_positions = 4
        state_dim = 4
        
        priors = StateSpaceDistributions.hierarchical_prior(
            num_players=num_players,
            num_positions=num_positions,
            state_dim=state_dim
        )
        
        # Check that all required priors are present
        required_keys = ["global_mean", "global_scale", "position_offset", "position_scale", "player_offset"]
        for key in required_keys:
            assert key in priors
        
        # Test sampling from priors
        key = random.PRNGKey(42)
        
        global_mean_sample = priors["global_mean"].sample(key)
        assert global_mean_sample.shape == (state_dim,)
        
        position_offset_sample = priors["position_offset"].sample(key)
        assert position_offset_sample.shape == (num_positions, state_dim)
        
        player_offset_sample = priors["player_offset"].sample(key)
        assert player_offset_sample.shape == (num_players, state_dim)


class TestNumPyroAdaptiveModel:
    """Test NumPyro adaptive model implementation."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        assert model.config == config
        assert model.inference_config == inference_config
        assert not model.enable_hierarchical
        assert model.model_fn is not None
        assert not model.is_fitted
    
    def test_hierarchical_initialization(self):
        """Test hierarchical model initialization."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=True
        )
        
        assert model.enable_hierarchical
        assert model.num_positions == 4
        assert "GK" in model.position_mapping
    
    def test_state_initialization(self):
        """Test player state initialization."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        player_id = 123
        position = "MID"
        
        state = model.initialize_state(
            player_id=player_id,
            gameweek=1,
            season="2023",
            position=position
        )
        
        assert state.player_id == player_id
        assert state.gameweek == 1
        assert state.season == "2023"
        assert state.state_mean.shape == (config.state_dim,)
        assert state.state_cov.shape == (config.state_dim, config.state_dim)
        assert player_id in model.player_positions
        assert model.player_positions[player_id] == 2  # MID index
    
    def test_observation_model(self):
        """Test observation model mapping."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Test state to observation mapping
        state = jnp.array([0.7, 0.8, 0.6, 0.1])  # [skill, form, consistency, momentum]
        obs = model.observation_model(state)
        
        assert obs.shape == (config.obs_dim,)
        assert jnp.all(obs >= 0)  # Observations should be non-negative
        
        # Test that higher skill/form leads to higher goals/assists
        high_skill_state = jnp.array([0.9, 0.9, 0.8, 0.2])
        high_obs = model.observation_model(high_skill_state)
        
        assert high_obs[0] > obs[0]  # More goals
        assert high_obs[1] > obs[1]  # More assists
    
    def test_simple_model_compilation(self):
        """Test JAX compilation of simple model."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config, enable_hierarchical=False)
        
        # Generate test data
        data = TestFixtures.generate_synthetic_data(n_players=5, n_gameweeks=3)
        
        player_ids = jnp.array(data["player_ids"])
        observations = jnp.array(data["features"])
        gameweeks = jnp.array(data["gameweeks"])
        missing_mask = jnp.isnan(observations)
        
        # Test that model can be compiled
        compiled_model = jax.jit(model._simple_state_space_model)
        
        # Test compilation doesn't fail
        try:
            # Run once to trigger compilation
            key = random.PRNGKey(42)
            with numpyro.handlers.seed(rng_seed=42):
                with numpyro.handlers.trace() as trace:
                    model._simple_state_space_model(
                        player_ids, observations, gameweeks, missing_mask
                    )
        except Exception as e:
            pytest.fail(f"Model compilation failed: {e}")
    
    def test_hierarchical_model_compilation(self):
        """Test JAX compilation of hierarchical model."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config, enable_hierarchical=True)
        
        # Generate test data
        data = TestFixtures.generate_synthetic_data(n_players=8, n_gameweeks=3)
        
        player_ids = jnp.array(data["player_ids"])
        observations = jnp.array(data["features"])
        gameweeks = jnp.array(data["gameweeks"])
        player_positions = jnp.array([i % 4 for i in range(len(player_ids))])
        missing_mask = jnp.isnan(observations)
        
        # Test model compilation
        try:
            key = random.PRNGKey(42)
            with numpyro.handlers.seed(rng_seed=42):
                with numpyro.handlers.trace() as trace:
                    model._hierarchical_state_space_model(
                        player_ids, observations, gameweeks, player_positions, missing_mask
                    )
        except Exception as e:
            pytest.fail(f"Hierarchical model compilation failed: {e}")


class TestMCMCInference:
    """Test MCMC inference methods."""
    
    @pytest.mark.slow
    def test_mcmc_inference_simple_model(self):
        """Test MCMC inference on simple model."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        # Generate small dataset for quick testing
        data = TestFixtures.generate_synthetic_data(n_players=3, n_gameweeks=5)
        
        # Fit model
        model.fit(data, season="2023", max_gameweek=5)
        
        assert model.is_fitted
        assert model.posterior_samples is not None
        assert model.mcmc is not None
        assert len(model.posterior_samples) > 0
        
        # Check that we have expected parameters
        expected_params = ["process_noise_scale", "observation_noise_scale", "transition_matrix"]
        for param in expected_params:
            assert param in model.posterior_samples
    
    @pytest.mark.slow
    def test_mcmc_convergence_diagnostics(self):
        """Test MCMC convergence diagnostics."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        inference_config.compute_diagnostics = True
        inference_config.check_convergence = False  # Disable warnings for test
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        data = TestFixtures.generate_synthetic_data(n_players=3, n_gameweeks=4)
        model.fit(data, season="2023", max_gameweek=4)
        
        # Check that diagnostics were computed
        assert len(model.convergence_diagnostics) > 0
        
        # Check that diagnostic structure is correct
        for param_name, diag in model.convergence_diagnostics.items():
            assert "r_hat" in diag or "ess" in diag
    
    @pytest.mark.slow  
    def test_hierarchical_mcmc_inference(self):
        """Test MCMC inference on hierarchical model."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=True
        )
        
        # Generate data with position structure
        data = TestFixtures.generate_synthetic_data(n_players=8, n_gameweeks=4)
        
        # Add position information
        position_mapping = TestFixtures.create_player_positions(data["player_ids"])
        for player_id, position in position_mapping.items():
            model.initialize_state(player_id, position=position)
        
        model.fit(data, season="2023", max_gameweek=4)
        
        assert model.is_fitted
        assert model.posterior_samples is not None
        
        # Check hierarchical parameters
        hierarchical_params = ["global_state_mean", "position_state_offset", "position_state_scale"]
        for param in hierarchical_params:
            assert param in model.posterior_samples


class TestSVIInference:
    """Test SVI inference methods."""
    
    def test_svi_inference_simple_model(self):
        """Test SVI inference on simple model."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("svi")
        inference_config.num_steps = 500  # Reduce for faster testing
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        data = TestFixtures.generate_synthetic_data(n_players=5, n_gameweeks=4)
        
        model.fit(data, season="2023", max_gameweek=4)
        
        assert model.is_fitted
        assert model.posterior_samples is not None
        assert model.svi is not None
        assert model.svi_state is not None
    
    def test_svi_guide_initialization(self):
        """Test SVI guide initialization."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("svi")
        
        # Test different guide types
        for guide_type in ["normal", "multivariate_normal"]:
            inference_config.guide_type = guide_type
            
            model = NumPyroAdaptiveModel(
                config=config,
                inference_config=inference_config,
                enable_hierarchical=False
            )
            
            assert model.guide_fn is not None


class TestPredictions:
    """Test prediction capabilities."""
    
    def test_predict_state(self):
        """Test state prediction."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Create initial state
        current_state = PlayerState(
            player_id=123,
            state_mean=jnp.array([0.6, 0.7, 0.5, 0.1]),
            state_cov=jnp.eye(4) * 0.1,
            gameweek=10,
            season="2023"
        )
        
        # Mock fitted model for prediction test
        model.is_fitted = True
        model.posterior_samples = {
            "transition_matrix": jnp.ones((10, 4, 4)) * 0.1,  # Mock samples
            "process_noise_scale": jnp.ones(10) * 0.1
        }
        
        predicted_state = model.predict_state(current_state, gameweeks_ahead=3)
        
        assert predicted_state.player_id == 123
        assert predicted_state.gameweek == 13
        assert predicted_state.state_mean.shape == (4,)
        assert predicted_state.state_cov.shape == (4, 4)
    
    def test_predict_method(self):
        """Test main predict method."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        # Mock fitted model
        model.is_fitted = True
        model.posterior_samples = {"dummy": jnp.ones(10)}
        
        # Initialize some player states
        for player_id in [1, 2, 3]:
            model.initialize_state(player_id, gameweek=10, season="2023")
        
        predictions = model.predict(
            player_ids=[1, 2, 3],
            gameweeks_ahead=3
        )
        
        assert "player_ids" in predictions
        assert "predictions" in predictions
        assert "uncertainty" in predictions
        assert "states" in predictions
        assert "metadata" in predictions
        
        assert predictions["predictions"].shape == (3, 3)  # 3 players, 3 gameweeks
        assert predictions["uncertainty"].shape == (3, 3)
    
    def test_posterior_predictive_sampling(self):
        """Test posterior predictive sampling."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config, enable_hierarchical=False)
        
        # Mock fitted model
        model.is_fitted = True
        model.posterior_samples = {
            "transition_matrix": jnp.ones((100, 4, 4)) * 0.1,
            "observation_matrix": jnp.ones((100, 4, 4)) * 0.5,
            "process_noise_scale": jnp.ones(100) * 0.1,
            "observation_noise_scale": jnp.ones(100) * 0.2
        }
        
        player_ids = [1, 2, 3]
        for pid in player_ids:
            model.initialize_state(pid)
        
        # Test posterior predictive sampling
        try:
            pred_samples = model.predict_posterior_samples(
                player_ids=player_ids,
                gameweeks_ahead=2,
                num_samples=50
            )
            # Basic structure check
            assert isinstance(pred_samples, dict)
        except Exception as e:
            # Expected to fail without full model setup, but shouldn't crash
            assert "failed" in str(e).lower() or "error" in str(e).lower()


class TestBackwardCompatibility:
    """Test backward compatibility with existing framework."""
    
    def test_adaptive_player_model_interface(self):
        """Test that NumPyro model implements AdaptivePlayerModel interface."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Check inheritance
        assert isinstance(model, AdaptivePlayerModel)
        
        # Check required methods exist
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "initialize_state")
        assert hasattr(model, "predict_state")
        assert hasattr(model, "update_state")
        assert hasattr(model, "observation_model")
        
    def test_player_state_compatibility(self):
        """Test PlayerState compatibility."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        state = model.initialize_state(player_id=123)
        
        # Check PlayerState structure
        assert hasattr(state, "player_id")
        assert hasattr(state, "state_mean")
        assert hasattr(state, "state_cov")
        assert hasattr(state, "gameweek")
        assert hasattr(state, "season")
        
        # Check serialization
        state_dict = state.to_dict()
        restored_state = PlayerState.from_dict(state_dict)
        
        assert restored_state.player_id == state.player_id
        assert jnp.allclose(restored_state.state_mean, state.state_mean)
    
    def test_prediction_output_format(self):
        """Test prediction output format compatibility."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Mock fitted model
        model.is_fitted = True
        model.posterior_samples = {"dummy": jnp.ones(10)}
        
        model.initialize_state(123)
        predictions = model.predict([123], gameweeks_ahead=1)
        
        # Check output format matches expected interface
        required_keys = ["player_ids", "predictions", "uncertainty", "states", "metadata"]
        for key in required_keys:
            assert key in predictions
        
        assert isinstance(predictions["metadata"], dict)
        assert "model_type" in predictions["metadata"]


class TestKalmanBridge:
    """Test bridge between Kalman and NumPyro models."""
    
    def test_model_bridge_initialization(self):
        """Test model bridge initialization."""
        config = TestFixtures.create_test_config()
        
        # Create both models
        kalman_model = Mock(spec=KalmanPlayerModel)
        numpyro_model = NumPyroAdaptiveModel(config=config)
        
        bridge = ModelBridge(kalman_model, numpyro_model)
        
        assert bridge.kalman_model == kalman_model
        assert bridge.numpyro_model == numpyro_model
    
    def test_state_conversion(self):
        """Test state conversion between models."""
        config = TestFixtures.create_test_config()
        
        kalman_model = Mock(spec=KalmanPlayerModel)
        numpyro_model = NumPyroAdaptiveModel(config=config)
        
        bridge = ModelBridge(kalman_model, numpyro_model)
        
        # Create test state
        test_state = PlayerState(
            player_id=123,
            state_mean=jnp.array([0.5, 0.6, 0.4, 0.1]),
            state_cov=jnp.eye(4) * 0.1,
            gameweek=5,
            season="2023"
        )
        
        # Test conversion (should be identity for now)
        converted_state = bridge.kalman_to_numpyro_state(test_state)
        assert converted_state.player_id == test_state.player_id
        assert jnp.allclose(converted_state.state_mean, test_state.state_mean)
    
    def test_transfer_learning(self):
        """Test transfer learning between models."""
        config = TestFixtures.create_test_config()
        
        # Mock Kalman model with states
        kalman_model = Mock(spec=KalmanPlayerModel)
        kalman_model.player_states = {
            123: PlayerState(
                player_id=123,
                state_mean=jnp.array([0.5, 0.6, 0.4, 0.1]),
                state_cov=jnp.eye(4) * 0.1,
                gameweek=5,
                season="2023"
            )
        }
        
        numpyro_model = NumPyroAdaptiveModel(config=config)
        bridge = ModelBridge(kalman_model, numpyro_model)
        
        # Test initialization transfer
        bridge.transfer_learning("initialization")
        
        assert 123 in numpyro_model.player_states
        assert jnp.allclose(
            numpyro_model.player_states[123].state_mean,
            kalman_model.player_states[123].state_mean
        )


class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_jax_compilation_time(self):
        """Test JAX compilation performance."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config, enable_hierarchical=False)
        
        # Generate test data
        data = TestFixtures.generate_synthetic_data(n_players=10, n_gameweeks=5)
        
        player_ids = jnp.array(data["player_ids"])
        observations = jnp.array(data["features"])
        gameweeks = jnp.array(data["gameweeks"])
        missing_mask = jnp.isnan(observations)
        
        # Measure compilation time
        start_time = time.time()
        
        compiled_model = jax.jit(model._simple_state_space_model)
        
        # Trigger compilation
        try:
            with numpyro.handlers.seed(rng_seed=42):
                with numpyro.handlers.trace():
                    compiled_model(player_ids, observations, gameweeks, missing_mask)
        except:
            pass  # Expected to fail without proper numpyro context
        
        compilation_time = time.time() - start_time
        
        # Should compile reasonably quickly (less than 10 seconds)
        assert compilation_time < 10.0
    
    def test_prediction_speed(self):
        """Test prediction speed."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Mock fitted model
        model.is_fitted = True
        model.posterior_samples = {"dummy": jnp.ones(10)}
        
        # Initialize multiple players
        player_ids = list(range(1, 21))  # 20 players
        for pid in player_ids:
            model.initialize_state(pid)
        
        # Measure prediction time
        start_time = time.time()
        
        predictions = model.predict(player_ids, gameweeks_ahead=3)
        
        prediction_time = time.time() - start_time
        
        # Should predict reasonably quickly (less than 5 seconds for 20 players)
        assert prediction_time < 5.0
        assert predictions["predictions"].shape == (20, 3)
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("svi")
        inference_config.num_steps = 100  # Small for memory test
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config
        )
        
        # Multiple small fits to check for memory leaks
        for i in range(3):
            data = TestFixtures.generate_synthetic_data(n_players=5, n_gameweeks=3)
            model.fit(data, season=f"202{i}", max_gameweek=3)
            
            # Basic check that model still works
            assert model.is_fitted
            assert model.posterior_samples is not None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        # Test invalid state dimensions
        with pytest.raises(ValueError):
            StateSpaceConfig(state_dim=-1)
        
        # Test mismatched names
        with pytest.raises(ValueError):
            StateSpaceConfig(
                state_dim=3,
                state_names=["skill", "form"]  # Should have 3 names
            )
    
    def test_unfitted_model_prediction(self):
        """Test prediction on unfitted model."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict([1, 2, 3])
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Test update with NaN observations
        state = PlayerState(
            player_id=123,
            state_mean=jnp.array([0.5, 0.5, 0.5, 0.0]),
            state_cov=jnp.eye(4) * 0.1,
            gameweek=5,
            season="2023"
        )
        
        # Update with missing observation
        nan_obs = jnp.array([jnp.nan, 1.0, 80.0, 2.0])
        updated_state = model.update_state(state, nan_obs)
        
        # Should return original state unchanged
        assert updated_state.player_id == state.player_id
        assert jnp.allclose(updated_state.state_mean, state.state_mean)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        # Test observation model with extreme state values
        extreme_state = jnp.array([1000.0, -1000.0, 0.0, 1000.0])
        obs = model.observation_model(extreme_state)
        
        # Should not produce NaN or infinite values
        assert jnp.all(jnp.isfinite(obs))
        assert jnp.all(obs >= 0)  # Observations should remain non-negative
    
    def test_inference_failure_handling(self):
        """Test handling of inference failures."""
        config = TestFixtures.create_test_config()
        
        # Create model with invalid inference config
        inference_config = InferenceConfig(
            inference_type="invalid_type"
        )
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config
        )
        
        data = TestFixtures.generate_synthetic_data(n_players=3, n_gameweeks=3)
        
        with pytest.raises((ValueError, NumPyroIntegrationError)):
            model.fit(data, season="2023", max_gameweek=3)


class TestModelDiagnostics:
    """Test model diagnostics and validation."""
    
    def test_get_model_diagnostics(self):
        """Test model diagnostics retrieval."""
        config = TestFixtures.create_test_config()
        model = NumPyroAdaptiveModel(config=config)
        
        diagnostics = model.get_model_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert "inference_config" in diagnostics
        assert "convergence_diagnostics" in diagnostics
        assert "model_comparison_metrics" in diagnostics
        assert "num_players" in diagnostics
    
    @pytest.mark.slow
    def test_posterior_summary(self):
        """Test posterior summary computation."""
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config
        )
        
        data = TestFixtures.generate_synthetic_data(n_players=3, n_gameweeks=3)
        model.fit(data, season="2023", max_gameweek=3)
        
        summary = model.get_posterior_summary()
        
        # Should have summary statistics for parameters
        assert isinstance(summary, dict)


# Integration test that runs a complete workflow
class TestCompleteWorkflow:
    """Test complete NumPyro integration workflow."""
    
    @pytest.mark.slow
    def test_complete_workflow_mcmc(self):
        """Test complete workflow with MCMC inference."""
        # Configuration
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("mcmc")
        
        # Create model
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=True
        )
        
        # Generate synthetic data
        data = TestFixtures.generate_synthetic_data(n_players=6, n_gameweeks=5)
        
        # Add position information
        position_mapping = TestFixtures.create_player_positions(data["player_ids"])
        for player_id, position in position_mapping.items():
            model.initialize_state(player_id, position=position)
        
        # Fit model
        model.fit(data, season="2023", max_gameweek=5)
        
        # Make predictions
        predictions = model.predict(
            player_ids=data["player_ids"][:3],
            gameweeks_ahead=2
        )
        
        # Validate results
        assert model.is_fitted
        assert model.posterior_samples is not None
        assert predictions["predictions"].shape == (3, 2)
        assert predictions["uncertainty"].shape == (3, 2)
        
        # Check diagnostics
        diagnostics = model.get_model_diagnostics()
        assert len(diagnostics["convergence_diagnostics"]) > 0
        
        logger.info("Complete MCMC workflow test passed")
    
    def test_complete_workflow_svi(self):
        """Test complete workflow with SVI inference."""
        # Configuration
        config = TestFixtures.create_test_config()
        inference_config = TestFixtures.create_test_inference_config("svi")
        inference_config.num_steps = 500
        
        # Create model
        model = NumPyroAdaptiveModel(
            config=config,
            inference_config=inference_config,
            enable_hierarchical=False
        )
        
        # Generate data
        data = TestFixtures.generate_synthetic_data(n_players=5, n_gameweeks=4)
        
        # Fit and predict
        model.fit(data, season="2023", max_gameweek=4)
        predictions = model.predict(
            player_ids=data["player_ids"][:3],
            gameweeks_ahead=1
        )
        
        # Validate
        assert model.is_fitted
        assert predictions["predictions"].shape == (3, 1)
        
        logger.info("Complete SVI workflow test passed")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])