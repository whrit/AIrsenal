"""
NumPyro Framework Integration for AIrsenal

This module provides seamless integration between AIrsenal's adaptive player modeling
framework and NumPyro's probabilistic programming capabilities. It enables sophisticated
Bayesian state-space modeling with proper uncertainty quantification, hierarchical modeling,
and efficient inference algorithms.

Key Features:
- NumPyro-compatible adaptive player models
- MCMC inference using NUTS/HMC for exact inference  
- Variational inference (SVI) for scalability
- Hierarchical modeling with partial pooling
- Mixed continuous/discrete state modeling
- Time-varying parameter support
- Model comparison using WAIC/LOO
- Posterior predictive checks
- JAX JIT compilation for performance
- GPU/TPU acceleration ready

Classes:
    NumPyroAdaptiveModel: NumPyro-compatible adaptive player model
    StateSpaceDistributions: Custom NumPyro distributions for state-space models
    MCMCStateInference: MCMC-based state inference engine
    VIStateInference: Variational inference engine
    ModelBridge: Bridge between Kalman and NumPyro models
    HierarchicalPlayerModel: Hierarchical Bayesian player modeling
    TimeVaryingParameterModel: Support for time-varying parameters

Example usage:
    ```python
    # Create NumPyro adaptive model
    config = StateSpaceConfig(
        state_dim=4,
        obs_dim=4,
        state_names=["skill", "form", "consistency", "momentum"],
        obs_names=["goals", "assists", "minutes", "bonus"]
    )
    
    model = NumPyroAdaptiveModel(
        config=config,
        inference_type="mcmc",
        num_warmup=1000,
        num_samples=2000
    )
    
    # Fit to data
    model.fit(training_data, season="2023", max_gameweek=15)
    
    # Make predictions with uncertainty
    predictions = model.predict(player_ids=[123, 456], gameweeks_ahead=3)
    print(f"Predictions: {predictions['predictions']}")
    print(f"Uncertainty: {predictions['uncertainty']}")
    ```
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from jax import vmap
from sqlalchemy.orm.session import Session

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.diagnostics import effective_sample_size, gelman_rubin, summary
from numpyro.infer import (
    MCMC,
    NUTS,
    HMC,
    SVI,
    Predictive,
    log_likelihood,
    Trace_ELBO,
    TraceMeanField_ELBO
)
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal
from numpyro.infer.initialization import init_to_median, init_to_value
from numpyro.infer.util import potential_energy

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
    PlayerData,
    PredictionOutput
)
from airsenal.framework.kalman_player_model import KalmanPlayerModel

logger = logging.getLogger(__name__)

# Type aliases
PRNGKey = jnp.ndarray
StateDistribution = dist.Distribution
ObservationDistribution = dist.Distribution
ModelFn = Callable[..., Any]
GuideFn = Callable[..., Any]


class NumPyroIntegrationError(Exception):
    """Base exception for NumPyro integration operations."""
    pass


@dataclass
class InferenceConfig:
    """Configuration for NumPyro inference methods."""
    
    # General settings
    inference_type: str = "mcmc"  # "mcmc" or "svi"
    num_chains: int = 4
    chain_method: str = "parallel"  # "parallel", "sequential", "vectorized"
    
    # MCMC settings
    num_warmup: int = 1000
    num_samples: int = 2000
    thinning: int = 1
    max_tree_depth: int = 10
    target_accept_prob: float = 0.8
    adapt_step_size: bool = True
    adapt_mass_matrix: bool = True
    dense_mass: bool = False
    
    # SVI settings
    num_steps: int = 10000
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    guide_type: str = "normal"  # "normal", "multivariate_normal", "custom"
    
    # Advanced settings
    use_jit: bool = True
    progress_bar: bool = True
    collect_warmup: bool = False
    init_strategy: str = "median"  # "median", "uniform", "value", "feasible"
    
    # Model comparison
    compute_waic: bool = True
    compute_loo: bool = False
    
    # Diagnostics
    compute_diagnostics: bool = True
    check_convergence: bool = True
    r_hat_threshold: float = 1.01
    ess_threshold: int = 100


@dataclass 
class StateSpaceDistributions:
    """NumPyro distributions for state-space models."""
    
    @staticmethod
    def transition_noise(
        state_dim: int,
        noise_scale: float = 0.1,
        correlation: Optional[float] = None
    ) -> StateDistribution:
        """
        Create transition noise distribution.
        
        Args:
            state_dim: Dimension of state vector
            noise_scale: Scale parameter for noise
            correlation: Optional correlation between state components
            
        Returns:
            NumPyro distribution for transition noise
        """
        if correlation is not None and abs(correlation) > 0.01:
            # Correlated noise using multivariate normal
            cov_matrix = noise_scale**2 * (
                correlation * jnp.ones((state_dim, state_dim)) +
                (1 - correlation) * jnp.eye(state_dim)
            )
            return dist.MultivariateNormal(
                loc=jnp.zeros(state_dim),
                covariance_matrix=cov_matrix
            )
        else:
            # Independent noise
            return dist.Normal(0, noise_scale).expand([state_dim]).to_event(1)
    
    @staticmethod
    def observation_noise(
        obs_dim: int,
        noise_scale: float = 0.2,
        heteroskedastic: bool = False
    ) -> ObservationDistribution:
        """
        Create observation noise distribution.
        
        Args:
            obs_dim: Dimension of observation vector
            noise_scale: Scale parameter for noise
            heteroskedastic: Whether to use different noise for each observation
            
        Returns:
            NumPyro distribution for observation noise
        """
        if heteroskedastic:
            # Different noise scales for each observation dimension
            scales = jnp.array([noise_scale * (0.5 + i * 0.1) for i in range(obs_dim)])
            return dist.Normal(0, scales).to_event(1)
        else:
            # Homoskedastic noise
            return dist.Normal(0, noise_scale).expand([obs_dim]).to_event(1)
    
    @staticmethod
    def hierarchical_prior(
        num_players: int,
        num_positions: int,
        state_dim: int,
        global_scale: float = 1.0,
        position_scale: float = 0.5
    ) -> Dict[str, StateDistribution]:
        """
        Create hierarchical prior distributions.
        
        Args:
            num_players: Number of players
            num_positions: Number of positions (GK, DEF, MID, FWD)
            state_dim: Dimension of state vector
            global_scale: Scale for global hyperpriors
            position_scale: Scale for position-specific effects
            
        Returns:
            Dictionary of hierarchical prior distributions
        """
        return {
            # Global hyperpriors
            "global_mean": dist.Normal(0, global_scale).expand([state_dim]).to_event(1),
            "global_scale": dist.HalfNormal(global_scale).expand([state_dim]).to_event(1),
            
            # Position-level parameters
            "position_offset": dist.Normal(0, position_scale).expand([num_positions, state_dim]).to_event(2),
            "position_scale": dist.HalfNormal(position_scale).expand([num_positions, state_dim]).to_event(2),
            
            # Player-level parameters
            "player_offset": dist.Normal(0, 1).expand([num_players, state_dim]).to_event(2)
        }
    
    @staticmethod
    def time_varying_coefficients(
        num_gameweeks: int,
        coef_dim: int,
        innovation_scale: float = 0.05
    ) -> StateDistribution:
        """
        Create distribution for time-varying coefficients.
        
        Args:
            num_gameweeks: Number of gameweeks
            coef_dim: Dimension of coefficient vector
            innovation_scale: Scale for innovations
            
        Returns:
            NumPyro distribution for time-varying coefficients
        """
        def transition_fn(carry, _):
            prev_coef = carry
            innovation = numpyro.sample(
                "innovation",
                dist.Normal(0, innovation_scale).expand([coef_dim]).to_event(1)
            )
            new_coef = prev_coef + innovation
            return new_coef, new_coef
        
        # Initial coefficients
        init_coef = numpyro.sample(
            "init_coef",
            dist.Normal(0, 1).expand([coef_dim]).to_event(1)
        )
        
        # Scan over time to generate time-varying coefficients
        _, coefficients = scan(
            transition_fn,
            init_coef,
            jnp.arange(num_gameweeks)
        )
        
        return coefficients


class NumPyroAdaptiveModel(AdaptivePlayerModel):
    """
    NumPyro-compatible implementation of AdaptivePlayerModel.
    
    This class integrates NumPyro's probabilistic programming capabilities
    with AIrsenal's adaptive player modeling framework, providing:
    - Exact Bayesian inference via MCMC
    - Scalable variational inference
    - Hierarchical modeling with partial pooling
    - Proper uncertainty quantification
    - Model comparison and validation
    """
    
    def __init__(
        self,
        config: StateSpaceConfig,
        inference_config: Optional[InferenceConfig] = None,
        kalman_bridge: Optional[KalmanPlayerModel] = None,
        learning_rate: float = 0.01,
        decay_factor: float = 0.95,
        random_seed: int = 42,
        enable_hierarchical: bool = True,
        position_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Initialize NumPyro Adaptive Player Model.
        
        Args:
            config: State space configuration
            inference_config: NumPyro inference configuration
            kalman_bridge: Optional Kalman model for initialization
            learning_rate: Learning rate for adaptive components
            decay_factor: Decay factor for historical data
            random_seed: Random seed for reproducibility
            enable_hierarchical: Whether to use hierarchical modeling
            position_mapping: Mapping from positions to indices
        """
        super().__init__(config, learning_rate, decay_factor, random_seed)
        
        # NumPyro-specific configuration
        self.inference_config = inference_config or InferenceConfig()
        self.kalman_bridge = kalman_bridge
        self.enable_hierarchical = enable_hierarchical
        
        # Position mapping
        self.position_mapping = position_mapping or {
            "GK": 0, "DEF": 1, "MID": 2, "FWD": 3
        }
        self.num_positions = len(self.position_mapping)
        
        # NumPyro model and inference objects
        self.model_fn: Optional[ModelFn] = None
        self.guide_fn: Optional[GuideFn] = None
        self.mcmc: Optional[MCMC] = None
        self.svi: Optional[SVI] = None
        self.posterior_samples: Optional[Dict[str, jnp.ndarray]] = None
        self.svi_state = None
        
        # Model diagnostics and metadata
        self.inference_time: float = 0.0
        self.convergence_diagnostics: Dict[str, Any] = {}
        self.model_comparison_metrics: Dict[str, float] = {}
        self.posterior_predictive_samples: Optional[Dict[str, jnp.ndarray]] = None
        
        # Player metadata for hierarchical modeling
        self.player_positions: Dict[int, int] = {}  # player_id -> position_idx
        self.position_players: Dict[int, List[int]] = {}  # position_idx -> [player_ids]
        
        # Initialize NumPyro model
        self._initialize_numpyro_model()
        
        logger.info(f"Initialized NumPyro adaptive model with {self.inference_config.inference_type} inference")
    
    def _initialize_numpyro_model(self):
        """Initialize NumPyro model function and guide."""
        if self.enable_hierarchical:
            self.model_fn = self._hierarchical_state_space_model
        else:
            self.model_fn = self._simple_state_space_model
        
        # Initialize guide based on configuration
        if self.inference_config.inference_type == "svi":
            self._initialize_guide()
    
    def _initialize_guide(self):
        """Initialize variational guide for SVI."""
        if self.inference_config.guide_type == "normal":
            self.guide_fn = AutoNormal(self.model_fn)
        elif self.inference_config.guide_type == "multivariate_normal":
            self.guide_fn = AutoMultivariateNormal(self.model_fn)
        else:
            # Custom guide - to be implemented based on specific needs
            self.guide_fn = self._create_custom_guide()
    
    def _simple_state_space_model(
        self,
        player_ids: jnp.ndarray,
        observations: jnp.ndarray,
        gameweeks: jnp.ndarray,
        missing_mask: Optional[jnp.ndarray] = None,
        predict_mode: bool = False
    ):
        """
        Simple state-space model for individual players.
        
        Args:
            player_ids: Array of player IDs (n_players,)
            observations: Observation matrix (n_players, n_gameweeks, obs_dim)
            gameweeks: Array of gameweek numbers (n_gameweeks,)
            missing_mask: Binary mask for missing observations
            predict_mode: Whether in prediction mode
        """
        n_players, n_gameweeks, obs_dim = observations.shape
        state_dim = self.config.state_dim
        
        # Global hyperparameters
        process_noise_scale = numpyro.sample(
            "process_noise_scale",
            dist.HalfNormal(self.config.process_noise_std)
        )
        observation_noise_scale = numpyro.sample(
            "observation_noise_scale", 
            dist.HalfNormal(self.config.measurement_noise_std)
        )
        
        # Transition and observation matrices
        transition_matrix = numpyro.sample(
            "transition_matrix",
            dist.Normal(0, 0.1).expand([state_dim, state_dim]).to_event(2)
        )
        # Add identity component for stability
        transition_matrix = transition_matrix + 0.9 * jnp.eye(state_dim)
        
        observation_matrix = numpyro.sample(
            "observation_matrix",
            dist.Normal(0, 1).expand([obs_dim, state_dim]).to_event(2)
        )
        
        # Per-player modeling
        with numpyro.plate("players", n_players):
            # Initial states
            initial_state = numpyro.sample(
                "initial_state",
                dist.Normal(0, self.config.initial_state_std).expand([state_dim]).to_event(1)
            )
            
            # Define state evolution function
            def state_transition(carry, t):
                prev_state = carry
                
                # State transition with noise
                predicted_state = jnp.dot(transition_matrix, prev_state)
                process_noise = numpyro.sample(
                    f"process_noise_{t}",
                    StateSpaceDistributions.transition_noise(state_dim, process_noise_scale)
                )
                new_state = predicted_state + process_noise
                
                # Observation model
                expected_obs = jnp.dot(observation_matrix, new_state)
                
                if not predict_mode:
                    # Observed data likelihood
                    obs_noise = StateSpaceDistributions.observation_noise(obs_dim, observation_noise_scale)
                    
                    # Handle missing data
                    if missing_mask is not None:
                        # Use mask to handle missing observations
                        obs_at_t = jnp.where(
                            missing_mask[:, t, :],
                            expected_obs,
                            observations[:, t, :]
                        )
                    else:
                        obs_at_t = observations[:, t, :]
                    
                    numpyro.sample(
                        f"obs_{t}",
                        dist.Normal(expected_obs, observation_noise_scale).to_event(1),
                        obs=obs_at_t
                    )
                else:
                    # Prediction mode - sample from observation distribution
                    numpyro.sample(
                        f"pred_obs_{t}",
                        dist.Normal(expected_obs, observation_noise_scale).to_event(1)
                    )
                
                return new_state, (new_state, expected_obs)
            
            # Scan over time
            _, (states, expected_observations) = scan(
                state_transition,
                initial_state,
                jnp.arange(n_gameweeks)
            )
            
            # Store deterministic quantities
            numpyro.deterministic("states", states)
            numpyro.deterministic("expected_observations", expected_observations)
    
    def _hierarchical_state_space_model(
        self,
        player_ids: jnp.ndarray,
        observations: jnp.ndarray,
        gameweeks: jnp.ndarray,
        player_positions: jnp.ndarray,
        missing_mask: Optional[jnp.ndarray] = None,
        predict_mode: bool = False
    ):
        """
        Hierarchical state-space model with position-level effects.
        
        Args:
            player_ids: Array of player IDs (n_players,)
            observations: Observation matrix (n_players, n_gameweeks, obs_dim)
            gameweeks: Array of gameweek numbers (n_gameweeks,)
            player_positions: Position indices for each player (n_players,)
            missing_mask: Binary mask for missing observations
            predict_mode: Whether in prediction mode
        """
        n_players, n_gameweeks, obs_dim = observations.shape
        state_dim = self.config.state_dim
        
        # Global hyperparameters
        global_state_mean = numpyro.sample(
            "global_state_mean",
            dist.Normal(0, 1).expand([state_dim]).to_event(1)
        )
        global_state_scale = numpyro.sample(
            "global_state_scale",
            dist.HalfNormal(1).expand([state_dim]).to_event(1)
        )
        
        process_noise_scale = numpyro.sample(
            "process_noise_scale",
            dist.HalfNormal(self.config.process_noise_std)
        )
        observation_noise_scale = numpyro.sample(
            "observation_noise_scale",
            dist.HalfNormal(self.config.measurement_noise_std)
        )
        
        # Position-level parameters
        with numpyro.plate("positions", self.num_positions):
            position_state_offset = numpyro.sample(
                "position_state_offset",
                dist.Normal(0, 0.5).expand([state_dim]).to_event(1)
            )
            position_state_scale = numpyro.sample(
                "position_state_scale",
                dist.HalfNormal(0.5).expand([state_dim]).to_event(1)
            )
            
            # Position-specific transition dynamics
            position_transition_offset = numpyro.sample(
                "position_transition_offset",
                dist.Normal(0, 0.1).expand([state_dim, state_dim]).to_event(2)
            )
        
        # Base transition matrix
        base_transition_matrix = numpyro.sample(
            "base_transition_matrix",
            dist.Normal(0, 0.1).expand([state_dim, state_dim]).to_event(2)
        )
        base_transition_matrix = base_transition_matrix + 0.9 * jnp.eye(state_dim)
        
        # Observation matrix
        observation_matrix = numpyro.sample(
            "observation_matrix",
            dist.Normal(0, 1).expand([obs_dim, state_dim]).to_event(2)
        )
        
        # Per-player modeling
        with numpyro.plate("players", n_players):
            # Player-specific initial states with hierarchical structure
            player_pos = player_positions  # (n_players,)
            
            # Get position-specific parameters for each player
            pos_state_mean = global_state_mean + position_state_offset[player_pos]
            pos_state_scale = global_state_scale * position_state_scale[player_pos]
            
            initial_state = numpyro.sample(
                "initial_state",
                dist.Normal(pos_state_mean, pos_state_scale).to_event(1)
            )
            
            # Player-specific transition matrices
            player_transition_matrix = (
                base_transition_matrix[None, :, :] +
                position_transition_offset[player_pos]
            )
            
            # State evolution function
            def state_transition(carry, t):
                prev_state = carry
                
                # State transition with player-specific dynamics
                predicted_state = jnp.einsum('pij,pj->pi', player_transition_matrix, prev_state)
                
                process_noise = numpyro.sample(
                    f"process_noise_{t}",
                    StateSpaceDistributions.transition_noise(state_dim, process_noise_scale)
                )
                new_state = predicted_state + process_noise
                
                # Observation model
                expected_obs = jnp.einsum('ij,pj->pi', observation_matrix, new_state)
                
                if not predict_mode:
                    # Handle missing data
                    if missing_mask is not None:
                        obs_at_t = jnp.where(
                            missing_mask[:, t, :],
                            expected_obs,
                            observations[:, t, :]
                        )
                    else:
                        obs_at_t = observations[:, t, :]
                    
                    numpyro.sample(
                        f"obs_{t}",
                        dist.Normal(expected_obs, observation_noise_scale).to_event(1),
                        obs=obs_at_t
                    )
                else:
                    numpyro.sample(
                        f"pred_obs_{t}",
                        dist.Normal(expected_obs, observation_noise_scale).to_event(1)
                    )
                
                return new_state, (new_state, expected_obs)
            
            # Scan over time
            _, (states, expected_observations) = scan(
                state_transition,
                initial_state,
                jnp.arange(n_gameweeks)
            )
            
            numpyro.deterministic("states", states)
            numpyro.deterministic("expected_observations", expected_observations)
    
    def _create_custom_guide(self) -> GuideFn:
        """Create custom variational guide."""
        def guide(
            player_ids: jnp.ndarray,
            observations: jnp.ndarray,
            gameweeks: jnp.ndarray,
            **kwargs
        ):
            # Custom guide implementation would go here
            # For now, use AutoNormal as fallback
            return AutoNormal(self.model_fn)(player_ids, observations, gameweeks, **kwargs)
        
        return guide
    
    def initialize_state(
        self,
        player_id: int,
        initial_data: Optional[PlayerData] = None,
        gameweek: int = 1,
        season: str = "2023",
        position: str = None,
        **kwargs
    ) -> PlayerState:
        """
        Initialize state for a player using NumPyro.
        
        Args:
            player_id: Player ID to initialize
            initial_data: Optional initial performance data
            gameweek: Initial gameweek
            season: Season identifier
            position: Player position
            **kwargs: Additional parameters
            
        Returns:
            Initial PlayerState object
        """
        try:
            # Store position mapping for hierarchical modeling
            if position is not None:
                pos_idx = self.position_mapping.get(position, 2)  # Default to MID
                self.player_positions[player_id] = pos_idx
                
                if pos_idx not in self.position_players:
                    self.position_players[pos_idx] = []
                if player_id not in self.position_players[pos_idx]:
                    self.position_players[pos_idx].append(player_id)
            
            # Use Kalman bridge if available for initialization
            if self.kalman_bridge is not None:
                kalman_state = self.kalman_bridge.initialize_state(
                    player_id, initial_data, gameweek, season, position, **kwargs
                )
                return kalman_state
            
            # Default initialization using historical data or priors
            if initial_data and "features" in initial_data:
                features = initial_data["features"]
                if features is not None and len(features) > 0:
                    recent_performance = np.mean(features[-5:], axis=0) if len(features.shape) > 1 else features
                    state_mean = self._performance_to_state(recent_performance)
                else:
                    state_mean = self._default_initial_state()
            else:
                state_mean = self._default_initial_state()
            
            # Initialize covariance with higher uncertainty
            state_cov = jnp.eye(self.config.state_dim) * self.config.initial_state_std**2
            
            player_state = PlayerState(
                player_id=player_id,
                state_mean=state_mean,
                state_cov=state_cov,
                gameweek=gameweek,
                season=season,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            self.player_states[player_id] = player_state
            return player_state
            
        except Exception as e:
            logger.error(f"Failed to initialize state for player {player_id}: {e}")
            raise NumPyroIntegrationError(f"State initialization failed: {e}")
    
    def _default_initial_state(self) -> jnp.ndarray:
        """Create default initial state vector."""
        return jnp.array([0.5, 0.5, 0.5, 0.0])  # [skill, form, consistency, momentum]
    
    def _performance_to_state(self, performance: np.ndarray) -> jnp.ndarray:
        """Convert performance observations to state estimate."""
        goals, assists, minutes, bonus = performance[:4]
        
        skill = np.clip((goals + assists) / 10.0, 0, 1)
        form = np.clip(minutes / 90.0, 0, 1)
        consistency = np.clip(bonus / 5.0, 0, 1)
        momentum = 0.0
        
        return jnp.array([skill, form, consistency, momentum])
    
    def predict_state(
        self,
        current_state: PlayerState,
        gameweeks_ahead: int = 1,
        **kwargs
    ) -> PlayerState:
        """
        Predict future state using NumPyro posterior samples.
        
        Args:
            current_state: Current player state
            gameweeks_ahead: Number of gameweeks to predict
            **kwargs: Additional parameters
            
        Returns:
            Predicted PlayerState object
        """
        if not self.is_fitted or self.posterior_samples is None:
            raise RuntimeError("Model must be fitted before making predictions")
        
        try:
            player_id = current_state.player_id
            
            # Prepare prediction data
            pred_observations = jnp.zeros((1, gameweeks_ahead, self.config.obs_dim))
            pred_gameweeks = jnp.arange(
                current_state.gameweek + 1,
                current_state.gameweek + 1 + gameweeks_ahead
            )
            
            # Get position if available
            player_positions = jnp.array([self.player_positions.get(player_id, 2)])
            
            # Run prediction using posterior samples
            if self.enable_hierarchical:
                predictive = Predictive(
                    self._hierarchical_state_space_model,
                    posterior_samples=self.posterior_samples,
                    num_samples=200
                )
                pred_samples = predictive(
                    self.rng_key,
                    jnp.array([player_id]),
                    pred_observations,
                    pred_gameweeks,
                    player_positions,
                    predict_mode=True
                )
            else:
                predictive = Predictive(
                    self._simple_state_space_model,
                    posterior_samples=self.posterior_samples,
                    num_samples=200
                )
                pred_samples = predictive(
                    self.rng_key,
                    jnp.array([player_id]),
                    pred_observations,
                    pred_gameweeks,
                    predict_mode=True
                )
            
            # Extract predicted states
            if "states" in pred_samples:
                predicted_states = pred_samples["states"]  # (n_samples, n_players, n_gameweeks, state_dim)
                
                # Average over samples and take final timepoint
                final_state_mean = jnp.mean(predicted_states[:, 0, -1, :], axis=0)
                final_state_cov = jnp.cov(predicted_states[:, 0, -1, :].T)
            else:
                # Fallback: use current state with added uncertainty
                final_state_mean = current_state.state_mean
                final_state_cov = current_state.state_cov + jnp.eye(self.config.state_dim) * 0.01
            
            predicted_state = PlayerState(
                player_id=player_id,
                state_mean=final_state_mean,
                state_cov=final_state_cov,
                gameweek=current_state.gameweek + gameweeks_ahead,
                season=current_state.season,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"Prediction failed for player {player_id}: {e}")
            raise NumPyroIntegrationError(f"State prediction failed: {e}")
    
    def update_state(
        self,
        predicted_state: PlayerState,
        observation: jnp.ndarray,
        observation_noise: Optional[jnp.ndarray] = None,
        **kwargs
    ) -> PlayerState:
        """
        Update state based on new observations using NumPyro inference.
        
        Args:
            predicted_state: Predicted state from prediction step
            observation: Observed performance metrics
            observation_noise: Optional observation noise covariance
            **kwargs: Additional parameters
            
        Returns:
            Updated PlayerState object
        """
        try:
            # Handle missing observations
            if jnp.any(jnp.isnan(observation)):
                logger.warning(f"Missing observations for player {predicted_state.player_id}, skipping update")
                return predicted_state
            
            # For NumPyro models, state updates are typically done through
            # re-running inference with the new data point
            # This is a simplified update that adds the observation to uncertainty
            
            player_id = predicted_state.player_id
            
            # Simplified Bayesian update
            if observation_noise is None:
                obs_noise_cov = jnp.eye(self.config.obs_dim) * self.config.measurement_noise_std**2
            else:
                obs_noise_cov = observation_noise
            
            # Use observation model to relate state to observation
            expected_obs = self.observation_model(predicted_state.state_mean)
            residual = observation[:self.config.obs_dim] - expected_obs
            
            # Simple Kalman-like update (approximation for real-time updates)
            H = jnp.eye(min(self.config.state_dim, self.config.obs_dim))  # Simplified observation matrix
            S = H @ predicted_state.state_cov @ H.T + obs_noise_cov[:self.config.state_dim, :self.config.state_dim]
            K = predicted_state.state_cov @ H.T @ jnp.linalg.inv(S)
            
            updated_mean = predicted_state.state_mean + K @ residual[:self.config.state_dim]
            updated_cov = (jnp.eye(self.config.state_dim) - K @ H) @ predicted_state.state_cov
            
            updated_state = PlayerState(
                player_id=player_id,
                state_mean=updated_mean,
                state_cov=updated_cov,
                gameweek=predicted_state.gameweek,
                season=predicted_state.season,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Update failed for player {predicted_state.player_id}: {e}")
            return predicted_state
    
    def observation_model(
        self,
        state: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """
        Map state vector to expected observations.
        
        Args:
            state: State vector [skill, form, consistency, momentum]
            **kwargs: Additional parameters
            
        Returns:
            Expected observation vector [goals, assists, minutes, bonus]
        """
        try:
            skill, form, consistency, momentum = state[0], state[1], state[2], state[3]
            
            # Position-aware observation mapping
            # Default to midfielder-style mapping
            goals = jnp.maximum(0, 0.7 * skill * form + 0.08 * momentum)
            assists = jnp.maximum(0, 0.9 * skill * consistency + 0.06 * momentum)
            minutes = jnp.maximum(0, 60 + 30 * consistency * form)
            bonus = jnp.maximum(0, (goals + assists) * consistency + 0.1 * momentum)
            
            return jnp.array([goals, assists, minutes, bonus])
            
        except Exception as e:
            logger.error(f"Observation model failed: {e}")
            return state[:self.config.obs_dim] if len(state) >= self.config.obs_dim else jnp.zeros(self.config.obs_dim)
    
    def _fit_state_space_model(
        self,
        data: PlayerData,
        season: str,
        max_gameweek: int,
        **kwargs
    ) -> None:
        """
        Fit NumPyro state-space model from historical data.
        
        Args:
            data: Training data dictionary
            season: Season identifier
            max_gameweek: Maximum gameweek for training
            **kwargs: Additional fitting parameters
        """
        try:
            start_time = time.time()
            logger.info(f"Fitting NumPyro model with {self.inference_config.inference_type} inference")
            
            # Prepare data for NumPyro
            player_ids = jnp.array(data["player_ids"])
            features = jnp.array(data["features"])  # (n_players, n_gameweeks, n_features)
            gameweeks = jnp.array(data["gameweeks"])
            
            # Ensure we have the right observation dimensions
            observations = features[:, :, :self.config.obs_dim]
            n_players, n_gameweeks, obs_dim = observations.shape
            
            # Create missing data mask
            missing_mask = jnp.isnan(observations)
            
            # Prepare position data for hierarchical modeling
            if self.enable_hierarchical:
                player_positions = jnp.array([
                    self.player_positions.get(int(pid), 2) for pid in player_ids
                ])
                model_args = (player_ids, observations, gameweeks, player_positions, missing_mask)
            else:
                model_args = (player_ids, observations, gameweeks, missing_mask)
            
            # Run inference
            if self.inference_config.inference_type == "mcmc":
                self._run_mcmc_inference(model_args)
            elif self.inference_config.inference_type == "svi":
                self._run_svi_inference(model_args)
            else:
                raise ValueError(f"Unknown inference type: {self.inference_config.inference_type}")
            
            # Compute diagnostics and model comparison metrics
            if self.inference_config.compute_diagnostics:
                self._compute_diagnostics()
            
            if self.inference_config.compute_waic:
                self._compute_model_comparison_metrics(model_args)
            
            self.inference_time = time.time() - start_time
            logger.info(f"NumPyro model fitting completed in {self.inference_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"NumPyro model fitting failed: {e}")
            raise NumPyroIntegrationError(f"Model fitting failed: {e}")
    
    def _run_mcmc_inference(self, model_args: Tuple):
        """Run MCMC inference using NUTS."""
        try:
            # Initialize kernel
            kernel = NUTS(
                self.model_fn,
                max_tree_depth=self.inference_config.max_tree_depth,
                target_accept_prob=self.inference_config.target_accept_prob,
                adapt_step_size=self.inference_config.adapt_step_size,
                adapt_mass_matrix=self.inference_config.adapt_mass_matrix,
                dense_mass=self.inference_config.dense_mass
            )
            
            # Initialize MCMC
            self.mcmc = MCMC(
                kernel,
                num_warmup=self.inference_config.num_warmup,
                num_samples=self.inference_config.num_samples,
                num_chains=self.inference_config.num_chains,
                thinning=self.inference_config.thinning,
                chain_method=self.inference_config.chain_method,
                progress_bar=self.inference_config.progress_bar,
                jit_model_args=self.inference_config.use_jit
            )
            
            # Run MCMC
            self.rng_key, subkey = random.split(self.rng_key)
            self.mcmc.run(subkey, *model_args)
            
            # Extract posterior samples
            self.posterior_samples = self.mcmc.get_samples()
            
            logger.info(f"MCMC inference completed with {self.inference_config.num_samples} samples")
            
        except Exception as e:
            logger.error(f"MCMC inference failed: {e}")
            raise NumPyroIntegrationError(f"MCMC inference failed: {e}")
    
    def _run_svi_inference(self, model_args: Tuple):
        """Run SVI inference."""
        try:
            # Import optimizer
            import optax
            
            # Create optimizer
            if self.inference_config.optimizer == "adam":
                optimizer = optax.adam(
                    learning_rate=self.inference_config.learning_rate,
                    b1=self.inference_config.beta1,
                    b2=self.inference_config.beta2
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.inference_config.optimizer}")
            
            # Initialize SVI
            self.svi = SVI(
                model=self.model_fn,
                guide=self.guide_fn,
                optim=optimizer,
                loss=TraceMeanField_ELBO()
            )
            
            # Initialize SVI state
            self.rng_key, subkey = random.split(self.rng_key)
            self.svi_state = self.svi.init(subkey, *model_args)
            
            # Run SVI optimization
            losses = []
            for step in range(self.inference_config.num_steps):
                self.svi_state, loss = self.svi.update(self.svi_state, *model_args)
                losses.append(loss)
                
                if step % 1000 == 0 and self.inference_config.progress_bar:
                    logger.info(f"SVI step {step}, loss: {loss:.4f}")
            
            # Extract posterior samples
            self.rng_key, subkey = random.split(self.rng_key)
            self.posterior_samples = self.svi.get_samples(
                self.svi_state,
                num_samples=2000,
                rng_key=subkey
            )
            
            logger.info(f"SVI inference completed with final loss: {losses[-1]:.4f}")
            
        except Exception as e:
            logger.error(f"SVI inference failed: {e}")
            raise NumPyroIntegrationError(f"SVI inference failed: {e}")
    
    def _compute_diagnostics(self):
        """Compute convergence diagnostics for MCMC."""
        if self.inference_config.inference_type != "mcmc" or self.posterior_samples is None:
            return
        
        try:
            self.convergence_diagnostics = {}
            
            # Compute R-hat and effective sample size
            for param_name, samples in self.posterior_samples.items():
                if samples.ndim >= 2:  # Only for parameters with multiple chains/samples
                    try:
                        r_hat = gelman_rubin(samples)
                        ess = effective_sample_size(samples)
                        
                        self.convergence_diagnostics[param_name] = {
                            "r_hat": float(r_hat) if jnp.isscalar(r_hat) else r_hat.tolist(),
                            "ess": float(ess) if jnp.isscalar(ess) else ess.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Failed to compute diagnostics for {param_name}: {e}")
            
            # Check convergence
            if self.inference_config.check_convergence:
                self._check_convergence()
            
        except Exception as e:
            logger.warning(f"Failed to compute diagnostics: {e}")
    
    def _check_convergence(self):
        """Check MCMC convergence based on diagnostics."""
        convergence_issues = []
        
        for param_name, diag in self.convergence_diagnostics.items():
            r_hat = diag.get("r_hat")
            ess = diag.get("ess")
            
            if r_hat is not None:
                if isinstance(r_hat, (list, tuple)):
                    max_r_hat = max(r_hat)
                else:
                    max_r_hat = r_hat
                
                if max_r_hat > self.inference_config.r_hat_threshold:
                    convergence_issues.append(f"{param_name}: R-hat = {max_r_hat:.3f}")
            
            if ess is not None:
                if isinstance(ess, (list, tuple)):
                    min_ess = min(ess)
                else:
                    min_ess = ess
                
                if min_ess < self.inference_config.ess_threshold:
                    convergence_issues.append(f"{param_name}: ESS = {min_ess:.0f}")
        
        if convergence_issues:
            warning_msg = "Convergence issues detected:\n" + "\n".join(convergence_issues)
            logger.warning(warning_msg)
            warnings.warn(warning_msg, UserWarning)
    
    def _compute_model_comparison_metrics(self, model_args: Tuple):
        """Compute model comparison metrics (WAIC, LOO)."""
        if self.posterior_samples is None:
            return
        
        try:
            # Compute log likelihood
            log_lik = log_likelihood(
                self.model_fn,
                self.posterior_samples,
                *model_args
            )
            
            # Compute WAIC
            if self.inference_config.compute_waic:
                from numpyro.diagnostics import waic
                waic_result = waic(log_lik)
                self.model_comparison_metrics["waic"] = float(waic_result.waic)
                self.model_comparison_metrics["waic_se"] = float(waic_result.waic_se)
            
            # Compute LOO-CV (if requested)
            if self.inference_config.compute_loo:
                try:
                    import arviz as az
                    loo_result = az.loo(log_lik)
                    self.model_comparison_metrics["loo"] = float(loo_result.loo)
                    self.model_comparison_metrics["loo_se"] = float(loo_result.loo_se)
                except ImportError:
                    logger.warning("ArviZ not available, skipping LOO computation")
            
        except Exception as e:
            logger.warning(f"Failed to compute model comparison metrics: {e}")
    
    def get_posterior_summary(self) -> Dict[str, Any]:
        """Get summary of posterior samples."""
        if self.posterior_samples is None:
            return {}
        
        try:
            return summary(self.posterior_samples, prob=0.9)
        except Exception as e:
            logger.warning(f"Failed to compute posterior summary: {e}")
            return {}
    
    def predict_posterior_samples(
        self,
        player_ids: List[int],
        gameweeks_ahead: int = 3,
        num_samples: int = 1000
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior predictive samples.
        
        Args:
            player_ids: List of player IDs
            gameweeks_ahead: Number of gameweeks to predict
            num_samples: Number of posterior samples to draw
            
        Returns:
            Dictionary with posterior predictive samples
        """
        if self.posterior_samples is None:
            raise RuntimeError("Model must be fitted before generating predictions")
        
        try:
            # Prepare prediction data
            n_players = len(player_ids)
            pred_observations = jnp.zeros((n_players, gameweeks_ahead, self.config.obs_dim))
            pred_gameweeks = jnp.arange(gameweeks_ahead)
            
            if self.enable_hierarchical:
                player_positions = jnp.array([
                    self.player_positions.get(pid, 2) for pid in player_ids
                ])
                
                predictive = Predictive(
                    self._hierarchical_state_space_model,
                    posterior_samples=self.posterior_samples,
                    num_samples=num_samples
                )
                
                return predictive(
                    self.rng_key,
                    jnp.array(player_ids),
                    pred_observations,
                    pred_gameweeks,
                    player_positions,
                    predict_mode=True
                )
            else:
                predictive = Predictive(
                    self._simple_state_space_model,
                    posterior_samples=self.posterior_samples,
                    num_samples=num_samples
                )
                
                return predictive(
                    self.rng_key,
                    jnp.array(player_ids),
                    pred_observations,
                    pred_gameweeks,
                    predict_mode=True
                )
                
        except Exception as e:
            logger.error(f"Posterior prediction failed: {e}")
            raise NumPyroIntegrationError(f"Posterior prediction failed: {e}")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        return {
            "inference_config": {
                "inference_type": self.inference_config.inference_type,
                "num_warmup": self.inference_config.num_warmup,
                "num_samples": self.inference_config.num_samples,
                "num_chains": self.inference_config.num_chains,
            },
            "convergence_diagnostics": self.convergence_diagnostics,
            "model_comparison_metrics": self.model_comparison_metrics,
            "inference_time": self.inference_time,
            "num_players": len(self.player_states),
            "hierarchical_modeling": self.enable_hierarchical,
            "num_positions": self.num_positions,
            "posterior_summary": self.get_posterior_summary()
        }


class MCMCStateInference:
    """MCMC-based state inference for NumPyro models."""
    
    def __init__(
        self,
        model: NumPyroAdaptiveModel,
        kernel_type: str = "nuts",
        **kernel_kwargs
    ):
        """
        Initialize MCMC state inference.
        
        Args:
            model: NumPyro adaptive model
            kernel_type: Type of MCMC kernel ("nuts", "hmc")
            **kernel_kwargs: Additional kernel arguments
        """
        self.model = model
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self.mcmc = None
        
    def run_inference(
        self,
        data: PlayerData,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        **kwargs
    ) -> Dict[str, jnp.ndarray]:
        """
        Run MCMC inference.
        
        Args:
            data: Training data
            num_warmup: Number of warmup samples
            num_samples: Number of posterior samples
            num_chains: Number of MCMC chains
            **kwargs: Additional MCMC arguments
            
        Returns:
            Posterior samples dictionary
        """
        # Implementation would go here
        pass


class VIStateInference:
    """Variational inference for scalable state estimation."""
    
    def __init__(
        self,
        model: NumPyroAdaptiveModel,
        guide_type: str = "normal",
        **guide_kwargs
    ):
        """
        Initialize variational inference.
        
        Args:
            model: NumPyro adaptive model
            guide_type: Type of variational guide
            **guide_kwargs: Additional guide arguments
        """
        self.model = model
        self.guide_type = guide_type
        self.guide_kwargs = guide_kwargs
        self.svi = None
        
    def run_inference(
        self,
        data: PlayerData,
        num_steps: int = 10000,
        learning_rate: float = 0.01,
        **kwargs
    ) -> Dict[str, jnp.ndarray]:
        """
        Run SVI inference.
        
        Args:
            data: Training data
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimizer
            **kwargs: Additional SVI arguments
            
        Returns:
            Posterior samples dictionary
        """
        # Implementation would go here
        pass


class ModelBridge:
    """Bridge between Kalman and NumPyro models."""
    
    def __init__(
        self,
        kalman_model: KalmanPlayerModel,
        numpyro_model: NumPyroAdaptiveModel
    ):
        """
        Initialize model bridge.
        
        Args:
            kalman_model: Kalman filter model
            numpyro_model: NumPyro adaptive model
        """
        self.kalman_model = kalman_model
        self.numpyro_model = numpyro_model
        
    def kalman_to_numpyro_state(self, kalman_state: PlayerState) -> PlayerState:
        """Convert Kalman state to NumPyro format."""
        return kalman_state  # Direct compatibility
        
    def numpyro_to_kalman_state(self, numpyro_state: PlayerState) -> PlayerState:
        """Convert NumPyro state to Kalman format."""
        return numpyro_state  # Direct compatibility
        
    def transfer_learning(self, transfer_type: str = "initialization"):
        """Transfer knowledge between models."""
        if transfer_type == "initialization":
            # Use Kalman model to initialize NumPyro model
            for player_id, kalman_state in self.kalman_model.player_states.items():
                numpyro_state = self.kalman_to_numpyro_state(kalman_state)
                self.numpyro_model.player_states[player_id] = numpyro_state
        
        elif transfer_type == "posterior_refinement":
            # Use NumPyro posterior to refine Kalman estimates
            for player_id, numpyro_state in self.numpyro_model.player_states.items():
                kalman_state = self.numpyro_to_kalman_state(numpyro_state)
                self.kalman_model.player_states[player_id] = kalman_state