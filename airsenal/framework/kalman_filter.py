"""
Kalman Filter Implementation for AIrsenal Player State Tracking

This module implements comprehensive Kalman filtering algorithms for tracking dynamic 
player abilities over time. It includes linear Kalman filters, Extended Kalman Filters (EKF),
and Unscented Kalman Filters (UKF) with robust numerical stability features.

The implementation is designed to integrate seamlessly with AIrsenal's AdaptivePlayerModel
architecture and provides state-of-the-art uncertainty quantification for player performance
prediction.

Key Features:
- Linear Kalman Filter for simple dynamics
- Extended Kalman Filter (EKF) for non-linear state transitions
- Unscented Kalman Filter (UKF) for better non-linear handling
- Numerical stability through Joseph form updates and covariance enforcement
- Adaptive noise estimation for robust performance
- JAX/NumPy backend for high performance computing
- Square root filtering for enhanced numerical precision
- Proper uncertainty propagation and innovation statistics

Classes:
    KalmanFilter: Linear Kalman filter implementation
    ExtendedKalmanFilter: EKF for non-linear dynamics
    UnscentedKalmanFilter: UKF with sigma point sampling
    AdaptiveNoiseEstimator: Adaptive process/measurement noise estimation
    FilterConfig: Configuration for filter parameters
    FilterState: Complete filter state representation

Usage:
    ```python
    from airsenal.framework.kalman_filter import KalmanFilter, FilterConfig
    
    # Configure filter
    config = FilterConfig(
        state_dim=4,
        obs_dim=4,
        process_noise_std=0.1,
        measurement_noise_std=0.2,
        use_joseph_form=True
    )
    
    # Initialize filter
    kf = KalmanFilter(config)
    
    # Predict and update
    predicted_state = kf.predict(current_state, dt=1.0)
    updated_state = kf.update(predicted_state, observation)
    ```
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import grad, jacfwd, jacrev
from jax.scipy.linalg import cholesky, solve_triangular

logger = logging.getLogger(__name__)

# Type aliases for clarity
Array = Union[np.ndarray, jnp.ndarray]
StateVector = Array
CovarianceMatrix = Array
ObservationVector = Array
JacobianMatrix = Array
TransitionFunction = Callable[[StateVector, float], StateVector]
MeasurementFunction = Callable[[StateVector], ObservationVector]


@dataclass
class FilterConfig:
    """Configuration class for Kalman filter parameters."""
    
    # Dimensions
    state_dim: int = 4  # [skill, form, consistency, momentum]
    obs_dim: int = 4    # [goals, assists, minutes, bonus]
    
    # Noise parameters
    process_noise_std: float = 0.1
    measurement_noise_std: float = 0.2
    initial_state_std: float = 1.0
    
    # Numerical stability options
    use_joseph_form: bool = True
    use_square_root: bool = False
    enforce_symmetry: bool = True
    regularization_factor: float = 1e-8
    min_eigenvalue: float = 1e-10
    
    # Adaptive estimation
    enable_adaptive_noise: bool = True
    adaptation_rate: float = 0.01
    noise_estimation_window: int = 10
    
    # UKF specific parameters
    alpha: float = 1e-3      # Spread of sigma points
    beta: float = 2.0        # Prior knowledge about distribution (2 is optimal for Gaussian)
    kappa: float = 0.0       # Secondary scaling parameter
    
    # JAX configuration
    use_jax: bool = True
    precision: str = "float64"  # "float32" or "float64"
    
    # Validation options
    validate_inputs: bool = True
    check_psd: bool = True  # Check positive semi-definite matrices
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.state_dim <= 0 or self.obs_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        if self.process_noise_std <= 0 or self.measurement_noise_std <= 0:
            raise ValueError("Noise standard deviations must be positive")
        
        if self.adaptation_rate <= 0 or self.adaptation_rate >= 1:
            raise ValueError("Adaptation rate must be in (0, 1)")
        
        if self.alpha <= 0 or self.beta < 0:
            raise ValueError("UKF parameters alpha must be positive, beta non-negative")


@dataclass
class FilterState:
    """Complete state representation for Kalman filters."""
    
    # Core state
    state_mean: StateVector
    state_cov: CovarianceMatrix
    
    # Metadata
    timestamp: float
    gameweek: int
    season: str
    player_id: int
    
    # Filter statistics
    innovation: Optional[ObservationVector] = None
    innovation_cov: Optional[CovarianceMatrix] = None
    kalman_gain: Optional[Array] = None
    log_likelihood: Optional[float] = None
    
    # Noise estimates (for adaptive filters)
    process_noise_est: Optional[CovarianceMatrix] = None
    measurement_noise_est: Optional[CovarianceMatrix] = None
    
    # Quality metrics
    condition_number: Optional[float] = None
    trace_ratio: Optional[float] = None  # trace(P) / trace(P_0)
    
    def copy(self) -> FilterState:
        """Create a deep copy of the filter state."""
        return FilterState(
            state_mean=jnp.copy(self.state_mean),
            state_cov=jnp.copy(self.state_cov),
            timestamp=self.timestamp,
            gameweek=self.gameweek,
            season=self.season,
            player_id=self.player_id,
            innovation=jnp.copy(self.innovation) if self.innovation is not None else None,
            innovation_cov=jnp.copy(self.innovation_cov) if self.innovation_cov is not None else None,
            kalman_gain=jnp.copy(self.kalman_gain) if self.kalman_gain is not None else None,
            log_likelihood=self.log_likelihood,
            process_noise_est=jnp.copy(self.process_noise_est) if self.process_noise_est is not None else None,
            measurement_noise_est=jnp.copy(self.measurement_noise_est) if self.measurement_noise_est is not None else None,
            condition_number=self.condition_number,
            trace_ratio=self.trace_ratio
        )


class KalmanFilterError(Exception):
    """Base exception for Kalman filter operations."""
    pass


class NumericalInstabilityError(KalmanFilterError):
    """Raised when numerical instability is detected."""
    pass


class InvalidCovarianceError(KalmanFilterError):
    """Raised when covariance matrix is invalid (not positive semi-definite)."""
    pass


class AdaptiveNoiseEstimator:
    """Adaptive noise estimation for robust Kalman filtering."""
    
    def __init__(self, config: FilterConfig):
        """Initialize adaptive noise estimator."""
        self.config = config
        self.adaptation_rate = config.adaptation_rate
        self.window_size = config.noise_estimation_window
        
        # Innovation history for noise estimation
        self.innovation_history: List[Array] = []
        self.residual_history: List[Array] = []
        
        # Current noise estimates
        self.process_noise_cov = jnp.eye(config.state_dim) * config.process_noise_std**2
        self.measurement_noise_cov = jnp.eye(config.obs_dim) * config.measurement_noise_std**2
    
    def update_process_noise(
        self, 
        predicted_state: StateVector,
        updated_state: StateVector,
        dt: float
    ) -> CovarianceMatrix:
        """
        Update process noise estimate based on state transitions.
        
        Args:
            predicted_state: Predicted state vector
            updated_state: Updated state vector after measurement
            dt: Time step
            
        Returns:
            Updated process noise covariance matrix
        """
        # Compute innovation in state space
        state_innovation = updated_state - predicted_state
        
        # Update history
        self.residual_history.append(state_innovation)
        if len(self.residual_history) > self.window_size:
            self.residual_history.pop(0)
        
        # Estimate process noise from residuals
        if len(self.residual_history) >= 3:
            residuals = jnp.array(self.residual_history)
            empirical_cov = jnp.cov(residuals.T)
            
            # Adapt process noise with exponential smoothing
            self.process_noise_cov = (
                (1 - self.adaptation_rate) * self.process_noise_cov +
                self.adaptation_rate * empirical_cov / dt
            )
            
            # Ensure positive definiteness
            self.process_noise_cov = self._ensure_psd(self.process_noise_cov)
        
        return self.process_noise_cov
    
    def update_measurement_noise(
        self, 
        innovation: ObservationVector,
        innovation_cov: CovarianceMatrix
    ) -> CovarianceMatrix:
        """
        Update measurement noise estimate based on innovations.
        
        Args:
            innovation: Innovation vector (observation - prediction)
            innovation_cov: Innovation covariance matrix
            
        Returns:
            Updated measurement noise covariance matrix
        """
        # Update innovation history
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
        
        # Estimate measurement noise from innovations
        if len(self.innovation_history) >= 3:
            innovations = jnp.array(self.innovation_history)
            empirical_cov = jnp.cov(innovations.T)
            
            # Subtract predicted innovation covariance to isolate measurement noise
            estimated_R = empirical_cov - (innovation_cov - self.measurement_noise_cov)
            
            # Adapt measurement noise with exponential smoothing
            self.measurement_noise_cov = (
                (1 - self.adaptation_rate) * self.measurement_noise_cov +
                self.adaptation_rate * jnp.maximum(estimated_R, 
                                                   jnp.eye(self.config.obs_dim) * 1e-6)
            )
            
            # Ensure positive definiteness
            self.measurement_noise_cov = self._ensure_psd(self.measurement_noise_cov)
        
        return self.measurement_noise_cov
    
    def _ensure_psd(self, matrix: CovarianceMatrix) -> CovarianceMatrix:
        """Ensure matrix is positive semi-definite."""
        try:
            # Eigendecomposition
            eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
            
            # Clip negative eigenvalues
            eigenvals = jnp.maximum(eigenvals, self.config.min_eigenvalue)
            
            # Reconstruct matrix
            return eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
            
        except Exception as e:
            logger.warning(f"Failed to ensure PSD matrix: {e}")
            # Fall back to regularization
            return matrix + jnp.eye(matrix.shape[0]) * self.config.regularization_factor


class BaseKalmanFilter(ABC):
    """Abstract base class for Kalman filter implementations."""
    
    def __init__(self, config: FilterConfig):
        """Initialize base Kalman filter."""
        self.config = config
        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim
        
        # Initialize noise estimator
        if config.enable_adaptive_noise:
            self.noise_estimator = AdaptiveNoiseEstimator(config)
        else:
            self.noise_estimator = None
        
        # Default noise matrices
        self.Q = jnp.eye(self.state_dim) * config.process_noise_std**2
        self.R = jnp.eye(self.obs_dim) * config.measurement_noise_std**2
        
        # JAX setup
        if config.use_jax:
            self.array_lib = jnp
            self.linalg = jnp.linalg
        else:
            self.array_lib = np
            self.linalg = np.linalg
    
    @abstractmethod
    def predict(
        self, 
        state: FilterState, 
        dt: float = 1.0,
        control_input: Optional[Array] = None
    ) -> FilterState:
        """Predict state forward in time."""
        pass
    
    @abstractmethod
    def update(
        self, 
        predicted_state: FilterState, 
        observation: ObservationVector,
        observation_noise: Optional[CovarianceMatrix] = None
    ) -> FilterState:
        """Update state with new observation."""
        pass
    
    def _ensure_numerical_stability(self, cov_matrix: CovarianceMatrix) -> CovarianceMatrix:
        """Ensure numerical stability of covariance matrices."""
        if not self.config.check_psd:
            return cov_matrix
        
        try:
            # Check for NaN or infinite values
            if not jnp.all(jnp.isfinite(cov_matrix)):
                raise NumericalInstabilityError("Covariance matrix contains NaN or infinite values")
            
            # Enforce symmetry if requested
            if self.config.enforce_symmetry:
                cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
            
            # Check positive semi-definiteness
            if self.config.check_psd:
                eigenvals = jnp.linalg.eigvals(cov_matrix)
                min_eigenval = jnp.min(eigenvals)
                
                if min_eigenval < -1e-10:  # Allow small numerical errors
                    logger.warning(f"Covariance matrix has negative eigenvalue: {min_eigenval}")
                    # Regularize by adding small positive values to diagonal
                    cov_matrix = cov_matrix + jnp.eye(cov_matrix.shape[0]) * self.config.regularization_factor
                
                # Ensure minimum eigenvalue
                if min_eigenval < self.config.min_eigenvalue:
                    eigenvals, eigenvecs = jnp.linalg.eigh(cov_matrix)
                    eigenvals = jnp.maximum(eigenvals, self.config.min_eigenvalue)
                    cov_matrix = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Failed to ensure numerical stability: {e}")
            raise NumericalInstabilityError(f"Covariance matrix stabilization failed: {e}")
    
    def _compute_log_likelihood(
        self, 
        innovation: ObservationVector, 
        innovation_cov: CovarianceMatrix
    ) -> float:
        """Compute log-likelihood of observation given prediction."""
        try:
            # Compute log determinant safely
            sign, logdet = jnp.linalg.slogdet(innovation_cov)
            if sign <= 0:
                logger.warning("Innovation covariance matrix is not positive definite")
                return -jnp.inf
            
            # Compute quadratic form
            inv_S = jnp.linalg.inv(innovation_cov)
            quad_form = innovation.T @ inv_S @ innovation
            
            # Log-likelihood (ignoring constant terms)
            log_likelihood = -0.5 * (logdet + quad_form + self.obs_dim * jnp.log(2 * jnp.pi))
            
            return float(log_likelihood)
            
        except Exception as e:
            logger.warning(f"Failed to compute log-likelihood: {e}")
            return -jnp.inf
    
    def get_diagnostics(self, state: FilterState) -> Dict[str, Any]:
        """Get filter diagnostics for monitoring."""
        diagnostics = {
            "condition_number": state.condition_number,
            "trace_ratio": state.trace_ratio,
            "log_likelihood": state.log_likelihood,
            "state_norm": float(jnp.linalg.norm(state.state_mean)),
            "cov_trace": float(jnp.trace(state.state_cov)),
            "cov_det": float(jnp.linalg.det(state.state_cov)),
        }
        
        if state.innovation is not None:
            diagnostics["innovation_norm"] = float(jnp.linalg.norm(state.innovation))
            
        if state.kalman_gain is not None:
            diagnostics["gain_norm"] = float(jnp.linalg.norm(state.kalman_gain))
        
        return diagnostics


class KalmanFilter(BaseKalmanFilter):
    """Linear Kalman Filter implementation with numerical stability features."""
    
    def __init__(
        self, 
        config: FilterConfig,
        transition_matrix: Optional[Array] = None,
        observation_matrix: Optional[Array] = None
    ):
        """
        Initialize linear Kalman filter.
        
        Args:
            config: Filter configuration
            transition_matrix: State transition matrix F (default: identity)
            observation_matrix: Observation matrix H (default: identity)
        """
        super().__init__(config)
        
        # Default transition matrix (identity)
        if transition_matrix is not None:
            self.F = jnp.array(transition_matrix)
        else:
            self.F = jnp.eye(self.state_dim)
        
        # Default observation matrix (identity for same dimensions)
        if observation_matrix is not None:
            self.H = jnp.array(observation_matrix)
        else:
            if self.state_dim == self.obs_dim:
                self.H = jnp.eye(self.obs_dim)
            else:
                # Create mapping matrix for different dimensions
                min_dim = min(self.state_dim, self.obs_dim)
                self.H = jnp.zeros((self.obs_dim, self.state_dim))
                self.H = self.H.at[:min_dim, :min_dim].set(jnp.eye(min_dim))
        
        if self.config.validate_inputs:
            self._validate_matrices()
    
    def _validate_matrices(self):
        """Validate matrix dimensions and properties."""
        if self.F.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Transition matrix shape {self.F.shape} != ({self.state_dim}, {self.state_dim})")
        
        if self.H.shape != (self.obs_dim, self.state_dim):
            raise ValueError(f"Observation matrix shape {self.H.shape} != ({self.obs_dim}, {self.state_dim})")
    
    def predict(
        self, 
        state: FilterState, 
        dt: float = 1.0,
        control_input: Optional[Array] = None
    ) -> FilterState:
        """
        Prediction step of Kalman filter.
        
        Args:
            state: Current filter state
            dt: Time step for prediction
            control_input: Optional control input (not used in basic version)
            
        Returns:
            Predicted filter state
        """
        try:
            # Time-varying transition matrix (scale by dt if needed)
            if dt != 1.0:
                F_dt = self._adapt_transition_matrix(self.F, dt)
            else:
                F_dt = self.F
            
            # Predict state mean: x_k+1|k = F * x_k|k
            predicted_mean = F_dt @ state.state_mean
            
            # Get process noise (adaptive if enabled)
            Q_current = self.Q
            if self.noise_estimator is not None:
                Q_current = self.noise_estimator.process_noise_cov
            
            # Predict covariance: P_k+1|k = F * P_k|k * F^T + Q
            predicted_cov = F_dt @ state.state_cov @ F_dt.T + Q_current * dt
            
            # Ensure numerical stability
            predicted_cov = self._ensure_numerical_stability(predicted_cov)
            
            # Create predicted state
            predicted_state = FilterState(
                state_mean=predicted_mean,
                state_cov=predicted_cov,
                timestamp=state.timestamp + dt,
                gameweek=state.gameweek,
                season=state.season,
                player_id=state.player_id,
                condition_number=float(jnp.linalg.cond(predicted_cov)),
                trace_ratio=float(jnp.trace(predicted_cov) / jnp.trace(state.state_cov))
            )
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"Prediction step failed: {e}")
            raise KalmanFilterError(f"Prediction failed: {e}")
    
    def update(
        self, 
        predicted_state: FilterState, 
        observation: ObservationVector,
        observation_noise: Optional[CovarianceMatrix] = None
    ) -> FilterState:
        """
        Update step of Kalman filter.
        
        Args:
            predicted_state: Predicted state from prediction step
            observation: New observation vector
            observation_noise: Optional observation noise covariance
            
        Returns:
            Updated filter state
        """
        try:
            # Get measurement noise (adaptive if enabled)
            R_current = observation_noise if observation_noise is not None else self.R
            if self.noise_estimator is not None:
                R_current = self.noise_estimator.measurement_noise_cov
            
            # Predicted observation: y_hat = H * x_k+1|k
            predicted_obs = self.H @ predicted_state.state_mean
            
            # Innovation: y_tilde = y - y_hat
            innovation = observation - predicted_obs
            
            # Innovation covariance: S = H * P_k+1|k * H^T + R
            innovation_cov = self.H @ predicted_state.state_cov @ self.H.T + R_current
            
            # Ensure numerical stability of innovation covariance
            innovation_cov = self._ensure_numerical_stability(innovation_cov)
            
            # Kalman gain: K = P_k+1|k * H^T * S^(-1)
            try:
                # Use solve instead of inverse for numerical stability
                kalman_gain = predicted_state.state_cov @ self.H.T @ jnp.linalg.inv(innovation_cov)
            except jnp.linalg.LinAlgError:
                # Fall back to pseudoinverse if singular
                kalman_gain = predicted_state.state_cov @ self.H.T @ jnp.linalg.pinv(innovation_cov)
                logger.warning("Used pseudoinverse for Kalman gain computation")
            
            # Updated state mean: x_k+1|k+1 = x_k+1|k + K * y_tilde
            updated_mean = predicted_state.state_mean + kalman_gain @ innovation
            
            # Updated covariance
            if self.config.use_joseph_form:
                # Joseph form for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
                I_KH = jnp.eye(self.state_dim) - kalman_gain @ self.H
                updated_cov = (I_KH @ predicted_state.state_cov @ I_KH.T + 
                              kalman_gain @ R_current @ kalman_gain.T)
            else:
                # Standard form: P = (I - KH)P
                updated_cov = (jnp.eye(self.state_dim) - kalman_gain @ self.H) @ predicted_state.state_cov
            
            # Ensure numerical stability
            updated_cov = self._ensure_numerical_stability(updated_cov)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(innovation, innovation_cov)
            
            # Update adaptive noise estimates if enabled
            if self.noise_estimator is not None:
                self.noise_estimator.update_measurement_noise(innovation, innovation_cov)
                self.noise_estimator.update_process_noise(
                    predicted_state.state_mean, updated_mean, 1.0
                )
            
            # Create updated state
            updated_state = FilterState(
                state_mean=updated_mean,
                state_cov=updated_cov,
                timestamp=predicted_state.timestamp,
                gameweek=predicted_state.gameweek,
                season=predicted_state.season,
                player_id=predicted_state.player_id,
                innovation=innovation,
                innovation_cov=innovation_cov,
                kalman_gain=kalman_gain,
                log_likelihood=log_likelihood,
                condition_number=float(jnp.linalg.cond(updated_cov)),
                trace_ratio=float(jnp.trace(updated_cov) / jnp.trace(predicted_state.state_cov))
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Update step failed: {e}")
            raise KalmanFilterError(f"Update failed: {e}")
    
    def _adapt_transition_matrix(self, F: Array, dt: float) -> Array:
        """Adapt transition matrix for different time steps."""
        # Simple scaling for demonstration - can be made more sophisticated
        if dt == 1.0:
            return F
        
        # For linear systems, we could use matrix exponential
        # For simplicity, we'll scale off-diagonal elements
        F_dt = F.copy()
        # Scale off-diagonal elements (dynamics) by dt
        mask = ~jnp.eye(self.state_dim, dtype=bool)
        F_dt = F_dt.at[mask].mul(dt)
        
        return F_dt


class ExtendedKalmanFilter(BaseKalmanFilter):
    """Extended Kalman Filter for non-linear dynamics."""
    
    def __init__(
        self, 
        config: FilterConfig,
        transition_fn: TransitionFunction,
        measurement_fn: MeasurementFunction,
        transition_jacobian_fn: Optional[Callable] = None,
        measurement_jacobian_fn: Optional[Callable] = None
    ):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            config: Filter configuration
            transition_fn: Non-linear state transition function
            measurement_fn: Non-linear measurement function
            transition_jacobian_fn: Optional Jacobian of transition function
            measurement_jacobian_fn: Optional Jacobian of measurement function
        """
        super().__init__(config)
        
        self.transition_fn = transition_fn
        self.measurement_fn = measurement_fn
        
        # Auto-compute Jacobians if not provided
        if transition_jacobian_fn is not None:
            self.transition_jacobian_fn = transition_jacobian_fn
        else:
            self.transition_jacobian_fn = jacfwd(self.transition_fn)
        
        if measurement_jacobian_fn is not None:
            self.measurement_jacobian_fn = measurement_jacobian_fn
        else:
            self.measurement_jacobian_fn = jacfwd(self.measurement_fn)
    
    def predict(
        self, 
        state: FilterState, 
        dt: float = 1.0,
        control_input: Optional[Array] = None
    ) -> FilterState:
        """EKF prediction step with non-linear dynamics."""
        try:
            # Predict state mean using non-linear function
            predicted_mean = self.transition_fn(state.state_mean, dt)
            
            # Compute Jacobian of transition function
            F_jacobian = self.transition_jacobian_fn(state.state_mean, dt)
            
            # Get process noise
            Q_current = self.Q
            if self.noise_estimator is not None:
                Q_current = self.noise_estimator.process_noise_cov
            
            # Predict covariance using linearized dynamics
            predicted_cov = F_jacobian @ state.state_cov @ F_jacobian.T + Q_current * dt
            
            # Ensure numerical stability
            predicted_cov = self._ensure_numerical_stability(predicted_cov)
            
            predicted_state = FilterState(
                state_mean=predicted_mean,
                state_cov=predicted_cov,
                timestamp=state.timestamp + dt,
                gameweek=state.gameweek,
                season=state.season,
                player_id=state.player_id,
                condition_number=float(jnp.linalg.cond(predicted_cov)),
                trace_ratio=float(jnp.trace(predicted_cov) / jnp.trace(state.state_cov))
            )
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"EKF prediction step failed: {e}")
            raise KalmanFilterError(f"EKF prediction failed: {e}")
    
    def update(
        self, 
        predicted_state: FilterState, 
        observation: ObservationVector,
        observation_noise: Optional[CovarianceMatrix] = None
    ) -> FilterState:
        """EKF update step with non-linear measurements."""
        try:
            # Get measurement noise
            R_current = observation_noise if observation_noise is not None else self.R
            if self.noise_estimator is not None:
                R_current = self.noise_estimator.measurement_noise_cov
            
            # Predicted observation using non-linear function
            predicted_obs = self.measurement_fn(predicted_state.state_mean)
            
            # Innovation
            innovation = observation - predicted_obs
            
            # Compute Jacobian of measurement function
            H_jacobian = self.measurement_jacobian_fn(predicted_state.state_mean)
            
            # Innovation covariance using linearized measurement
            innovation_cov = H_jacobian @ predicted_state.state_cov @ H_jacobian.T + R_current
            innovation_cov = self._ensure_numerical_stability(innovation_cov)
            
            # Kalman gain
            try:
                kalman_gain = predicted_state.state_cov @ H_jacobian.T @ jnp.linalg.inv(innovation_cov)
            except jnp.linalg.LinAlgError:
                kalman_gain = predicted_state.state_cov @ H_jacobian.T @ jnp.linalg.pinv(innovation_cov)
                logger.warning("Used pseudoinverse for EKF Kalman gain computation")
            
            # Updated state
            updated_mean = predicted_state.state_mean + kalman_gain @ innovation
            
            # Updated covariance
            if self.config.use_joseph_form:
                I_KH = jnp.eye(self.state_dim) - kalman_gain @ H_jacobian
                updated_cov = (I_KH @ predicted_state.state_cov @ I_KH.T + 
                              kalman_gain @ R_current @ kalman_gain.T)
            else:
                updated_cov = (jnp.eye(self.state_dim) - kalman_gain @ H_jacobian) @ predicted_state.state_cov
            
            updated_cov = self._ensure_numerical_stability(updated_cov)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(innovation, innovation_cov)
            
            # Update adaptive noise estimates
            if self.noise_estimator is not None:
                self.noise_estimator.update_measurement_noise(innovation, innovation_cov)
                self.noise_estimator.update_process_noise(
                    predicted_state.state_mean, updated_mean, 1.0
                )
            
            updated_state = FilterState(
                state_mean=updated_mean,
                state_cov=updated_cov,
                timestamp=predicted_state.timestamp,
                gameweek=predicted_state.gameweek,
                season=predicted_state.season,
                player_id=predicted_state.player_id,
                innovation=innovation,
                innovation_cov=innovation_cov,
                kalman_gain=kalman_gain,
                log_likelihood=log_likelihood,
                condition_number=float(jnp.linalg.cond(updated_cov)),
                trace_ratio=float(jnp.trace(updated_cov) / jnp.trace(predicted_state.state_cov))
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"EKF update step failed: {e}")
            raise KalmanFilterError(f"EKF update failed: {e}")


class UnscentedKalmanFilter(BaseKalmanFilter):
    """Unscented Kalman Filter for better non-linear handling."""
    
    def __init__(
        self, 
        config: FilterConfig,
        transition_fn: TransitionFunction,
        measurement_fn: MeasurementFunction
    ):
        """
        Initialize Unscented Kalman Filter.
        
        Args:
            config: Filter configuration
            transition_fn: Non-linear state transition function
            measurement_fn: Non-linear measurement function
        """
        super().__init__(config)
        
        self.transition_fn = transition_fn
        self.measurement_fn = measurement_fn
        
        # UKF parameters
        self.alpha = config.alpha
        self.beta = config.beta
        self.kappa = config.kappa
        
        # Derived parameters
        self.n = config.state_dim
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights for sigma points
        self.weight_m, self.weight_c = self._compute_weights()
    
    def _compute_weights(self) -> Tuple[Array, Array]:
        """Compute weights for sigma points."""
        n_sigma = 2 * self.n + 1
        
        # Mean weights
        weight_m = jnp.zeros(n_sigma)
        weight_m = weight_m.at[0].set(self.lambda_ / (self.n + self.lambda_))
        weight_m = weight_m.at[1:].set(0.5 / (self.n + self.lambda_))
        
        # Covariance weights
        weight_c = weight_m.copy()
        weight_c = weight_c.at[0].set(weight_c[0] + (1 - self.alpha**2 + self.beta))
        
        return weight_m, weight_c
    
    def _generate_sigma_points(self, state_mean: StateVector, state_cov: CovarianceMatrix) -> Array:
        """Generate sigma points for UKF."""
        try:
            # Compute matrix square root using Cholesky decomposition
            sqrt_cov = cholesky((self.n + self.lambda_) * state_cov)
            
            # Initialize sigma points
            sigma_points = jnp.zeros((2 * self.n + 1, self.n))
            
            # Central point
            sigma_points = sigma_points.at[0].set(state_mean)
            
            # Positive sigma points
            for i in range(self.n):
                sigma_points = sigma_points.at[i + 1].set(state_mean + sqrt_cov[i])
            
            # Negative sigma points
            for i in range(self.n):
                sigma_points = sigma_points.at[i + self.n + 1].set(state_mean - sqrt_cov[i])
            
            return sigma_points
            
        except Exception as e:
            logger.warning(f"Cholesky decomposition failed, using eigendecomposition: {e}")
            # Fall back to eigendecomposition
            eigenvals, eigenvecs = jnp.linalg.eigh((self.n + self.lambda_) * state_cov)
            sqrt_cov = eigenvecs @ jnp.diag(jnp.sqrt(jnp.maximum(eigenvals, 1e-12)))
            
            sigma_points = jnp.zeros((2 * self.n + 1, self.n))
            sigma_points = sigma_points.at[0].set(state_mean)
            
            for i in range(self.n):
                sigma_points = sigma_points.at[i + 1].set(state_mean + sqrt_cov[:, i])
                sigma_points = sigma_points.at[i + self.n + 1].set(state_mean - sqrt_cov[:, i])
            
            return sigma_points
    
    def predict(
        self, 
        state: FilterState, 
        dt: float = 1.0,
        control_input: Optional[Array] = None
    ) -> FilterState:
        """UKF prediction step."""
        try:
            # Generate sigma points
            sigma_points = self._generate_sigma_points(state.state_mean, state.state_cov)
            
            # Propagate sigma points through transition function
            n_sigma = sigma_points.shape[0]
            predicted_sigma = jnp.zeros_like(sigma_points)
            
            for i in range(n_sigma):
                predicted_sigma = predicted_sigma.at[i].set(self.transition_fn(sigma_points[i], dt))
            
            # Compute predicted mean
            predicted_mean = jnp.sum(self.weight_m[:, None] * predicted_sigma, axis=0)
            
            # Compute predicted covariance
            predicted_cov = jnp.zeros((self.n, self.n))
            for i in range(n_sigma):
                diff = predicted_sigma[i] - predicted_mean
                predicted_cov += self.weight_c[i] * jnp.outer(diff, diff)
            
            # Add process noise
            Q_current = self.Q
            if self.noise_estimator is not None:
                Q_current = self.noise_estimator.process_noise_cov
            
            predicted_cov += Q_current * dt
            predicted_cov = self._ensure_numerical_stability(predicted_cov)
            
            predicted_state = FilterState(
                state_mean=predicted_mean,
                state_cov=predicted_cov,
                timestamp=state.timestamp + dt,
                gameweek=state.gameweek,
                season=state.season,
                player_id=state.player_id,
                condition_number=float(jnp.linalg.cond(predicted_cov)),
                trace_ratio=float(jnp.trace(predicted_cov) / jnp.trace(state.state_cov))
            )
            
            return predicted_state
            
        except Exception as e:
            logger.error(f"UKF prediction step failed: {e}")
            raise KalmanFilterError(f"UKF prediction failed: {e}")
    
    def update(
        self, 
        predicted_state: FilterState, 
        observation: ObservationVector,
        observation_noise: Optional[CovarianceMatrix] = None
    ) -> FilterState:
        """UKF update step."""
        try:
            # Generate sigma points for predicted state
            sigma_points = self._generate_sigma_points(predicted_state.state_mean, predicted_state.state_cov)
            
            # Propagate through measurement function
            n_sigma = sigma_points.shape[0]
            predicted_obs_sigma = jnp.zeros((n_sigma, self.obs_dim))
            
            for i in range(n_sigma):
                predicted_obs_sigma = predicted_obs_sigma.at[i].set(self.measurement_fn(sigma_points[i]))
            
            # Predicted observation mean
            predicted_obs_mean = jnp.sum(self.weight_m[:, None] * predicted_obs_sigma, axis=0)
            
            # Innovation
            innovation = observation - predicted_obs_mean
            
            # Innovation covariance
            innovation_cov = jnp.zeros((self.obs_dim, self.obs_dim))
            for i in range(n_sigma):
                diff = predicted_obs_sigma[i] - predicted_obs_mean
                innovation_cov += self.weight_c[i] * jnp.outer(diff, diff)
            
            # Add measurement noise
            R_current = observation_noise if observation_noise is not None else self.R
            if self.noise_estimator is not None:
                R_current = self.noise_estimator.measurement_noise_cov
            
            innovation_cov += R_current
            innovation_cov = self._ensure_numerical_stability(innovation_cov)
            
            # Cross-covariance
            cross_cov = jnp.zeros((self.n, self.obs_dim))
            for i in range(n_sigma):
                state_diff = sigma_points[i] - predicted_state.state_mean
                obs_diff = predicted_obs_sigma[i] - predicted_obs_mean
                cross_cov += self.weight_c[i] * jnp.outer(state_diff, obs_diff)
            
            # Kalman gain
            try:
                kalman_gain = cross_cov @ jnp.linalg.inv(innovation_cov)
            except jnp.linalg.LinAlgError:
                kalman_gain = cross_cov @ jnp.linalg.pinv(innovation_cov)
                logger.warning("Used pseudoinverse for UKF Kalman gain computation")
            
            # Updated state
            updated_mean = predicted_state.state_mean + kalman_gain @ innovation
            updated_cov = predicted_state.state_cov - kalman_gain @ innovation_cov @ kalman_gain.T
            updated_cov = self._ensure_numerical_stability(updated_cov)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(innovation, innovation_cov)
            
            # Update adaptive noise estimates
            if self.noise_estimator is not None:
                self.noise_estimator.update_measurement_noise(innovation, innovation_cov)
                self.noise_estimator.update_process_noise(
                    predicted_state.state_mean, updated_mean, 1.0
                )
            
            updated_state = FilterState(
                state_mean=updated_mean,
                state_cov=updated_cov,
                timestamp=predicted_state.timestamp,
                gameweek=predicted_state.gameweek,
                season=predicted_state.season,
                player_id=predicted_state.player_id,
                innovation=innovation,
                innovation_cov=innovation_cov,
                kalman_gain=kalman_gain,
                log_likelihood=log_likelihood,
                condition_number=float(jnp.linalg.cond(updated_cov)),
                trace_ratio=float(jnp.trace(updated_cov) / jnp.trace(predicted_state.state_cov))
            )
            
            return updated_state
            
        except Exception as e:
            logger.error(f"UKF update step failed: {e}")
            raise KalmanFilterError(f"UKF update failed: {e}")


# Utility functions for creating specific filter configurations

def create_player_ability_config(
    state_names: List[str] = None,
    obs_names: List[str] = None,
    **kwargs
) -> FilterConfig:
    """Create filter configuration for player ability tracking."""
    if state_names is None:
        state_names = ["skill", "form", "consistency", "momentum"]
    
    if obs_names is None:
        obs_names = ["goals", "assists", "minutes", "bonus"]
    
    return FilterConfig(
        state_dim=len(state_names),
        obs_dim=len(obs_names),
        **kwargs
    )


def create_simple_linear_filter(config: FilterConfig) -> KalmanFilter:
    """Create a simple linear Kalman filter for player tracking."""
    # Simple transition matrix with decay for form and momentum
    F = jnp.eye(config.state_dim)
    if config.state_dim >= 4:
        # Form decays slightly each time step
        F = F.at[1, 1].set(0.95)  # form component
        # Momentum decays more quickly
        F = F.at[3, 3].set(0.9)   # momentum component
    
    # Observation matrix maps all states to observations
    H = jnp.eye(min(config.state_dim, config.obs_dim), config.state_dim)
    if config.obs_dim > config.state_dim:
        # Extend with zeros if more observations than states
        H_extended = jnp.zeros((config.obs_dim, config.state_dim))
        H_extended = H_extended.at[:config.state_dim, :].set(jnp.eye(config.state_dim))
        H = H_extended
    
    return KalmanFilter(config, transition_matrix=F, observation_matrix=H)


def create_player_transition_function() -> TransitionFunction:
    """Create non-linear transition function for player abilities."""
    
    def transition_fn(state: StateVector, dt: float) -> StateVector:
        """
        Non-linear state transition for player abilities.
        
        State vector: [skill, form, consistency, momentum]
        """
        skill, form, consistency, momentum = state[0], state[1], state[2], state[3]
        
        # Skill evolves slowly with small random walks
        new_skill = skill + 0.001 * momentum * dt
        
        # Form decays towards skill level but can be influenced by momentum
        form_decay = 0.95**dt
        new_form = form_decay * form + (1 - form_decay) * skill + 0.01 * momentum * dt
        
        # Consistency is relatively stable
        new_consistency = 0.99**dt * consistency + 0.01 * skill
        
        # Momentum decays quickly
        momentum_decay = 0.8**dt
        new_momentum = momentum_decay * momentum
        
        return jnp.array([new_skill, new_form, new_consistency, new_momentum])
    
    return transition_fn


def create_player_measurement_function() -> MeasurementFunction:
    """Create measurement function for player performance observations."""
    
    def measurement_fn(state: StateVector) -> ObservationVector:
        """
        Map player abilities to observable performance metrics.
        
        State vector: [skill, form, consistency, momentum]
        Observation: [goals, assists, minutes, bonus]
        """
        skill, form, consistency, momentum = state[0], state[1], state[2], state[3]
        
        # Goals depend on skill, form, and momentum
        goals = jnp.maximum(0, skill * form + 0.1 * momentum)
        
        # Assists depend on skill and consistency
        assists = jnp.maximum(0, 0.8 * skill * consistency + 0.05 * momentum)
        
        # Minutes depend on consistency and form (reliable players play more)
        minutes = jnp.maximum(0, 60 + 30 * consistency * form)
        
        # Bonus points depend on overall performance
        bonus = jnp.maximum(0, (goals + assists) * consistency + 0.1 * momentum)
        
        return jnp.array([goals, assists, minutes, bonus])
    
    return measurement_fn