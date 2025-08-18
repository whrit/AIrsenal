"""
Adaptive Learning Rate System for AIrsenal Player Models

This module implements an adaptive learning rate system that adjusts model update speeds
based on prediction errors and player volatility. It provides sophisticated algorithms
for optimizing learning rates in real-time to improve model convergence while maintaining
stability.

ARCHITECTURE OVERVIEW
====================

The system consists of several interconnected components:

1. **AdaptiveLearningRateSystem**: Main controller orchestrating all components
2. **ErrorBasedAdapter**: Adjusts rates based on prediction errors and performance
3. **VolatilityDetector**: Identifies volatile players requiring different learning approaches
4. **ConvergenceMonitor**: Tracks model convergence and detects instability
5. **LearningScheduler**: Implements time-based learning rate schedules
6. **Adaptation Algorithms**: AdaGrad, RMSProp, Adam, AdaDelta implementations

KEY FEATURES
============

- Multiple adaptation algorithms with automatic selection
- Error-based rate adjustment with position-specific thresholds  
- Player volatility detection and regime change identification
- Comprehensive convergence monitoring with early stopping
- Integration with Kalman filter systems
- Risk management with bounds and safety checks
- Gradient-based optimization with numerical stability

USAGE EXAMPLE
=============

```python
from airsenal.framework.adaptive_learning import AdaptiveLearningRateSystem

# Initialize system with configuration
system = AdaptiveLearningRateSystem(
    base_learning_rate=0.01,
    adaptation_method='adam',
    volatility_threshold=0.3,
    max_learning_rate=0.1,
    min_learning_rate=1e-6
)

# Adapt learning rate based on player performance
player_id = 123
prediction_error = 0.25
actual_points = 5.0
predicted_points = 4.0

new_rate = system.adapt_learning_rate(
    player_id=player_id,
    prediction_error=prediction_error,
    actual_value=actual_points,
    predicted_value=predicted_points,
    gameweek=15
)

# Monitor convergence
is_converged = system.check_convergence(player_id, loss_history=[0.5, 0.4, 0.35, 0.33])
```

INTEGRATION WITH KALMAN FILTERS
===============================

The adaptive learning system integrates seamlessly with Kalman filters by:
- Adjusting Kalman gain based on learning rate
- Modifying process noise dynamically
- Controlling measurement trust based on volatility
- Providing stable state updates with convergence monitoring

Classes:
    AdaptiveLearningRateSystem: Main learning rate controller
    ErrorBasedAdapter: Prediction error-based rate adjustment
    VolatilityDetector: Player volatility and regime change detection
    ConvergenceMonitor: Model convergence tracking and early stopping
    LearningScheduler: Time-based learning rate schedules
    AdaptationAlgorithm: Base class for adaptation algorithms
    AdaGradOptimizer: AdaGrad implementation
    RMSPropOptimizer: RMSProp implementation  
    AdamOptimizer: Adam optimization algorithm
    AdaDeltaOptimizer: AdaDelta implementation
    CustomFPLOptimizer: FPL-specific adaptation algorithm
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import grad, jit
from jax.scipy.stats import norm

logger = logging.getLogger(__name__)

# Type aliases
Array = Union[np.ndarray, jnp.ndarray]
PlayerId = int
GameweekId = int
LearningRate = float
PredictionError = float


class AdaptationMethod(Enum):
    """Available adaptation algorithms."""
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADADELTA = "adadelta"
    CUSTOM_FPL = "custom_fpl"


class PlayerPosition(Enum):
    """Player positions for position-specific adaptations."""
    GOALKEEPER = "GK"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"


@dataclass
class AdaptiveLearningConfig:
    """Configuration for adaptive learning rate system."""
    
    # Base parameters
    base_learning_rate: float = 0.01
    max_learning_rate: float = 0.1
    min_learning_rate: float = 1e-6
    
    # Adaptation method
    adaptation_method: AdaptationMethod = AdaptationMethod.ADAM
    
    # Error-based adaptation
    error_sensitivity: float = 1.0
    asymmetric_adjustment: bool = True
    over_prediction_factor: float = 0.8  # Reduce rate more for over-predictions
    under_prediction_factor: float = 1.2  # Increase rate more for under-predictions
    
    # Position-specific thresholds
    position_error_thresholds: Dict[PlayerPosition, float] = field(default_factory=lambda: {
        PlayerPosition.GOALKEEPER: 0.5,
        PlayerPosition.DEFENDER: 0.8,
        PlayerPosition.MIDFIELDER: 1.0,
        PlayerPosition.FORWARD: 1.2
    })
    
    # Volatility detection
    volatility_threshold: float = 0.3
    volatility_window: int = 10
    regime_change_threshold: float = 2.0
    
    # Convergence monitoring
    convergence_patience: int = 5
    convergence_threshold: float = 1e-4
    min_improvement: float = 1e-5
    oscillation_threshold: float = 0.1
    
    # Learning schedules
    schedule_type: str = "exponential"  # exponential, step, cosine, warm_restart
    decay_rate: float = 0.95
    decay_steps: int = 10
    warmup_steps: int = 5
    
    # Risk management
    gradient_clip_norm: float = 1.0
    enable_gradient_clipping: bool = True
    rollback_on_divergence: bool = True
    numerical_stability_eps: float = 1e-8
    
    # Integration settings
    kalman_integration: bool = True
    update_process_noise: bool = True
    trust_adjustment_factor: float = 0.1


@dataclass
class PlayerLearningState:
    """State information for a single player's learning rate adaptation."""
    
    player_id: PlayerId
    position: PlayerPosition
    current_learning_rate: float
    
    # Error tracking
    error_history: deque = field(default_factory=lambda: deque(maxlen=50))
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Volatility tracking
    volatility_score: float = 0.0
    regime_change_detected: bool = False
    last_regime_change: Optional[GameweekId] = None
    
    # Convergence tracking
    loss_history: deque = field(default_factory=lambda: deque(maxlen=20))
    gradient_history: deque = field(default_factory=lambda: deque(maxlen=10))
    is_converged: bool = False
    patience_counter: int = 0
    
    # Algorithm-specific state
    algorithm_state: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    last_update: Optional[GameweekId] = None
    created_at: Optional[GameweekId] = None


class AdaptationAlgorithm(ABC):
    """Base class for learning rate adaptation algorithms."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        
    @abstractmethod
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize algorithm-specific state for a player."""
        pass
        
    @abstractmethod
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate based on gradient and loss information."""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Return algorithm name."""
        pass


class AdaGradOptimizer(AdaptationAlgorithm):
    """AdaGrad adaptive learning rate algorithm."""
    
    def get_name(self) -> str:
        return "AdaGrad"
        
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize AdaGrad state."""
        player_state.algorithm_state.update({
            'gradient_squared_sum': 0.0,
            'step_count': 0
        })
        
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate using AdaGrad algorithm."""
        state = player_state.algorithm_state
        
        # Accumulate squared gradients
        gradient_norm_squared = float(jnp.sum(gradient ** 2))
        state['gradient_squared_sum'] += gradient_norm_squared
        state['step_count'] += 1
        
        # Compute adaptive learning rate
        eps = self.config.numerical_stability_eps
        adaptive_rate = self.config.base_learning_rate / (
            jnp.sqrt(state['gradient_squared_sum']) + eps
        )
        
        # Apply bounds
        adaptive_rate = jnp.clip(
            adaptive_rate, 
            self.config.min_learning_rate, 
            self.config.max_learning_rate
        )
        
        return float(adaptive_rate)


class RMSPropOptimizer(AdaptationAlgorithm):
    """RMSProp adaptive learning rate algorithm."""
    
    def __init__(self, config: AdaptiveLearningConfig, decay_rate: float = 0.9):
        super().__init__(config)
        self.decay_rate = decay_rate
        
    def get_name(self) -> str:
        return "RMSProp"
        
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize RMSProp state."""
        player_state.algorithm_state.update({
            'exponential_avg_squared': 0.0,
            'step_count': 0
        })
        
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate using RMSProp algorithm."""
        state = player_state.algorithm_state
        
        # Compute exponentially weighted moving average of squared gradients
        gradient_norm_squared = float(jnp.sum(gradient ** 2))
        state['exponential_avg_squared'] = (
            self.decay_rate * state['exponential_avg_squared'] + 
            (1 - self.decay_rate) * gradient_norm_squared
        )
        state['step_count'] += 1
        
        # Compute adaptive learning rate
        eps = self.config.numerical_stability_eps
        adaptive_rate = self.config.base_learning_rate / (
            jnp.sqrt(state['exponential_avg_squared']) + eps
        )
        
        # Apply bounds
        adaptive_rate = jnp.clip(
            adaptive_rate, 
            self.config.min_learning_rate, 
            self.config.max_learning_rate
        )
        
        return float(adaptive_rate)


class AdamOptimizer(AdaptationAlgorithm):
    """Adam adaptive learning rate algorithm with bias correction."""
    
    def __init__(self, config: AdaptiveLearningConfig, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(config)
        self.beta1 = beta1
        self.beta2 = beta2
        
    def get_name(self) -> str:
        return "Adam"
        
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize Adam state."""
        player_state.algorithm_state.update({
            'momentum': 0.0,
            'velocity': 0.0,
            'step_count': 0
        })
        
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate using Adam algorithm."""
        state = player_state.algorithm_state
        state['step_count'] += 1
        
        # Compute gradient norm
        gradient_norm = float(jnp.sqrt(jnp.sum(gradient ** 2)))
        
        # Update biased first moment estimate (momentum)
        state['momentum'] = self.beta1 * state['momentum'] + (1 - self.beta1) * gradient_norm
        
        # Update biased second raw moment estimate (velocity)
        state['velocity'] = self.beta2 * state['velocity'] + (1 - self.beta2) * (gradient_norm ** 2)
        
        # Compute bias-corrected first moment estimate
        momentum_corrected = state['momentum'] / (1 - self.beta1 ** state['step_count'])
        
        # Compute bias-corrected second raw moment estimate
        velocity_corrected = state['velocity'] / (1 - self.beta2 ** state['step_count'])
        
        # Compute adaptive learning rate
        eps = self.config.numerical_stability_eps
        adaptive_rate = self.config.base_learning_rate / (
            jnp.sqrt(velocity_corrected) + eps
        )
        
        # Apply bounds
        adaptive_rate = jnp.clip(
            adaptive_rate, 
            self.config.min_learning_rate, 
            self.config.max_learning_rate
        )
        
        return float(adaptive_rate)


class AdaDeltaOptimizer(AdaptationAlgorithm):
    """AdaDelta adaptive learning rate algorithm."""
    
    def __init__(self, config: AdaptiveLearningConfig, rho: float = 0.95):
        super().__init__(config)
        self.rho = rho
        
    def get_name(self) -> str:
        return "AdaDelta"
        
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize AdaDelta state."""
        player_state.algorithm_state.update({
            'exponential_avg_squared_grad': 0.0,
            'exponential_avg_squared_delta': 0.0,
            'step_count': 0
        })
        
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate using AdaDelta algorithm."""
        state = player_state.algorithm_state
        state['step_count'] += 1
        
        # Compute gradient norm squared
        gradient_norm_squared = float(jnp.sum(gradient ** 2))
        
        # Update exponential average of squared gradients
        state['exponential_avg_squared_grad'] = (
            self.rho * state['exponential_avg_squared_grad'] + 
            (1 - self.rho) * gradient_norm_squared
        )
        
        # Compute adaptive learning rate
        eps = self.config.numerical_stability_eps
        rms_grad = jnp.sqrt(state['exponential_avg_squared_grad'] + eps)
        rms_delta = jnp.sqrt(state['exponential_avg_squared_delta'] + eps)
        
        adaptive_rate = rms_delta / rms_grad
        
        # Update exponential average of squared parameter updates
        delta_squared = adaptive_rate ** 2 * gradient_norm_squared
        state['exponential_avg_squared_delta'] = (
            self.rho * state['exponential_avg_squared_delta'] + 
            (1 - self.rho) * delta_squared
        )
        
        # Apply bounds
        adaptive_rate = jnp.clip(
            adaptive_rate, 
            self.config.min_learning_rate, 
            self.config.max_learning_rate
        )
        
        return float(adaptive_rate)


class CustomFPLOptimizer(AdaptationAlgorithm):
    """Custom FPL-specific adaptive learning rate algorithm."""
    
    def get_name(self) -> str:
        return "CustomFPL"
        
    def initialize_state(self, player_state: PlayerLearningState) -> None:
        """Initialize custom FPL state."""
        player_state.algorithm_state.update({
            'form_momentum': 0.0,
            'consistency_factor': 1.0,
            'recent_performance': deque(maxlen=5),
            'step_count': 0
        })
        
    def update_learning_rate(
        self, 
        player_state: PlayerLearningState,
        gradient: Array,
        loss: float,
        gameweek: GameweekId
    ) -> float:
        """Update learning rate using custom FPL algorithm."""
        state = player_state.algorithm_state
        state['step_count'] += 1
        
        # Track recent performance
        if len(player_state.error_history) > 0:
            recent_error = player_state.error_history[-1]
            state['recent_performance'].append(recent_error)
        
        # Compute form momentum (trend in recent errors)
        if len(state['recent_performance']) >= 3:
            recent_errors = list(state['recent_performance'])
            form_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
            state['form_momentum'] = 0.7 * state['form_momentum'] + 0.3 * form_trend
        
        # Compute consistency factor (inverse of error variance)
        if len(player_state.error_history) >= 5:
            error_variance = np.var(list(player_state.error_history)[-10:])
            state['consistency_factor'] = 1.0 / (1.0 + error_variance)
        
        # Position-specific base rate adjustment
        position_multiplier = {
            PlayerPosition.GOALKEEPER: 0.8,    # GKs are more predictable
            PlayerPosition.DEFENDER: 0.9,     # DEFs moderately predictable  
            PlayerPosition.MIDFIELDER: 1.1,   # MIDs need more adaptation
            PlayerPosition.FORWARD: 1.2       # FWDs most volatile
        }.get(player_state.position, 1.0)
        
        # Compute adaptive rate based on multiple factors
        base_rate = self.config.base_learning_rate * position_multiplier
        
        # Adjust for form momentum (increase rate if form is declining)
        momentum_adjustment = 1.0 + max(0, state['form_momentum']) * 0.5
        
        # Adjust for consistency (higher rates for inconsistent players)
        consistency_adjustment = 2.0 - state['consistency_factor']
        
        # Adjust for volatility
        volatility_adjustment = 1.0 + player_state.volatility_score * 0.3
        
        adaptive_rate = (base_rate * momentum_adjustment * 
                        consistency_adjustment * volatility_adjustment)
        
        # Apply bounds
        adaptive_rate = jnp.clip(
            adaptive_rate, 
            self.config.min_learning_rate, 
            self.config.max_learning_rate
        )
        
        return float(adaptive_rate)


class ErrorBasedAdapter:
    """Adjusts learning rates based on prediction errors and performance metrics."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        
    def adjust_for_error(
        self, 
        base_rate: float,
        prediction_error: float,
        actual_value: float,
        predicted_value: float,
        player_position: PlayerPosition
    ) -> float:
        """Adjust learning rate based on prediction error magnitude and direction."""
        
        # Get position-specific error threshold
        error_threshold = self.config.position_error_thresholds[player_position]
        
        # Normalize error by threshold
        normalized_error = abs(prediction_error) / error_threshold
        
        # Determine if over or under prediction
        is_over_prediction = predicted_value > actual_value
        
        # Apply asymmetric adjustment if enabled
        if self.config.asymmetric_adjustment:
            if is_over_prediction:
                adjustment_factor = 1.0 + normalized_error * self.config.over_prediction_factor
            else:
                adjustment_factor = 1.0 + normalized_error * self.config.under_prediction_factor
        else:
            adjustment_factor = 1.0 + normalized_error * self.config.error_sensitivity
        
        # Apply adjustment
        adjusted_rate = base_rate * adjustment_factor
        
        # Apply bounds
        return np.clip(adjusted_rate, self.config.min_learning_rate, self.config.max_learning_rate)
        
    def compute_windowed_error_metrics(
        self, 
        player_state: PlayerLearningState,
        window_size: int = 10
    ) -> Dict[str, float]:
        """Compute error metrics over a sliding window."""
        
        if len(player_state.error_history) < window_size:
            window_size = len(player_state.error_history)
            
        if window_size == 0:
            return {
                'mean_error': 0.0,
                'error_std': 0.0,
                'error_trend': 0.0,
                'max_error': 0.0
            }
        
        recent_errors = list(player_state.error_history)[-window_size:]
        
        return {
            'mean_error': float(np.mean(recent_errors)),
            'error_std': float(np.std(recent_errors)),
            'error_trend': float(np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]) 
                          if len(recent_errors) > 1 else 0.0,
            'max_error': float(np.max(np.abs(recent_errors)))
        }


class VolatilityDetector:
    """Detects player volatility and regime changes for adaptive learning."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        
    def compute_volatility_score(
        self, 
        player_state: PlayerLearningState
    ) -> float:
        """Compute volatility score based on prediction history variance."""
        
        if len(player_state.prediction_history) < 3:
            return 0.0
            
        # Use recent performance data
        recent_data = list(player_state.prediction_history)[-self.config.volatility_window:]
        
        if len(recent_data) < 3:
            return 0.0
            
        # Compute rolling variance
        volatility = float(np.std(recent_data))
        
        # Normalize by mean to get coefficient of variation
        mean_value = float(np.mean(recent_data))
        if mean_value > 0:
            volatility = volatility / mean_value
            
        return min(volatility, 1.0)  # Cap at 1.0
        
    def detect_regime_change(
        self, 
        player_state: PlayerLearningState,
        current_gameweek: GameweekId
    ) -> bool:
        """Detect if player has undergone a regime change (form shift)."""
        
        if len(player_state.prediction_history) < 8:
            return False
            
        recent_data = list(player_state.prediction_history)
        
        # Split data into two halves
        mid_point = len(recent_data) // 2
        early_half = recent_data[:mid_point]
        recent_half = recent_data[mid_point:]
        
        if len(early_half) < 2 or len(recent_half) < 2:
            return False
            
        # Compute means of each half
        early_mean = np.mean(early_half)
        recent_mean = np.mean(recent_half)
        
        # Compute pooled standard deviation
        pooled_std = np.sqrt(
            (np.var(early_half) * (len(early_half) - 1) + 
             np.var(recent_half) * (len(recent_half) - 1)) /
            (len(early_half) + len(recent_half) - 2)
        )
        
        # Compute t-statistic for difference in means
        if pooled_std > 0:
            t_stat = abs(recent_mean - early_mean) / (
                pooled_std * np.sqrt(1/len(early_half) + 1/len(recent_half))
            )
            
            # Check if change is significant
            regime_change = t_stat > self.config.regime_change_threshold
            
            if regime_change:
                player_state.last_regime_change = current_gameweek
                
            return regime_change
            
        return False
        
    def identify_outliers(
        self, 
        player_state: PlayerLearningState,
        z_threshold: float = 2.5
    ) -> List[int]:
        """Identify outlier predictions that might indicate volatility."""
        
        if len(player_state.prediction_history) < 5:
            return []
            
        data = np.array(list(player_state.prediction_history))
        
        # Compute z-scores
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return []
            
        z_scores = np.abs((data - mean_val) / std_val)
        
        # Find outliers
        outlier_indices = np.where(z_scores > z_threshold)[0]
        
        return outlier_indices.tolist()


class ConvergenceMonitor:
    """Monitors model convergence and detects training instability."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        
    def check_convergence(
        self, 
        player_state: PlayerLearningState,
        current_loss: float
    ) -> bool:
        """Check if model has converged for a player."""
        
        # Add current loss to history
        player_state.loss_history.append(current_loss)
        
        if len(player_state.loss_history) < self.config.convergence_patience:
            return False
            
        # Check for convergence criteria
        recent_losses = list(player_state.loss_history)[-self.config.convergence_patience:]
        
        # Criterion 1: Loss change is below threshold
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        loss_range = max_loss - min_loss
        
        if loss_range < self.config.convergence_threshold:
            player_state.patience_counter += 1
        else:
            player_state.patience_counter = 0
            
        # Criterion 2: Check for improvement
        if len(player_state.loss_history) >= 2:
            recent_improvement = player_state.loss_history[-2] - player_state.loss_history[-1]
            if recent_improvement < self.config.min_improvement:
                player_state.patience_counter += 1
            else:
                player_state.patience_counter = max(0, player_state.patience_counter - 1)
        
        # Converged if patience exceeded
        converged = player_state.patience_counter >= self.config.convergence_patience
        player_state.is_converged = converged
        
        return converged
        
    def detect_oscillation(
        self, 
        player_state: PlayerLearningState
    ) -> bool:
        """Detect if training is oscillating."""
        
        if len(player_state.loss_history) < 6:
            return False
            
        recent_losses = list(player_state.loss_history)[-6:]
        
        # Check for alternating increases/decreases
        changes = [recent_losses[i+1] - recent_losses[i] for i in range(len(recent_losses)-1)]
        
        # Detect oscillation pattern
        sign_changes = sum(1 for i in range(len(changes)-1) 
                          if changes[i] * changes[i+1] < 0)
        
        oscillation_ratio = sign_changes / (len(changes) - 1) if len(changes) > 1 else 0
        
        return oscillation_ratio > self.config.oscillation_threshold
        
    def detect_divergence(
        self, 
        player_state: PlayerLearningState,
        divergence_factor: float = 2.0
    ) -> bool:
        """Detect if training is diverging."""
        
        if len(player_state.loss_history) < 3:
            return False
            
        # Check if recent loss is much higher than historical average
        recent_loss = player_state.loss_history[-1]
        historical_losses = list(player_state.loss_history)[:-1]
        
        if len(historical_losses) == 0:
            return False
            
        historical_mean = np.mean(historical_losses)
        
        return recent_loss > historical_mean * divergence_factor
        
    def get_convergence_metrics(
        self, 
        player_state: PlayerLearningState
    ) -> Dict[str, Any]:
        """Get comprehensive convergence metrics."""
        
        if len(player_state.loss_history) == 0:
            return {
                'is_converged': False,
                'is_oscillating': False,
                'is_diverging': False,
                'loss_trend': 0.0,
                'loss_variance': 0.0,
                'patience_counter': 0
            }
            
        losses = list(player_state.loss_history)
        
        # Compute trend
        if len(losses) > 1:
            loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
        else:
            loss_trend = 0.0
            
        return {
            'is_converged': player_state.is_converged,
            'is_oscillating': self.detect_oscillation(player_state),
            'is_diverging': self.detect_divergence(player_state),
            'loss_trend': float(loss_trend),
            'loss_variance': float(np.var(losses)),
            'patience_counter': player_state.patience_counter,
            'recent_loss': float(losses[-1]) if losses else 0.0,
            'mean_loss': float(np.mean(losses))
        }


class LearningScheduler:
    """Implements various time-based learning rate schedules."""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        
    def exponential_decay(
        self, 
        base_rate: float, 
        gameweek: GameweekId, 
        start_gameweek: GameweekId = 1
    ) -> float:
        """Exponential decay schedule: lr(t) = lr_0 * γ^t"""
        
        t = max(0, gameweek - start_gameweek)
        decay_factor = self.config.decay_rate ** (t // self.config.decay_steps)
        
        return base_rate * decay_factor
        
    def step_decay(
        self, 
        base_rate: float, 
        gameweek: GameweekId,
        step_size: int = 10,
        gamma: float = 0.1
    ) -> float:
        """Step decay schedule with fixed intervals."""
        
        decay_factor = gamma ** (gameweek // step_size)
        return base_rate * decay_factor
        
    def cosine_annealing(
        self, 
        base_rate: float, 
        gameweek: GameweekId,
        total_gameweeks: int = 38,
        min_rate: Optional[float] = None
    ) -> float:
        """Cosine annealing schedule."""
        
        if min_rate is None:
            min_rate = self.config.min_learning_rate
            
        normalized_gameweek = gameweek / total_gameweeks
        cosine_factor = 0.5 * (1 + np.cos(np.pi * normalized_gameweek))
        
        return min_rate + (base_rate - min_rate) * cosine_factor
        
    def warm_restart(
        self, 
        base_rate: float, 
        gameweek: GameweekId,
        restart_period: int = 10,
        restart_factor: float = 2.0
    ) -> float:
        """Warm restart schedule with periodic rate resets."""
        
        # Determine current cycle
        cycle_gameweek = gameweek % restart_period
        
        # Apply cosine annealing within cycle
        cycle_rate = self.cosine_annealing(base_rate, cycle_gameweek, restart_period)
        
        # Apply restart factor based on cycle number
        cycle_number = gameweek // restart_period
        restart_adjustment = restart_factor ** cycle_number
        
        return cycle_rate / restart_adjustment
        
    def custom_fpl_schedule(
        self, 
        base_rate: float, 
        gameweek: GameweekId
    ) -> float:
        """Custom FPL-specific schedule accounting for fixture patterns."""
        
        # Higher rates during busy periods (double gameweeks, etc.)
        busy_periods = [16, 17, 18, 25, 26, 27, 33, 34, 35]  # Typical busy periods
        
        if gameweek in busy_periods:
            busy_multiplier = 1.3
        else:
            busy_multiplier = 1.0
            
        # Seasonal adjustment (higher rates early, lower late)
        if gameweek <= 10:
            seasonal_factor = 1.2  # Early season uncertainty
        elif gameweek <= 25:
            seasonal_factor = 1.0  # Mid season stability
        else:
            seasonal_factor = 0.8  # Late season predictability
            
        # Apply warmup for very early gameweeks
        if gameweek <= self.config.warmup_steps:
            warmup_factor = gameweek / self.config.warmup_steps
        else:
            warmup_factor = 1.0
            
        return base_rate * busy_multiplier * seasonal_factor * warmup_factor
        
    def get_scheduled_rate(
        self, 
        base_rate: float, 
        gameweek: GameweekId,
        schedule_type: Optional[str] = None
    ) -> float:
        """Get learning rate according to specified schedule."""
        
        if schedule_type is None:
            schedule_type = self.config.schedule_type
            
        if schedule_type == "exponential":
            return self.exponential_decay(base_rate, gameweek)
        elif schedule_type == "step":
            return self.step_decay(base_rate, gameweek)
        elif schedule_type == "cosine":
            return self.cosine_annealing(base_rate, gameweek)
        elif schedule_type == "warm_restart":
            return self.warm_restart(base_rate, gameweek)
        elif schedule_type == "custom_fpl":
            return self.custom_fpl_schedule(base_rate, gameweek)
        else:
            logger.warning(f"Unknown schedule type: {schedule_type}, using base rate")
            return base_rate


class AdaptiveLearningRateSystem:
    """
    Main controller for adaptive learning rate system.
    
    Orchestrates all components to provide intelligent learning rate adaptation
    based on prediction errors, player volatility, and convergence monitoring.
    """
    
    def __init__(self, config: Optional[AdaptiveLearningConfig] = None):
        """Initialize the adaptive learning rate system."""
        
        self.config = config or AdaptiveLearningConfig()
        
        # Initialize components
        self.error_adapter = ErrorBasedAdapter(self.config)
        self.volatility_detector = VolatilityDetector(self.config)
        self.convergence_monitor = ConvergenceMonitor(self.config)
        self.scheduler = LearningScheduler(self.config)
        
        # Initialize adaptation algorithm
        self.adaptation_algorithm = self._create_adaptation_algorithm()
        
        # Player state storage
        self.player_states: Dict[PlayerId, PlayerLearningState] = {}
        
        # Global statistics
        self.global_stats = {
            'total_adaptations': 0,
            'convergence_count': 0,
            'divergence_count': 0,
            'regime_changes': 0
        }
        
        logger.info(f"Initialized AdaptiveLearningRateSystem with {self.adaptation_algorithm.get_name()}")
        
    def _create_adaptation_algorithm(self) -> AdaptationAlgorithm:
        """Create the specified adaptation algorithm."""
        
        if self.config.adaptation_method == AdaptationMethod.ADAGRAD:
            return AdaGradOptimizer(self.config)
        elif self.config.adaptation_method == AdaptationMethod.RMSPROP:
            return RMSPropOptimizer(self.config)
        elif self.config.adaptation_method == AdaptationMethod.ADAM:
            return AdamOptimizer(self.config)
        elif self.config.adaptation_method == AdaptationMethod.ADADELTA:
            return AdaDeltaOptimizer(self.config)
        elif self.config.adaptation_method == AdaptationMethod.CUSTOM_FPL:
            return CustomFPLOptimizer(self.config)
        else:
            logger.warning(f"Unknown adaptation method: {self.config.adaptation_method}, using Adam")
            return AdamOptimizer(self.config)
            
    def get_or_create_player_state(
        self, 
        player_id: PlayerId, 
        position: PlayerPosition,
        gameweek: GameweekId
    ) -> PlayerLearningState:
        """Get existing player state or create new one."""
        
        if player_id not in self.player_states:
            player_state = PlayerLearningState(
                player_id=player_id,
                position=position,
                current_learning_rate=self.config.base_learning_rate,
                created_at=gameweek
            )
            
            # Initialize algorithm-specific state
            self.adaptation_algorithm.initialize_state(player_state)
            
            self.player_states[player_id] = player_state
            logger.debug(f"Created new player state for player {player_id}")
            
        return self.player_states[player_id]
        
    def adapt_learning_rate(
        self, 
        player_id: PlayerId,
        position: PlayerPosition,
        prediction_error: float,
        actual_value: float,
        predicted_value: float,
        gameweek: GameweekId,
        gradient: Optional[Array] = None,
        loss: Optional[float] = None
    ) -> float:
        """
        Adapt learning rate for a specific player based on multiple factors.
        
        Args:
            player_id: Unique player identifier
            position: Player position (GK, DEF, MID, FWD)
            prediction_error: Absolute prediction error
            actual_value: Actual performance value
            predicted_value: Predicted performance value  
            gameweek: Current gameweek
            gradient: Optional gradient information for algorithm-based adaptation
            loss: Optional loss value for convergence monitoring
            
        Returns:
            Adapted learning rate for the player
        """
        
        # Get or create player state
        player_state = self.get_or_create_player_state(player_id, position, gameweek)
        
        # Update player history
        player_state.error_history.append(prediction_error)
        player_state.prediction_history.append(actual_value)
        player_state.last_update = gameweek
        
        # Update volatility score
        player_state.volatility_score = self.volatility_detector.compute_volatility_score(player_state)
        
        # Check for regime change
        player_state.regime_change_detected = self.volatility_detector.detect_regime_change(
            player_state, gameweek
        )
        if player_state.regime_change_detected:
            self.global_stats['regime_changes'] += 1
        
        # Start with scheduled base rate
        base_rate = self.scheduler.get_scheduled_rate(
            self.config.base_learning_rate, gameweek
        )
        
        # Apply error-based adaptation
        error_adjusted_rate = self.error_adapter.adjust_for_error(
            base_rate, prediction_error, actual_value, predicted_value, position
        )
        
        # Apply algorithm-based adaptation if gradient available
        if gradient is not None and loss is not None:
            algorithm_rate = self.adaptation_algorithm.update_learning_rate(
                player_state, gradient, loss, gameweek
            )
            # Combine error-based and algorithm-based rates
            combined_rate = 0.6 * error_adjusted_rate + 0.4 * algorithm_rate
        else:
            combined_rate = error_adjusted_rate
            
        # Apply volatility adjustment
        volatility_adjustment = 1.0 + player_state.volatility_score * 0.2
        volatility_adjusted_rate = combined_rate * volatility_adjustment
        
        # Apply regime change boost
        if player_state.regime_change_detected:
            regime_adjustment = 1.5  # Boost learning rate for regime changes
            volatility_adjusted_rate *= regime_adjustment
            
        # Apply risk management bounds
        final_rate = self._apply_risk_management(
            volatility_adjusted_rate, player_state, gameweek
        )
        
        # Update player state
        player_state.current_learning_rate = final_rate
        
        # Update global statistics
        self.global_stats['total_adaptations'] += 1
        
        logger.debug(
            f"Adapted learning rate for player {player_id}: "
            f"{self.config.base_learning_rate:.6f} -> {final_rate:.6f} "
            f"(error: {prediction_error:.3f}, volatility: {player_state.volatility_score:.3f})"
        )
        
        return final_rate
        
    def _apply_risk_management(
        self, 
        rate: float, 
        player_state: PlayerLearningState,
        gameweek: GameweekId
    ) -> float:
        """Apply risk management constraints to learning rate."""
        
        # Apply hard bounds
        bounded_rate = np.clip(rate, self.config.min_learning_rate, self.config.max_learning_rate)
        
        # Check for divergence and apply rollback if needed
        if self.config.rollback_on_divergence and len(player_state.loss_history) > 0:
            is_diverging = self.convergence_monitor.detect_divergence(player_state)
            if is_diverging:
                # Reduce rate significantly
                bounded_rate *= 0.1
                logger.warning(f"Divergence detected for player {player_state.player_id}, reducing rate")
                self.global_stats['divergence_count'] += 1
                
        # Check for oscillation and dampen if needed
        is_oscillating = self.convergence_monitor.detect_oscillation(player_state)
        if is_oscillating:
            bounded_rate *= 0.5
            logger.debug(f"Oscillation detected for player {player_state.player_id}, dampening rate")
            
        return float(bounded_rate)
        
    def check_convergence(
        self, 
        player_id: PlayerId, 
        loss: float
    ) -> bool:
        """Check if model has converged for a player."""
        
        if player_id not in self.player_states:
            return False
            
        player_state = self.player_states[player_id]
        converged = self.convergence_monitor.check_convergence(player_state, loss)
        
        if converged and not player_state.is_converged:
            self.global_stats['convergence_count'] += 1
            logger.info(f"Convergence achieved for player {player_id}")
            
        return converged
        
    def get_convergence_metrics(
        self, 
        player_id: PlayerId
    ) -> Dict[str, Any]:
        """Get convergence metrics for a player."""
        
        if player_id not in self.player_states:
            return {}
            
        return self.convergence_monitor.get_convergence_metrics(
            self.player_states[player_id]
        )
        
    def get_current_learning_rate(self, player_id: PlayerId) -> float:
        """Get current learning rate for a player."""
        
        if player_id in self.player_states:
            return self.player_states[player_id].current_learning_rate
        return self.config.base_learning_rate
        
    def get_player_volatility(self, player_id: PlayerId) -> float:
        """Get volatility score for a player."""
        
        if player_id in self.player_states:
            return self.player_states[player_id].volatility_score
        return 0.0
        
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global system statistics."""
        
        stats = dict(self.global_stats)
        stats.update({
            'total_players': len(self.player_states),
            'converged_players': sum(1 for state in self.player_states.values() 
                                   if state.is_converged),
            'avg_learning_rate': np.mean([state.current_learning_rate 
                                        for state in self.player_states.values()]),
            'avg_volatility': np.mean([state.volatility_score 
                                     for state in self.player_states.values()]),
            'adaptation_algorithm': self.adaptation_algorithm.get_name()
        })
        
        return stats
        
    def reset_player_state(self, player_id: PlayerId) -> None:
        """Reset learning state for a specific player."""
        
        if player_id in self.player_states:
            del self.player_states[player_id]
            logger.info(f"Reset learning state for player {player_id}")
            
    def save_state(self) -> Dict[str, Any]:
        """Save current system state for persistence."""
        
        # Convert player states to serializable format
        serializable_states = {}
        for player_id, state in self.player_states.items():
            serializable_states[player_id] = {
                'player_id': state.player_id,
                'position': state.position.value,
                'current_learning_rate': state.current_learning_rate,
                'error_history': list(state.error_history),
                'prediction_history': list(state.prediction_history),
                'volatility_score': state.volatility_score,
                'regime_change_detected': state.regime_change_detected,
                'last_regime_change': state.last_regime_change,
                'loss_history': list(state.loss_history),
                'is_converged': state.is_converged,
                'patience_counter': state.patience_counter,
                'algorithm_state': state.algorithm_state,
                'last_update': state.last_update,
                'created_at': state.created_at
            }
            
        return {
            'config': {
                'base_learning_rate': self.config.base_learning_rate,
                'adaptation_method': self.config.adaptation_method.value,
                'volatility_threshold': self.config.volatility_threshold,
                'max_learning_rate': self.config.max_learning_rate,
                'min_learning_rate': self.config.min_learning_rate
            },
            'player_states': serializable_states,
            'global_stats': self.global_stats
        }
        
    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load system state from saved data."""
        
        # Load player states
        for player_id_str, state_data in state_dict.get('player_states', {}).items():
            player_id = int(player_id_str)
            
            # Reconstruct player state
            player_state = PlayerLearningState(
                player_id=state_data['player_id'],
                position=PlayerPosition(state_data['position']),
                current_learning_rate=state_data['current_learning_rate'],
                volatility_score=state_data['volatility_score'],
                regime_change_detected=state_data['regime_change_detected'],
                last_regime_change=state_data['last_regime_change'],
                is_converged=state_data['is_converged'],
                patience_counter=state_data['patience_counter'],
                last_update=state_data['last_update'],
                created_at=state_data['created_at']
            )
            
            # Restore history
            player_state.error_history.extend(state_data['error_history'])
            player_state.prediction_history.extend(state_data['prediction_history'])
            player_state.loss_history.extend(state_data['loss_history'])
            
            # Restore algorithm state
            player_state.algorithm_state = state_data['algorithm_state']
            
            self.player_states[player_id] = player_state
            
        # Load global stats
        self.global_stats.update(state_dict.get('global_stats', {}))
        
        logger.info(f"Loaded state for {len(self.player_states)} players")
        
    def integrate_with_kalman_filter(
        self, 
        player_id: PlayerId,
        kalman_gain: Array,
        process_noise: Array,
        measurement_noise: Array
    ) -> Tuple[Array, Array, Array]:
        """
        Integrate adaptive learning rates with Kalman filter parameters.
        
        Args:
            player_id: Player identifier
            kalman_gain: Current Kalman gain matrix
            process_noise: Current process noise covariance
            measurement_noise: Current measurement noise covariance
            
        Returns:
            Tuple of adjusted (kalman_gain, process_noise, measurement_noise)
        """
        
        if player_id not in self.player_states:
            return kalman_gain, process_noise, measurement_noise
            
        player_state = self.player_states[player_id]
        learning_rate = player_state.current_learning_rate
        volatility = player_state.volatility_score
        
        # Adjust Kalman gain based on learning rate
        # Higher learning rates -> higher Kalman gain (trust observations more)
        gain_adjustment = learning_rate / self.config.base_learning_rate
        adjusted_kalman_gain = kalman_gain * gain_adjustment
        
        # Adjust process noise based on volatility
        # Higher volatility -> higher process noise (more uncertainty in dynamics)
        if self.config.update_process_noise:
            noise_adjustment = 1.0 + volatility * self.config.trust_adjustment_factor
            adjusted_process_noise = process_noise * noise_adjustment
        else:
            adjusted_process_noise = process_noise
            
        # Adjust measurement noise based on recent errors
        if len(player_state.error_history) > 0:
            recent_error_std = np.std(list(player_state.error_history)[-5:])
            measurement_adjustment = 1.0 + recent_error_std * 0.1
            adjusted_measurement_noise = measurement_noise * measurement_adjustment
        else:
            adjusted_measurement_noise = measurement_noise
            
        return adjusted_kalman_gain, adjusted_process_noise, adjusted_measurement_noise