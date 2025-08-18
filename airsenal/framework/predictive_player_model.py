"""
Predictive Player Model for AIrsenal

This module extends the KalmanPlayerModel with comprehensive state prediction capabilities,
integrating the state prediction framework with the existing Kalman filtering system to
provide a unified interface for multi-step forecasting with uncertainty quantification.

The PredictivePlayerModel serves as the main interface for AIrsenal's optimization
algorithms, providing accurate, well-calibrated forecasts with proper risk assessment
that can be used for risk-aware decision making.

Key Features:
- Seamless integration with KalmanPlayerModel and state prediction framework
- Multi-step ahead predictions with uncertainty propagation
- Fixture-aware adjustments and seasonality handling
- Risk metrics calculation for optimization integration
- Scenario analysis capabilities
- Comprehensive validation and backtesting
- Caching for efficient repeated predictions
- API compatibility with existing AIrsenal optimization systems

Classes:
    PredictivePlayerModel: Main predictive model extending KalmanPlayerModel
    ModelEnsemble: Ensemble of predictive models for robust forecasting
    PredictiveModelError: Custom exceptions for predictive modeling

Usage:
    ```python
    from airsenal.framework.predictive_player_model import PredictivePlayerModel
    from airsenal.framework.kalman_filter import FilterConfig
    
    # Configure and create model
    config = FilterConfig(state_dim=4, obs_dim=4)
    model = PredictivePlayerModel(
        filter_config=config,
        enable_fixture_awareness=True,
        enable_seasonality=True
    )
    
    # Fit to historical data
    model.fit(training_data, season="2023", max_gameweek=15)
    
    # Multi-step predictions
    predictions = model.predict_player_performance(
        player_id=123,
        gameweeks_ahead=3,
        include_intervals=True,
        include_risk_metrics=True
    )
    
    # Optimization integration
    expected_points = model.get_expected_points([123, 456, 789], gameweeks=3)
    risk_adjusted_points = model.get_risk_adjusted_points([123, 456, 789], risk_aversion=0.5)
    ```
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.adaptive_player_model import (
    AdaptivePlayerModel,
    PlayerState,
    StateSpaceConfig,
    PlayerData,
    PredictionOutput
)
from airsenal.framework.kalman_player_model import (
    KalmanPlayerModel,
    KalmanPlayerModelError,
    FilterType,
    ModelMode
)
from airsenal.framework.kalman_filter import (
    FilterConfig,
    FilterState,
    KalmanFilterError
)
from airsenal.framework.state_prediction import (
    StatePredictor,
    PredictionConfig,
    PredictionResult,
    PredictionInterval,
    RiskMetrics,
    StatePredictionError,
    get_predictions,
    get_prediction_intervals,
    get_risk_metrics,
    scenario_analysis
)

logger = logging.getLogger(__name__)


class PredictiveModelError(Exception):
    """Base exception for predictive player model operations."""
    pass


class PredictivePlayerModel(KalmanPlayerModel):
    """
    Comprehensive predictive player model extending KalmanPlayerModel with
    state prediction capabilities, risk assessment, and optimization integration.
    
    This class provides the main interface for AIrsenal's optimization algorithms,
    combining Kalman filtering with advanced prediction techniques for accurate,
    well-calibrated forecasts with proper uncertainty quantification.
    """
    
    def __init__(
        self,
        config: Optional[StateSpaceConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        prediction_config: Optional[PredictionConfig] = None,
        filter_type: FilterType = "extended",
        model_mode: ModelMode = "single",
        enable_fixture_awareness: bool = True,
        enable_seasonality: bool = True,
        learning_rate: float = 0.01,
        decay_factor: float = 0.95,
        random_seed: int = 42,
        session: Optional[Session] = None
    ):
        """
        Initialize Predictive Player Model.
        
        Args:
            config: State space configuration
            filter_config: Kalman filter configuration
            prediction_config: State prediction configuration
            filter_type: Type of Kalman filter to use
            model_mode: Model mode (single, position_specific, ensemble)
            enable_fixture_awareness: Enable fixture-aware predictions
            enable_seasonality: Enable seasonal adjustments
            learning_rate: Learning rate for adaptive components
            decay_factor: Decay factor for historical data
            random_seed: Random seed for reproducibility
            session: Database session for fixture data
        """
        # Initialize base Kalman model
        super().__init__(
            config=config,
            filter_config=filter_config,
            filter_type=filter_type,
            model_mode=model_mode,
            learning_rate=learning_rate,
            decay_factor=decay_factor,
            random_seed=random_seed
        )
        
        # Prediction configuration
        self.prediction_config = prediction_config or PredictionConfig()
        self.enable_fixture_awareness = enable_fixture_awareness
        self.enable_seasonality = enable_seasonality
        self.session = session
        
        # Initialize state predictor (will be set after fitting)
        self.state_predictor: Optional[StatePredictor] = None
        
        # Performance tracking
        self.prediction_performance = {
            "total_predictions": 0,
            "cache_hits": 0,
            "average_prediction_time": 0.0,
            "validation_results": []
        }
        
        # Optimization interface cache
        self.optimization_cache = {}
        
        logger.info(f"Initialized PredictivePlayerModel with fixture_awareness={enable_fixture_awareness}, seasonality={enable_seasonality}")
    
    def fit(
        self,
        data: PlayerData,
        season: str = "2023",
        max_gameweek: int = 38,
        validate: bool = True,
        **kwargs
    ) -> None:
        """
        Fit the predictive model to historical data.
        
        Args:
            data: Training data
            season: Season identifier
            max_gameweek: Maximum gameweek for training
            validate: Whether to perform validation
            **kwargs: Additional fitting parameters
        """
        try:
            logger.info(f"Fitting PredictivePlayerModel for season {season}, max_gameweek {max_gameweek}")
            
            # Fit base Kalman model
            super().fit(data, season, max_gameweek, **kwargs)
            
            # Initialize state predictor with fitted model
            self.state_predictor = StatePredictor(
                kalman_model=self,
                config=self.prediction_config,
                enable_fixture_awareness=self.enable_fixture_awareness,
                enable_seasonality=self.enable_seasonality,
                session=self.session
            )
            
            # Perform validation if requested
            if validate and len(self.player_states) > 0:
                self._validate_model_performance(data, season, max_gameweek)
            
            logger.info(f"Successfully fitted PredictivePlayerModel with {len(self.player_states)} players")
            
        except Exception as e:
            logger.error(f"Failed to fit PredictivePlayerModel: {e}")
            raise PredictiveModelError(f"Model fitting failed: {e}")
    
    def _validate_model_performance(
        self,
        data: PlayerData,
        season: str,
        max_gameweek: int
    ) -> None:
        """Validate model performance using backtesting."""
        try:
            # Simple validation: predict last few gameweeks and compare to actuals
            validation_horizon = min(3, max_gameweek // 4)  # Use last 25% of season for validation
            training_cutoff = max_gameweek - validation_horizon
            
            predictions = []
            actuals = []
            player_ids = []
            
            for player_id in list(self.player_states.keys())[:50]:  # Validate on subset for efficiency
                try:
                    # Get validation period data
                    player_idx = data["player_ids"].index(player_id) if player_id in data["player_ids"] else None
                    if player_idx is None:
                        continue
                    
                    # Predict from training cutoff
                    for gw in range(training_cutoff + 1, max_gameweek + 1):
                        pred_result = self.predict_player_performance(
                            player_id=player_id,
                            gameweeks_ahead=gw - training_cutoff,
                            current_gameweek=training_cutoff,
                            season=season
                        )
                        
                        # Get actual performance for this gameweek
                        actual_performance = data["features"][player_idx][gw - 1]  # Assuming 0-indexed
                        
                        if not np.any(np.isnan(actual_performance)):
                            predictions.append(pred_result)
                            actuals.append(actual_performance[:self.config.obs_dim])
                            player_ids.append(player_id)
                
                except Exception as e:
                    logger.warning(f"Validation failed for player {player_id}: {e}")
                    continue
            
            if len(predictions) >= self.prediction_config.min_validation_samples:
                validation_results = self.state_predictor.validator.validate_predictions(
                    predictions, actuals, player_ids
                )
                self.prediction_performance["validation_results"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "season": season,
                    "results": validation_results
                })
                
                logger.info(f"Validation completed with {len(predictions)} samples: "
                           f"MAE={validation_results['calibration_metrics']['mean_absolute_error']:.3f}, "
                           f"RMSE={validation_results['calibration_metrics']['root_mean_square_error']:.3f}")
        
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
    
    def predict_player_performance(
        self,
        player_id: int,
        gameweeks_ahead: int = 3,
        confidence_levels: Optional[List[float]] = None,
        current_gameweek: Optional[int] = None,
        season: str = "2024",
        include_intervals: bool = True,
        include_risk_metrics: bool = True,
        external_factors: Optional[Dict[str, float]] = None
    ) -> PredictionResult:
        """
        Predict player performance with comprehensive uncertainty quantification.
        
        Args:
            player_id: Player ID to predict
            gameweeks_ahead: Number of gameweeks ahead to predict
            confidence_levels: Confidence levels for prediction intervals
            current_gameweek: Current gameweek (auto-detected if None)
            season: Season identifier
            include_intervals: Whether to calculate prediction intervals
            include_risk_metrics: Whether to calculate risk metrics
            external_factors: Additional factors affecting uncertainty
            
        Returns:
            Comprehensive prediction result
        """
        if not self.is_fitted or self.state_predictor is None:
            raise PredictiveModelError("Model must be fitted before making predictions")
        
        start_time = time.time()
        
        try:
            # Use state predictor for comprehensive prediction
            result = self.state_predictor.predict_multi_step(
                player_id=player_id,
                gameweeks_ahead=gameweeks_ahead,
                confidence_levels=confidence_levels if include_intervals else None,
                current_gameweek=current_gameweek,
                season=season,
                include_risk_metrics=include_risk_metrics,
                external_factors=external_factors
            )
            
            # Update performance tracking
            prediction_time = time.time() - start_time
            self.prediction_performance["total_predictions"] += 1
            
            # Update running average of prediction time
            total_preds = self.prediction_performance["total_predictions"]
            current_avg = self.prediction_performance["average_prediction_time"]
            new_avg = ((total_preds - 1) * current_avg + prediction_time) / total_preds
            self.prediction_performance["average_prediction_time"] = new_avg
            
            return result
            
        except Exception as e:
            logger.error(f"Player performance prediction failed for player {player_id}: {e}")
            raise PredictiveModelError(f"Prediction failed: {e}")
    
    def predict_batch(
        self,
        player_ids: List[int],
        gameweeks_ahead: int = 3,
        **kwargs
    ) -> Dict[int, PredictionResult]:
        """
        Predict performance for multiple players efficiently.
        
        Args:
            player_ids: List of player IDs
            gameweeks_ahead: Prediction horizon
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary mapping player IDs to prediction results
        """
        if not self.state_predictor:
            raise PredictiveModelError("Model must be fitted before making predictions")
        
        return self.state_predictor.get_predictions(player_ids, gameweeks_ahead, **kwargs)
    
    def get_expected_points(
        self,
        player_ids: List[int],
        gameweeks: int = 3,
        season: str = "2024"
    ) -> Dict[int, float]:
        """
        Get expected FPL points for optimization algorithms.
        
        Args:
            player_ids: List of player IDs
            gameweeks: Number of gameweeks ahead
            season: Season identifier
            
        Returns:
            Dictionary mapping player IDs to expected points
        """
        cache_key = f"expected_points_{tuple(player_ids)}_{gameweeks}_{season}"
        
        if cache_key in self.optimization_cache:
            self.prediction_performance["cache_hits"] += 1
            return self.optimization_cache[cache_key]
        
        try:
            predictions = self.predict_batch(player_ids, gameweeks)
            expected_points = {}
            
            for player_id, pred_result in predictions.items():
                expected_points[player_id] = pred_result.get_expected_points()
            
            # Cache result
            self.optimization_cache[cache_key] = expected_points
            
            return expected_points
            
        except Exception as e:
            logger.error(f"Failed to get expected points: {e}")
            raise PredictiveModelError(f"Expected points calculation failed: {e}")
    
    def get_risk_adjusted_points(
        self,
        player_ids: List[int],
        risk_aversion: float = 0.5,
        gameweeks: int = 3,
        risk_measure: str = "sharpe"
    ) -> Dict[int, float]:
        """
        Get risk-adjusted expected points for risk-aware optimization.
        
        Args:
            player_ids: List of player IDs
            risk_aversion: Risk aversion parameter (0 = risk-neutral, 1 = maximum risk aversion)
            gameweeks: Number of gameweeks ahead
            risk_measure: Risk measure to use ("sharpe", "var", "cvar")
            
        Returns:
            Dictionary mapping player IDs to risk-adjusted expected points
        """
        try:
            predictions = self.predict_batch(player_ids, gameweeks, include_risk_metrics=True)
            risk_adjusted_points = {}
            
            for player_id, pred_result in predictions.items():
                expected_points = pred_result.get_expected_points()
                
                if pred_result.risk_metrics is None:
                    risk_adjusted_points[player_id] = expected_points
                    continue
                
                # Apply risk adjustment based on selected measure
                if risk_measure == "sharpe":
                    risk_penalty = risk_aversion * (1.0 - pred_result.risk_metrics.sharpe_ratio)
                elif risk_measure == "var":
                    risk_penalty = risk_aversion * pred_result.risk_metrics.value_at_risk
                elif risk_measure == "cvar":
                    risk_penalty = risk_aversion * pred_result.risk_metrics.conditional_var
                else:
                    risk_penalty = 0.0
                
                risk_adjusted_points[player_id] = expected_points - risk_penalty
            
            return risk_adjusted_points
            
        except Exception as e:
            logger.error(f"Failed to get risk-adjusted points: {e}")
            raise PredictiveModelError(f"Risk-adjusted points calculation failed: {e}")
    
    def get_prediction_intervals(
        self,
        player_ids: List[int],
        confidence_level: float = 0.8,
        gameweeks: int = 3
    ) -> Dict[int, PredictionInterval]:
        """
        Get prediction intervals for multiple players.
        
        Args:
            player_ids: List of player IDs
            confidence_level: Confidence level for intervals
            gameweeks: Number of gameweeks ahead
            
        Returns:
            Dictionary mapping player IDs to prediction intervals
        """
        try:
            predictions = self.predict_batch(player_ids, gameweeks, confidence_levels=[confidence_level])
            intervals = {}
            
            for player_id, pred_result in predictions.items():
                interval = pred_result.get_interval(confidence_level)
                if interval:
                    intervals[player_id] = interval
            
            return intervals
            
        except Exception as e:
            logger.error(f"Failed to get prediction intervals: {e}")
            raise PredictiveModelError(f"Prediction intervals calculation failed: {e}")
    
    def scenario_analysis(
        self,
        player_id: int,
        scenarios: Dict[str, Dict[str, Any]],
        gameweeks_ahead: int = 3
    ) -> Dict[str, PredictionResult]:
        """
        Perform scenario analysis for a player.
        
        Args:
            player_id: Player ID
            scenarios: Dictionary of scenario parameters
            gameweeks_ahead: Prediction horizon
            
        Returns:
            Dictionary mapping scenario names to prediction results
        """
        if not self.state_predictor:
            raise PredictiveModelError("Model must be fitted before scenario analysis")
        
        return scenario_analysis(player_id, scenarios, self.state_predictor, gameweeks_ahead)
    
    def compare_players(
        self,
        player_ids: List[int],
        metrics: List[str] = None,
        gameweeks_ahead: int = 3
    ) -> pd.DataFrame:
        """
        Compare multiple players across specified metrics.
        
        Args:
            player_ids: List of player IDs to compare
            metrics: List of metrics to compare
            gameweeks_ahead: Prediction horizon
            
        Returns:
            DataFrame with player comparison
        """
        if not self.state_predictor:
            raise PredictiveModelError("Model must be fitted before player comparison")
        
        return self.state_predictor.compare_players(player_ids, metrics, gameweeks_ahead)
    
    def get_top_performers(
        self,
        position: Optional[str] = None,
        metric: str = "expected_points",
        top_n: int = 10,
        gameweeks_ahead: int = 3,
        min_minutes: float = 60.0
    ) -> List[Tuple[int, float]]:
        """
        Get top performing players by specified metric.
        
        Args:
            position: Filter by position (None for all positions)
            metric: Metric to rank by
            top_n: Number of top players to return
            gameweeks_ahead: Prediction horizon
            min_minutes: Minimum expected minutes threshold
            
        Returns:
            List of (player_id, metric_value) tuples sorted by metric
        """
        try:
            # Get all available players
            player_ids = list(self.player_states.keys())
            
            # Filter by position if specified
            if position:
                # This would require position data - simplified for now
                pass
            
            # Get predictions
            predictions = self.predict_batch(player_ids, gameweeks_ahead, include_risk_metrics=True)
            
            # Calculate metrics and filter
            player_metrics = []
            for player_id, pred_result in predictions.items():
                # Filter by minimum minutes
                expected_minutes = pred_result.expected_performance[2] if len(pred_result.expected_performance) > 2 else 0
                if expected_minutes < min_minutes:
                    continue
                
                # Calculate metric value
                if metric == "expected_points":
                    value = pred_result.get_expected_points()
                elif metric == "sharpe_ratio" and pred_result.risk_metrics:
                    value = pred_result.risk_metrics.sharpe_ratio
                elif metric == "value_at_risk" and pred_result.risk_metrics:
                    value = -pred_result.risk_metrics.value_at_risk  # Negative for ranking
                else:
                    continue
                
                player_metrics.append((player_id, value))
            
            # Sort and return top N
            player_metrics.sort(key=lambda x: x[1], reverse=True)
            return player_metrics[:top_n]
            
        except Exception as e:
            logger.error(f"Failed to get top performers: {e}")
            return []
    
    def get_transfer_recommendations(
        self,
        current_squad: List[int],
        available_players: List[int],
        budget: float,
        num_transfers: int = 1,
        gameweeks_ahead: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get transfer recommendations based on predictions and budget constraints.
        
        Args:
            current_squad: Current squad player IDs
            available_players: Available players for transfer
            budget: Available budget
            num_transfers: Number of transfers to recommend
            gameweeks_ahead: Prediction horizon
            
        Returns:
            List of transfer recommendations
        """
        try:
            # Get predictions for current squad and available players
            all_players = list(set(current_squad + available_players))
            predictions = self.predict_batch(all_players, gameweeks_ahead, include_risk_metrics=True)
            
            recommendations = []
            
            # Simple recommendation logic - replace lowest expected points with highest available
            squad_performance = {}
            for player_id in current_squad:
                if player_id in predictions:
                    squad_performance[player_id] = predictions[player_id].get_expected_points()
            
            available_performance = {}
            for player_id in available_players:
                if player_id in predictions:
                    available_performance[player_id] = predictions[player_id].get_expected_points()
            
            # Sort current squad by performance (ascending)
            sorted_squad = sorted(squad_performance.items(), key=lambda x: x[1])
            
            # Sort available players by performance (descending)
            sorted_available = sorted(available_performance.items(), key=lambda x: x[1], reverse=True)
            
            # Generate transfer recommendations
            for i in range(min(num_transfers, len(sorted_squad), len(sorted_available))):
                transfer_out = sorted_squad[i]
                transfer_in = sorted_available[i]
                
                expected_improvement = transfer_in[1] - transfer_out[1]
                
                if expected_improvement > 0:
                    recommendation = {
                        "transfer_out": transfer_out[0],
                        "transfer_in": transfer_in[0],
                        "expected_improvement": expected_improvement,
                        "transfer_out_points": transfer_out[1],
                        "transfer_in_points": transfer_in[1]
                    }
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get transfer recommendations: {e}")
            return []
    
    def get_captain_recommendation(
        self,
        squad_player_ids: List[int],
        gameweeks_ahead: int = 1,
        risk_tolerance: float = 0.5
    ) -> Optional[int]:
        """
        Recommend captain based on expected points and risk profile.
        
        Args:
            squad_player_ids: Current squad player IDs
            gameweeks_ahead: Prediction horizon (typically 1 for captaincy)
            risk_tolerance: Risk tolerance (0 = conservative, 1 = aggressive)
            
        Returns:
            Recommended captain player ID
        """
        try:
            predictions = self.predict_batch(squad_player_ids, gameweeks_ahead, include_risk_metrics=True)
            
            captain_scores = {}
            for player_id, pred_result in predictions.items():
                expected_points = pred_result.get_expected_points()
                
                if pred_result.risk_metrics:
                    # Combine expected points with risk-adjusted score
                    risk_adjustment = risk_tolerance * pred_result.risk_metrics.sharpe_ratio
                    captain_score = expected_points + risk_adjustment
                else:
                    captain_score = expected_points
                
                captain_scores[player_id] = captain_score
            
            if captain_scores:
                return max(captain_scores, key=captain_scores.get)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get captain recommendation: {e}")
            return None
    
    def clear_optimization_cache(self):
        """Clear optimization interface cache."""
        self.optimization_cache.clear()
        if self.state_predictor:
            self.state_predictor.clear_cache()
        logger.debug("Optimization cache cleared")
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics including prediction performance."""
        base_diagnostics = super().get_model_diagnostics()
        
        predictive_diagnostics = {
            "prediction_performance": self.prediction_performance.copy(),
            "optimization_cache_size": len(self.optimization_cache),
            "state_predictor_diagnostics": self.state_predictor.get_diagnostics() if self.state_predictor else {},
            "prediction_config": {
                "max_gameweeks_ahead": self.prediction_config.max_gameweeks_ahead,
                "enable_fixture_awareness": self.enable_fixture_awareness,
                "enable_seasonality": self.enable_seasonality,
                "uncertainty_inflation_rate": self.prediction_config.uncertainty_inflation_rate
            }
        }
        
        return {**base_diagnostics, **predictive_diagnostics}
    
    def save_model_state(self, filepath: str) -> None:
        """
        Save complete model state for persistence.
        
        Args:
            filepath: Path to save model state
        """
        try:
            import pickle
            
            model_state = {
                "base_model_state": {
                    "player_states": dict(self.player_states),
                    "filter_states": dict(self.filter_states),
                    "model_metrics": dict(self.model_metrics),
                    "update_counts": dict(self.update_counts),
                    "is_fitted": self.is_fitted
                },
                "prediction_config": self.prediction_config,
                "prediction_performance": self.prediction_performance,
                "enable_fixture_awareness": self.enable_fixture_awareness,
                "enable_seasonality": self.enable_seasonality,
                "filter_config": self.filter_config,
                "config": self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info(f"Model state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            raise PredictiveModelError(f"Model save failed: {e}")
    
    def load_model_state(self, filepath: str) -> None:
        """
        Load complete model state from file.
        
        Args:
            filepath: Path to load model state from
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore base model state
            base_state = model_state["base_model_state"]
            self.player_states = base_state["player_states"]
            self.filter_states = base_state["filter_states"]
            self.model_metrics = base_state["model_metrics"]
            self.update_counts = base_state["update_counts"]
            self.is_fitted = base_state["is_fitted"]
            
            # Restore prediction configuration
            self.prediction_config = model_state["prediction_config"]
            self.prediction_performance = model_state["prediction_performance"]
            self.enable_fixture_awareness = model_state["enable_fixture_awareness"]
            self.enable_seasonality = model_state["enable_seasonality"]
            
            # Reinitialize state predictor if model is fitted
            if self.is_fitted:
                self.state_predictor = StatePredictor(
                    kalman_model=self,
                    config=self.prediction_config,
                    enable_fixture_awareness=self.enable_fixture_awareness,
                    enable_seasonality=self.enable_seasonality,
                    session=self.session
                )
            
            logger.info(f"Model state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            raise PredictiveModelError(f"Model load failed: {e}")


class ModelEnsemble:
    """
    Ensemble of multiple predictive models for robust forecasting.
    
    This class combines predictions from multiple PredictivePlayerModel instances
    to provide more robust and accurate forecasts with improved uncertainty
    quantification.
    """
    
    def __init__(
        self,
        models: List[PredictivePlayerModel],
        weights: Optional[List[float]] = None,
        combination_method: str = "weighted_average"
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of fitted predictive models
            weights: Model weights (equal weights if None)
            combination_method: Method for combining predictions
        """
        if not models:
            raise ValueError("At least one model required for ensemble")
        
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.combination_method = combination_method
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        logger.info(f"Initialized ModelEnsemble with {len(models)} models")
    
    def predict_player_performance(
        self,
        player_id: int,
        gameweeks_ahead: int = 3,
        **kwargs
    ) -> PredictionResult:
        """
        Generate ensemble prediction for a player.
        
        Args:
            player_id: Player ID
            gameweeks_ahead: Prediction horizon
            **kwargs: Additional prediction parameters
            
        Returns:
            Combined prediction result
        """
        try:
            # Get predictions from all models
            individual_predictions = []
            for model in self.models:
                try:
                    pred = model.predict_player_performance(
                        player_id=player_id,
                        gameweeks_ahead=gameweeks_ahead,
                        **kwargs
                    )
                    individual_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Model prediction failed: {e}")
                    continue
            
            if not individual_predictions:
                raise PredictiveModelError("All ensemble models failed to predict")
            
            # Combine predictions
            if self.combination_method == "weighted_average":
                return self._weighted_average_combination(individual_predictions)
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed for player {player_id}: {e}")
            raise PredictiveModelError(f"Ensemble prediction failed: {e}")
    
    def _weighted_average_combination(
        self,
        predictions: List[PredictionResult]
    ) -> PredictionResult:
        """Combine predictions using weighted averaging."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Use weights only for available predictions
        active_weights = self.weights[:len(predictions)]
        active_weights = [w / sum(active_weights) for w in active_weights]
        
        # Combine expected states
        combined_state = sum(w * pred.expected_state for w, pred in zip(active_weights, predictions))
        
        # Combine state covariances (approximate)
        combined_state_cov = sum(w * pred.state_covariance for w, pred in zip(active_weights, predictions))
        
        # Add between-model variance
        state_mean_diff_cov = sum(
            w * jnp.outer(pred.expected_state - combined_state, pred.expected_state - combined_state)
            for w, pred in zip(active_weights, predictions)
        )
        combined_state_cov += state_mean_diff_cov
        
        # Combine expected performance
        combined_performance = sum(w * pred.expected_performance for w, pred in zip(active_weights, predictions))
        
        # Combine performance variance
        combined_perf_var = sum(w * pred.performance_variance for w, pred in zip(active_weights, predictions))
        
        # Combine risk metrics (if available)
        risk_metrics = None
        if all(pred.risk_metrics for pred in predictions):
            combined_expected = sum(w * pred.risk_metrics.expected_value for w, pred in zip(active_weights, predictions))
            combined_variance = sum(w * pred.risk_metrics.variance for w, pred in zip(active_weights, predictions))
            combined_std = np.sqrt(combined_variance)
            
            # Simple combination of other metrics
            combined_var = sum(w * pred.risk_metrics.value_at_risk for w, pred in zip(active_weights, predictions))
            combined_cvar = sum(w * pred.risk_metrics.conditional_var for w, pred in zip(active_weights, predictions))
            combined_sharpe = sum(w * pred.risk_metrics.sharpe_ratio for w, pred in zip(active_weights, predictions))
            combined_drawdown = sum(w * pred.risk_metrics.max_drawdown for w, pred in zip(active_weights, predictions))
            combined_downside = sum(w * pred.risk_metrics.downside_deviation for w, pred in zip(active_weights, predictions))
            
            risk_metrics = RiskMetrics(
                expected_value=combined_expected,
                variance=combined_variance,
                standard_deviation=combined_std,
                value_at_risk=combined_var,
                conditional_var=combined_cvar,
                sharpe_ratio=combined_sharpe,
                max_drawdown=combined_drawdown,
                downside_deviation=combined_downside
            )
        
        # Use first prediction as template
        template = predictions[0]
        
        return PredictionResult(
            player_id=template.player_id,
            gameweeks_ahead=template.gameweeks_ahead,
            prediction_date=datetime.now(timezone.utc).isoformat(),
            expected_state=combined_state,
            state_covariance=combined_state_cov,
            expected_performance=combined_performance,
            performance_variance=combined_perf_var,
            prediction_intervals={},  # Would need to recalculate
            risk_metrics=risk_metrics,
            fixture_adjustments=template.fixture_adjustments,
            seasonality_factors=template.seasonality_factors,
            model_diagnostics={
                "ensemble_size": len(predictions),
                "combination_method": self.combination_method,
                "active_weights": active_weights
            }
        )
    
    def get_expected_points(
        self,
        player_ids: List[int],
        gameweeks: int = 3
    ) -> Dict[int, float]:
        """Get ensemble expected points for multiple players."""
        results = {}
        
        for player_id in player_ids:
            try:
                pred = self.predict_player_performance(player_id, gameweeks)
                results[player_id] = pred.get_expected_points()
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for player {player_id}: {e}")
                continue
        
        return results