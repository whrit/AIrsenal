"""
Versioned Prediction Utilities

This module extends the existing prediction utilities to work with the model versioning system.
It provides backward-compatible interfaces while adding model versioning capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
from sqlalchemy.orm.session import Session

from airsenal.framework.bpl_interface import (
    get_fitted_team_model,
    get_goal_probabilities_for_fixtures,
)
from airsenal.framework.model_versioning import (
    ModelVersionManager,
    ModelVersioningConfig,
    register_model,
    load_model,
    get_production_model,
)
from airsenal.framework.player_model import (
    BasePlayerModel,
    NumpyroPlayerModel,
    ConjugatePlayerModel,
)
from airsenal.framework.prediction_utils import (
    calc_predicted_points_for_player,
    fit_bonus_points,
    fit_card_points,
    fit_save_points,
    get_all_fitted_player_data,
    make_prediction,
    process_player_data,
)
from airsenal.framework.schema import PlayerPrediction, session
from airsenal.framework.utils import (
    CURRENT_SEASON,
    NEXT_GAMEWEEK,
    get_fixtures_for_gameweek,
    list_players,
)

logger = logging.getLogger(__name__)


class VersionedPredictionManager:
    """Manager for predictions using versioned models"""
    
    def __init__(self, config: Optional[ModelVersioningConfig] = None, dbsession: Session = session):
        self.config = config or ModelVersioningConfig()
        self.model_manager = ModelVersionManager(config, dbsession)
        self.dbsession = dbsession
    
    def fit_and_register_player_model(
        self,
        position: str,
        season: str,
        gameweek: int,
        model: BasePlayerModel | None = None,
        version: str | None = None,
        training_params: Dict[str, Any] | None = None,
        performance_metrics: Dict[str, float] | None = None,
        notes: str | None = None,
        register_model: bool = True,
    ) -> BasePlayerModel:
        """
        Fit a player model and optionally register it in the versioning system
        """
        if model is None:
            model = ConjugatePlayerModel()
        
        if version is None:
            from datetime import datetime
            version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Fit the model using existing utilities
        logger.info(f"Fitting {position} player model...")
        data = process_player_data(position, season, gameweek, self.dbsession)
        fitted_model = model.fit(data)
        
        # Register the model if requested
        if register_model:
            model_name = f"player_model_{position.lower()}"
            
            # Calculate basic performance metrics if not provided
            if performance_metrics is None:
                performance_metrics = self._calculate_player_model_metrics(
                    fitted_model, data
                )
            
            logger.info(f"Registering {model_name} version {version}")
            self.model_manager.register_model_version(
                fitted_model,
                model_name,
                version,
                training_params=training_params or {"position": position, "season": season, "gameweek": gameweek},
                feature_set=f"Player performance data for {position} players",
                performance_metrics=performance_metrics,
                notes=notes,
            )
        
        return fitted_model
    
    def fit_and_register_team_model(
        self,
        season: str,
        gameweek: int,
        model: Any = None,
        version: str | None = None,
        model_args: Dict[str, Any] | None = None,
        performance_metrics: Dict[str, float] | None = None,
        notes: str | None = None,
        register_model: bool = True,
    ) -> Any:
        """
        Fit a team model and optionally register it in the versioning system
        """
        if version is None:
            from datetime import datetime
            version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if model_args is None:
            model_args = {"epsilon": 0.0}
        
        # Fit the model using existing utilities
        logger.info("Fitting team model...")
        fitted_model = get_fitted_team_model(
            season=season,
            gameweek=gameweek,
            dbsession=self.dbsession,
            model=model,
            **model_args,
        )
        
        # Register the model if requested
        if register_model:
            model_name = "team_model"
            model_type = type(fitted_model).__name__
            
            logger.info(f"Registering {model_name} ({model_type}) version {version}")
            self.model_manager.register_model_version(
                fitted_model,
                model_name,
                version,
                training_params=model_args,
                feature_set="Team performance and FIFA ratings data",
                performance_metrics=performance_metrics,
                notes=notes or f"Team model using {model_type}",
            )
        
        return fitted_model
    
    def run_versioned_predictions(
        self,
        gw_range: List[int],
        season: str = CURRENT_SEASON,
        player_model_versions: Dict[str, str] | None = None,
        team_model_version: str | None = None,
        use_production_models: bool = False,
        include_bonus: bool = True,
        include_cards: bool = True,
        include_saves: bool = True,
        tag_prefix: str | None = None,
        experiment_name: str | None = None,
    ) -> str:
        """
        Run predictions using specific model versions or production models
        
        Args:
            gw_range: List of gameweeks to predict
            season: Season to predict for
            player_model_versions: Dict mapping position to model version (e.g., {"gk": "v1.0.0"})
            team_model_version: Specific team model version to use
            use_production_models: If True, use production models instead of specific versions
            include_bonus: Include bonus points in predictions
            include_cards: Include card points in predictions
            include_saves: Include save points in predictions
            tag_prefix: Prefix for prediction tag
            experiment_name: Name of experiment (for A/B testing)
        
        Returns:
            Prediction tag for retrieving results
        """
        
        # Generate prediction tag
        tag = (tag_prefix or "") + str(uuid4())
        if experiment_name:
            tag = f"{experiment_name}_{tag}"
        
        logger.info(f"Running versioned predictions with tag: {tag}")
        
        # Load models
        if use_production_models:
            logger.info("Using production models")
            team_model = get_production_model("team_model")
            player_models = {
                pos: get_production_model(f"player_model_{pos.lower()}")
                for pos in ["GK", "DEF", "MID", "FWD"]
            }
        else:
            # Load team model
            if team_model_version:
                team_model = load_model("team_model", team_model_version)
                logger.info(f"Loaded team model version {team_model_version}")
            else:
                # Fall back to latest
                team_model = load_model("team_model")
                logger.info("Loaded latest team model")
            
            # Load player models
            player_models = {}
            for pos in ["GK", "DEF", "MID", "FWD"]:
                model_name = f"player_model_{pos.lower()}"
                if player_model_versions and pos.lower() in player_model_versions:
                    version = player_model_versions[pos.lower()]
                    player_models[pos] = load_model(model_name, version)
                    logger.info(f"Loaded {pos} model version {version}")
                else:
                    # Fall back to latest
                    player_models[pos] = load_model(model_name)
                    logger.info(f"Loaded latest {pos} model")
        
        # Get fixture probabilities using the loaded team model
        logger.info("Calculating fixture score probabilities...")
        fixtures = get_fixtures_for_gameweek(gw_range, season=season, dbsession=self.dbsession)
        fixture_goal_probs = get_goal_probabilities_for_fixtures(fixtures, team_model, max_goals=10)
        
        # Convert loaded models to expected format
        df_player = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            model = player_models[pos]
            probs = model.get_probs()
            df = pd.DataFrame(probs)
            df["pos"] = pos
            df = df.rename(columns={"index": "player_id"}).sort_values("player_id").set_index("player_id")
            df_player[pos] = df
        
        # Get additional models
        if include_bonus:
            df_bonus = fit_bonus_points(gameweek=gw_range[0], season=season)
        else:
            df_bonus = None
            
        if include_saves:
            df_saves = fit_save_points(gameweek=gw_range[0], season=season)
        else:
            df_saves = None
            
        if include_cards:
            df_cards = fit_card_points(gameweek=gw_range[0], season=season)
        else:
            df_cards = None
        
        # Run predictions for all players
        players = list_players(season=season, gameweek=gw_range[0], dbsession=self.dbsession)
        
        for player in players:
            predictions = calc_predicted_points_for_player(
                player,
                fixture_goal_probs,
                df_player,
                df_bonus,
                df_saves,
                df_cards,
                season,
                gw_range=gw_range,
                tag=tag,
                dbsession=self.dbsession,
            )
            for pred in predictions:
                self.dbsession.add(pred)
        
        self.dbsession.commit()
        logger.info(f"Completed versioned predictions with tag: {tag}")
        
        return tag
    
    def run_model_comparison(
        self,
        model_name: str,
        versions: List[str],
        gw_range: List[int],
        season: str = CURRENT_SEASON,
        sample_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compare different model versions on the same prediction task
        
        Args:
            model_name: Name of the model to compare (e.g., "player_model_fwd")
            versions: List of versions to compare
            gw_range: Gameweeks to predict
            season: Season to predict for
            sample_size: Fraction of players to use for comparison (for faster testing)
        
        Returns:
            Dictionary with comparison results
        """
        
        logger.info(f"Comparing {len(versions)} versions of {model_name}")
        
        comparison_results = {
            "model_name": model_name,
            "versions": versions,
            "gw_range": gw_range,
            "season": season,
            "tags": {},
            "summary": {},
        }
        
        # Run predictions for each version
        for version in versions:
            logger.info(f"Running predictions for {model_name} version {version}")
            
            # Create version-specific model configuration
            if "player_model" in model_name:
                position = model_name.split("_")[-1].upper()
                player_model_versions = {position.lower(): version}
                team_model_version = None
            else:
                player_model_versions = None
                team_model_version = version
            
            tag = self.run_versioned_predictions(
                gw_range=gw_range,
                season=season,
                player_model_versions=player_model_versions,
                team_model_version=team_model_version,
                tag_prefix=f"comparison_{model_name}_v{version}_",
                experiment_name=f"comparison_{model_name}",
            )
            
            comparison_results["tags"][version] = tag
        
        logger.info(f"Model comparison completed for {model_name}")
        return comparison_results
    
    def _calculate_player_model_metrics(
        self,
        model: BasePlayerModel,
        data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate basic performance metrics for a player model"""
        
        # This is a simplified implementation
        # In practice, you would want more sophisticated validation
        metrics = {}
        
        try:
            # Get model probabilities
            probs = model.get_probs()
            
            # Basic sanity checks
            prob_score = probs.get("prob_score", [])
            prob_assist = probs.get("prob_assist", [])
            prob_neither = probs.get("prob_neither", [])
            
            if len(prob_score) > 0:
                # Check if probabilities sum to approximately 1
                total_probs = [s + a + n for s, a, n in zip(prob_score, prob_assist, prob_neither)]
                avg_total_prob = sum(total_probs) / len(total_probs)
                metrics["avg_probability_sum"] = avg_total_prob
                
                # Basic distribution metrics
                metrics["avg_prob_score"] = sum(prob_score) / len(prob_score)
                metrics["avg_prob_assist"] = sum(prob_assist) / len(prob_assist)
                metrics["avg_prob_neither"] = sum(prob_neither) / len(prob_neither)
                
                # Validation score (how close to 1 are the probability sums)
                metrics["probability_consistency"] = 1.0 - abs(1.0 - avg_total_prob)
            
        except Exception as e:
            logger.warning(f"Could not calculate model metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics


# Backward compatibility functions
def fit_versioned_player_data(
    position: str,
    season: str,
    gameweek: int,
    model: BasePlayerModel | None = None,
    register_model: bool = False,
    version: str | None = None,
    dbsession: Session = session,
) -> pd.DataFrame:
    """
    Fit player data with optional model versioning
    
    This function extends the original fit_player_data to optionally register models
    """
    manager = VersionedPredictionManager(dbsession=dbsession)
    
    if register_model:
        fitted_model = manager.fit_and_register_player_model(
            position, season, gameweek, model, version
        )
    else:
        # Use existing function for backward compatibility
        from airsenal.framework.prediction_utils import fit_player_data
        return fit_player_data(position, season, gameweek, model, dbsession)
    
    # Convert to expected DataFrame format
    probs = fitted_model.get_probs()
    df = pd.DataFrame(probs)
    df["pos"] = position
    return df.rename(columns={"index": "player_id"}).sort_values("player_id").set_index("player_id")


def make_versioned_predictedscore_table(
    gw_range: List[int] | None = None,
    season: str = CURRENT_SEASON,
    use_versioned_models: bool = False,
    player_model_versions: Dict[str, str] | None = None,
    team_model_version: str | None = None,
    use_production_models: bool = False,
    experiment_name: str | None = None,
    **kwargs,
) -> str:
    """
    Enhanced version of make_predictedscore_table with model versioning support
    
    Args:
        gw_range: Gameweeks to predict
        season: Season to predict for
        use_versioned_models: Whether to use versioned models
        player_model_versions: Specific player model versions to use
        team_model_version: Specific team model version to use
        use_production_models: Whether to use production models
        experiment_name: Name for the experiment/comparison
        **kwargs: Additional arguments passed to original function
    
    Returns:
        Prediction tag
    """
    
    if use_versioned_models:
        manager = VersionedPredictionManager()
        return manager.run_versioned_predictions(
            gw_range=gw_range or list(range(NEXT_GAMEWEEK, NEXT_GAMEWEEK + 3)),
            season=season,
            player_model_versions=player_model_versions,
            team_model_version=team_model_version,
            use_production_models=use_production_models,
            experiment_name=experiment_name,
            **{k: v for k, v in kwargs.items() if k in [
                "include_bonus", "include_cards", "include_saves", "tag_prefix"
            ]},
        )
    else:
        # Fall back to original function
        from airsenal.scripts.fill_predictedscore_table import make_predictedscore_table
        return make_predictedscore_table(gw_range, season, **kwargs)